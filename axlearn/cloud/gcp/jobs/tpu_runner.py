# Copyright © 2023 Apple Inc.

"""Creates a TPU-VM and executes the given command on it.

By default, the script will monitor the status of the job and restart/delete if necessary.

There are several design considerations:
1. No long-running SSH connections. Instead of keeping the connection open until command completes,
    we run commands in detached mode to avoid failing the job when connection inevitably breaks.
    Each detached command runs in a named screen session, where we can easily monitor whether the
    command is still running, or can terminate the session as desired (i.e., soft-restart the job).
2. The script is (mostly) pre-emptible. In other words, we rely on minimal local state. If the
    script is run, terminated, and re-run, it will resume monitoring the job instead of re-bundling
    and launching the command. This makes it suitable for launching from bastion VMs, where the
    bastion VM can fail without terminating all associated jobs.

Notes:
- While the script is mostly pre-emptible, it is not fully idempotent. You may get unexpected
    behavior if launching the script multiple times concurrently, although under normal
    circumstances this shouldn't happen.
- In most cases you may want to launch the TPU job via bastion (see bastion_vm), which comes with
    queueing and scheduling. However, running this script directly can be useful for debugging.
    To launch via bastion, please refer to the `gcp launch` command.

Possible actions: [start|stop|list]

    Start: starts (or resumes monitoring) the TPU job.
    Stop: stops the TPU job and tears down resources.
    List: lists all TPUs.

Examples:

    # Simple launch on v4-8. Everything after -- is treated as the command.
    axlearn gcp tpu start --instance_type=tpu-v4-8 -- python3 my_script.py

    # Launch with env and retries.
    axlearn gcp tpu start \
        --max_tries=3 --env=MY_ENV=1 -- python3 my_script.py

    # Launch with docker.
    axlearn gcp tpu start \
        --bundler_type=artifactregistry \
        --bundler_spec=repo=my-repo \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=build_arg1=my-build-arg ...

    # List running TPUs.
    axlearn gcp tpu list

    # Stop and teardown the job.
    axlearn gcp tpu stop --name=my-job

"""
# TODO(markblee):
# - Emit hostname of worker 0 so we can point users to the main log.
# - On failure, attempt to point users to the right worker log.
# - Monitor idle core duration and automatically terminate.

import enum
import os
import pathlib
import shlex
import subprocess
import time
from collections.abc import Sequence
from typing import Optional

from absl import app, flags, logging
from googleapiclient import errors

from axlearn.cloud.common.bundler import BaseDockerBundler, get_bundler_config
from axlearn.cloud.common.utils import (
    configure_logging,
    generate_job_name,
    parse_action,
    parse_kv_flags,
)
from axlearn.cloud.gcp.bundler import GCSTarBundler, with_tpu_extras
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import TPUQRMJob, docker_command
from axlearn.cloud.gcp.jobs import runner_utils
from axlearn.cloud.gcp.jobs.tpu_utils import get_default_env
from axlearn.cloud.gcp.scopes import DEFAULT_TPU_SCOPES
from axlearn.cloud.gcp.tpu import (
    create_queued_tpu,
    delete_queued_tpu,
    get_queued_tpu_node,
    get_queued_tpu_node_status,
    get_tpu_node,
    infer_tpu_type,
    infer_tpu_workers,
    list_tpu_info,
    qrm_resource,
    tpu_info_table,
    tpu_resource,
)
from axlearn.cloud.gcp.utils import catch_auth, get_credentials, running_from_vm
from axlearn.cloud.gcp.vertexai_tensorboard import (
    VertexAITensorboardUploader,
    is_vertexai_tensorboard_configured,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.liveness_monitor import LivenessMonitor

FLAGS = flags.FLAGS
_COMMAND_SESSION_NAME = "command"


def _infer_reservation(node: Optional[dict]) -> Optional[bool]:
    """Infers reservation given a QRM node."""
    return (node or {}).get("guaranteed", {}).get("reserved", None)


# TODO(markblee): Use composition instead of inheritance for TPUQRMJob.
# This can help consolidate TPURunnerJob and CPURunnerJob by switching the inner implementation.
class TPURunnerJob(TPUQRMJob):
    """Launches and monitors a TPU job."""

    @config_class
    class Config(TPUQRMJob.Config):
        """Configures TPURunnerJob."""

        # Remote output directory. Should be a gs:// path.
        # TODO(markblee): Support other cloud storages using tf.io.
        output_dir: Required[str] = REQUIRED
        # Optional env vars to set.
        env_vars: dict[str, str] = {}
        # Interval to poll status.
        status_interval_seconds: float = 30
        # A monitor that checks if a job is stalled/stuck.
        monitor: Optional[LivenessMonitor.Config] = None
        # Optional VertexAI Tensorboard Uploader.
        vertexai_tb_uploader: Optional[VertexAITensorboardUploader.Config] = None
        # Whether to enable TPU ICI resiliency.
        # If True, the job will persist through some types of network failure,
        # but with degraded performance.
        # If None, we leave it to GCP to determine whether it's appropriate for the
        # requested TPU topology.
        enable_tpu_ici_resiliency: Optional[bool] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_multi_string("env", [], "Env var in the format key:value.", **common_kwargs)
        flags.DEFINE_boolean(
            "enable_tpu_ici_resiliency",
            None,
            "Whether to enable TPU ICI resiliency. If None, the decision is left to GCP, as "
            "not all TPU types support this flag.",
            **common_kwargs,
        )
        # TODO(markblee): Remove these, which are for backwards compat with old client.
        flags.DEFINE_alias("tpu_type", "instance_type", flag_values=fv)
        flags.DEFINE_alias("num_slices", "num_replicas", flag_values=fv)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: TPURunnerJob.Config = super().from_flags(fv, **kwargs)
        # NOTE: if running on TPU, name is required as there may not be a $USER.
        cfg.name = cfg.name or generate_job_name()
        cfg.max_tries = cfg.max_tries or 10
        cfg.retry_interval = cfg.retry_interval or 60
        cfg.env_vars = {**cfg.env_vars, **parse_kv_flags(fv.env)}
        cfg.output_dir = (
            cfg.output_dir or f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{cfg.name}"
        )
        cfg.bundler = get_bundler_config(
            bundler_type=fv.bundler_type or GCSTarBundler.TYPE,
            spec=fv.bundler_spec,
            fv=fv,
        )
        return with_tpu_training_defaults(cfg, flag_values=fv)

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        cfg = self.config
        # Output directory on the TPU-VM.
        self._output_dir = pathlib.Path("/output") / cfg.name
        # Per-process status file.
        self._status_file = self._output_dir / "status"
        # Local run log.
        self._run_log = self._output_dir / "run.log"

        if cfg.monitor:
            self._monitor: LivenessMonitor = cfg.monitor.instantiate()
        else:
            self._monitor = None
        # Log sync process.
        if cfg.vertexai_tb_uploader:
            self._vertexai_tb_uploader: VertexAITensorboardUploader = cfg.vertexai_tb_uploader.set(
                summary_dir=cfg.output_dir
            ).instantiate()
        else:
            self._vertexai_tb_uploader = None
        self._qrm_resource = None
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)

    def _sync_outputs(self, *, session: str, src: str, dst: str, interval_s: int):
        """Starts a screen session to sync outputs to gs."""
        logging.info("Starting log sync %s -> %s...", src, dst)
        self._execute_remote_cmd(
            f"while true; do gsutil -m rsync -r {src} {dst}; sleep {interval_s}; done",
            detached_session=session,
            shell=True,
        )
        logging.info("Log sync started.")

    def _copy_outputs(self, *, src: str, dst: str):
        """Copies outputs to gs, blocking until complete."""
        # Set check=False, since _delete can be invoked prior to outputs being written.
        self._execute_remote_cmd(f"gsutil cp -r {src} {dst}", shell=True, check=False)

    def _wrap(self, cmd: str, *, env: Optional[dict[str, str]] = None):
        """Wraps the command with env vars, and docker run if using docker bundler."""
        cfg: TPURunnerJob.Config = self.config
        if isinstance(self._bundler, BaseDockerBundler):
            cmd = docker_command(
                cmd,
                image=self._bundler.id(cfg.name),
                env=env,
                volumes={"/output": "/output", "/tmp": "/tmp", "/etc": "/etc"},
            )
        # Format env vars for the command line.
        if env:
            cmd = f"{' '.join([f'{key}={value}' for key, value in env.items()])} {cmd}"
        return cmd.strip()

    def _prepare_env(self) -> dict[str, str]:
        """Returns env vars to use in the command."""
        logging.info("Preparing env...")
        cfg: TPURunnerJob.Config = self.config
        # Make a copy of env vars.
        env_vars = {**cfg.env_vars}
        # Prepare environment variables for multislice training.
        # TODO(markblee,tom_gunter): Delete this when no longer necessary.
        if cfg.accelerator.num_replicas > 1:
            logging.info("Preparing environment on VMs for multislice training...")
            # We don't use `self._call_qrm_api` here because `get_queued_tpu_node` does not seem to
            # return 'networkEndpoints'. This is probably acceptable since this call happens once.
            master_tpu_node = get_tpu_node(
                f"{cfg.name}-0",
                tpu_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
            )
            coordinator_address = f"{master_tpu_node['networkEndpoints'][0]['ipAddress']}:8080"
            self._execute_remote_cmd(
                f"echo 'MEGASCALE_COORDINATOR_ADDRESS=\"{coordinator_address}\"' | "
                "sudo tee -a /etc/environment",
            )
            # To propagate the env to docker run.
            env_vars["MEGASCALE_COORDINATOR_ADDRESS"] = coordinator_address
        logging.info("Done preparing env.")
        return env_vars

    def _install_bundle(self):
        """Installs the bundle on remote TPU-VM."""
        cfg: TPURunnerJob.Config = self.config
        logging.info("Installing bundle...")
        # Install the bundle.
        install_cmd = self._bundler.install_command(self._bundler.id(cfg.name))
        pip_freeze_cmd = self._wrap("python3 -m pip freeze")
        self._execute_remote_cmd(
            f"set -o pipefail; mkdir -p {self._output_dir}; "
            f"sleep $((1 + $RANDOM % 30)) && "
            f"({install_cmd} && {pip_freeze_cmd}) 2>&1 | tee -a {self._run_log}",
            shell=True,
        )
        logging.info("Done installing bundle.")

    def _start(self):
        """Provisions TPU-VMs and installs the bundle."""
        cfg: TPURunnerJob.Config = self.config
        # If not on bastion, bundle early so the user can cd away from cwd.
        if not running_from_vm():
            self._bundler.bundle(cfg.name)

        # We delete the TPU to ensure that all state is cleared before a restart.
        # TODO(markblee): We can support killing/restarting a command without recreating the TPU,
        # via killing the screen session, e.g. screen -XS <screen> quit, or killing all processes
        # holding the TPU.
        self._call_qrm_api(delete_queued_tpu, cfg.name)

        tpu_metadata = {}
        if isinstance(self._bundler, BaseDockerBundler):
            tpu_metadata["docker_image"] = self._bundler.id(cfg.name)
        if cfg.enable_tpu_ici_resiliency is not None:
            tpu_metadata["enable_ici_resiliency"] = cfg.enable_tpu_ici_resiliency

        # If running from bastion, a scheduling tier will be specified in env.
        # Tier "0" corresponds to reserved; otherwise we use preemptible.
        # If running locally for testing, we leave as None to respect the `reserved_tpu` configured
        # in the settings file.
        reserved = None
        tier = os.environ.get("BASTION_TIER", None)
        if tier is not None:
            reserved = str(tier) == "0"
            logging.info("Found tier=%s in env. Using reserved=%s", tier, reserved)

        # Create labels for vm tier that can be used to group tpu metrics.
        # In QRM, vm tier can be one of guaranteed, spot, or, bestEffort.
        # The "reserved" label is used for guaranteed instances and
        # "spot" for other instances (e.g. best-effort or spot instances).
        # BASTION_TIER env has presendence over the reserved_tpu.
        if reserved is None:
            reserved = gcp_settings("reserved_tpu", default=False, required=False)
        labels = {"bastion_tier": "reserved" if reserved else "spot"}

        self._call_qrm_api(
            create_queued_tpu,
            name=cfg.name,
            tpu_type=self._tpu_type,
            bundler_type=self._bundler.TYPE,
            num_slices=cfg.accelerator.num_replicas,
            service_account=cfg.service_account,
            labels=labels,
            metadata=tpu_metadata,
            reserved=reserved,
        )

    class Status(enum.Enum):
        """TPU job status."""

        UNKNOWN = "UNKNOWN"
        FAILED = "FAILED"
        SUCCESS = "SUCCESS"
        # The VM itself has not been provisioned.
        NOT_STARTED = "NOT_STARTED"
        # The VM is provisioned, but the command is not running.
        NOT_RUNNING = "NOT_RUNNING"
        # The VM is provisioned and the command is running.
        RUNNING = "RUNNING"
        # The liveness check failed.
        STUCK = "STUCK"
        # The job was rescheduled on a different tier.
        RESCHEDULED = "RESCHEDULED"

    # pylint: disable-next=no-self-use
    def _status_flag(self, status: Status):
        return f"STATUS_${{HOSTNAME}}_{status.name}"

    def _set_status_command(self, status: Status):
        """Returns a command to set the current process status."""
        return (
            f"mkdir -p {self._output_dir} && echo {self._status_flag(status)} > {self._status_file}"
        )

    def _run_command(self):
        """Launches the command on the TPU-VMs."""
        cfg: TPURunnerJob.Config = self.config
        # Start syncing run log to GS.
        # TODO(markblee): Sync XLA logs.
        self._sync_outputs(
            session="sync_outputs",
            src=f"{self._output_dir}/",
            dst=f"{cfg.output_dir}/output/$HOSTNAME/",
            interval_s=60,
        )
        # Install the bundle.
        self._install_bundle()
        # Prepare command environment variables.
        env_vars = self._prepare_env()
        # Possibly wrap with docker run.
        cmd = self._wrap(cfg.command, env=env_vars)
        # Set env vars, run the command and pipe outputs to run log.
        # Depending on command returncode, emit either success or failure flag.
        # Note that we use PIPESTATUS[0] to check the returncode of the first command in the pipe.
        cmd = f"""ulimit -n 100000;
            mkdir -p {self._output_dir}; echo "Starting command..." >> {self._run_log};
            {cmd} 2>&1 | stdbuf -oL sed "s/^/$HOSTNAME: /" | tee -a {self._run_log};
            if [ ${{PIPESTATUS[0]}} -eq 0 ]; then
                echo "Setting status to SUCCESS..." >> {self._run_log};
                {self._set_status_command(TPURunnerJob.Status.SUCCESS)};
            else
                echo "Setting status to FAILED..." >> {self._run_log};
                {self._set_status_command(TPURunnerJob.Status.FAILED)};
            fi
            """
        logging.info("Starting remote command...")
        procs = self._execute_remote_cmd(
            cmd,
            shell=True,
            # Run in a detached session, to avoid a long-running SSH connection.
            detached_session=_COMMAND_SESSION_NAME,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        # TODO(markblee): Sync consolidated stdout/stderr to output dir.
        for proc in procs:
            logging.info("Proc %s has returncode: %s", proc, proc.returncode)
            if proc.returncode != 0:
                raise ValueError(
                    f"Command failed with stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                )
        logging.info("All commands submitted (%s total).", len(procs))
        # Start/reset the monitor. This ensures that grace period (for TPU startup) is reset.
        if self._monitor is not None:
            self._monitor.reset()

    def _delete(self):
        cfg: TPURunnerJob.Config = self.config
        if self._call_qrm_api(get_queued_tpu_node, cfg.name) is None:
            logging.info("TPU %s doesn't exist.", cfg.name)
            return
        logging.info("Copying outputs from %s...", self._output_dir)
        self._copy_outputs(src=f"{self._output_dir}/*", dst=f"{cfg.output_dir}/output/$HOSTNAME/")
        logging.info("Start deleting TPU %s...", cfg.name)
        self._call_qrm_api(delete_queued_tpu, cfg.name)
        logging.info("Finished deleting %s.", cfg.name)

    def _num_workers(self) -> int:
        cfg: TPURunnerJob.Config = self.config
        return infer_tpu_workers(self._tpu_type) * cfg.accelerator.num_replicas

    def _call_qrm_api(self, fn, *args, max_tries: int = 2, **kwargs):
        for i in range(max_tries):
            try:
                if self._qrm_resource is None:
                    logging.info("Building QRM resource...")
                    credentials = self._get_job_credentials(DEFAULT_TPU_SCOPES)
                    self._qrm_resource = qrm_resource(credentials)

                return fn(*args, resource_qrm=self._qrm_resource, **kwargs)
            except (errors.HttpError, OSError) as e:
                logging.warning("QRM API %s call failed with: %s", fn.__name__, e)
                if i < max_tries - 1:
                    logging.info("Will attempt to retry with new connection...")
                if self._qrm_resource is not None:
                    self._qrm_resource.close()  # Also closes http.
                time.sleep(10)

    def _get_status(self) -> Status:
        """Attempts to infer the status of the job.

        This checks all workers' statuses. We report a status only when all workers report and agree
        on the same status. Otherwise, we report UNKNOWN.

        TODO(markblee): If TPU util is low or idle core duration is high, return IDLE.
        """
        cfg: TPURunnerJob.Config = self.config

        # If no TPU, or TPU not fully booted, return NOT_STARTED.
        num_booted = 0
        node = self._call_qrm_api(get_queued_tpu_node, cfg.name)
        num_vms = self._num_workers()
        if node is not None:
            num_booted = get_queued_tpu_node_status(cfg.name, node=node)["num_booted"]
        if num_booted < num_vms:
            logging.info("TPU doesn't exist or not fully booted: %d/%d", num_booted, num_vms)
            return TPURunnerJob.Status.NOT_STARTED

        tier = os.environ.get("BASTION_TIER", 0)
        reservation = _infer_reservation(node)
        # If tier has changed, we may need to recreate the TPUs.
        # Note that in the QRM case, if the TPUs are pre-empted, they will also be recreated with
        # the correct tier/reservation, so it's not necessary to always recreate proactively.
        if runner_utils.should_recreate_job(tier, reservation):
            return TPURunnerJob.Status.RESCHEDULED

        # Probe liveness monitor.
        if self._monitor is not None and self._monitor.started() and not self._monitor.ping():
            return TPURunnerJob.Status.STUCK

        # If screen session is running, return RUNNING.
        # Otherwise, if the process' status flag is set, return the status.
        # Otherwise, return NOT_RUNNING.
        procs = self._execute_remote_cmd(
            f"""sudo screen -wipe;
            if sudo screen -ls {_COMMAND_SESSION_NAME}; then
                echo {self._status_flag(TPURunnerJob.Status.RUNNING)};
            elif [[ -f {self._status_file} ]]; then
                cat {self._status_file};
            else
                echo {self._status_flag(TPURunnerJob.Status.NOT_RUNNING)};
            fi
            """,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Filter out statuses from each worker.
        statuses = {}
        valid_statuses = {status.name for status in TPURunnerJob.Status}
        for proc in procs:
            if proc.returncode == 0:
                for line in proc.stdout.split("\n"):  # pytype: disable=attribute-error
                    if line.strip().startswith("STATUS_"):
                        _, worker_id, status = line.strip().split("_", maxsplit=2)
                        if status in valid_statuses:
                            statuses[worker_id] = TPURunnerJob.Status[status]
            else:
                logging.warning(
                    "Failed to get job status. stdout=%s, stderr=%s",
                    proc.stdout,
                    proc.stderr,
                )

        logging.info("Worker statuses: %s", statuses)
        status_values = set(statuses.values())
        # If any worker failed, whole job failed.
        if TPURunnerJob.Status.FAILED in status_values:
            return TPURunnerJob.Status.FAILED
        # Otherwise, if all workers agree on status, return it.
        if len(statuses) == self._num_workers() and len(status_values) == 1:
            return list(status_values)[0]

        # Unable to infer status (or workers don't all agree), return UNKNOWN.
        return TPURunnerJob.Status.UNKNOWN

    def _execute(self):
        cfg: TPURunnerJob.Config = self.config

        while True:
            status = self._get_status()
            if status == TPURunnerJob.Status.SUCCESS:
                logging.info("Job completed successfully.")
                self._delete()
                return
            # Command is stuck, teardown and raise so we can retry.
            elif status == TPURunnerJob.Status.STUCK:
                self._delete()
                raise ValueError("Job is stuck.")
            # Command failed, teardown and raise so we can retry.
            elif status == TPURunnerJob.Status.FAILED:
                self._delete()
                raise ValueError("Job failed.")
            elif status == TPURunnerJob.Status.RESCHEDULED:
                logging.info("Jobset does not match scheduling tier. Deleting the TPU...")
                self._delete()
            # TPU-VM doesn't exist -- create it and launch the command.
            elif status == TPURunnerJob.Status.NOT_STARTED:
                self._start()
                logging.info("TPU-VMs have started.")
            # TPU-VM is ready but not running command -- start the command.
            elif status == TPURunnerJob.Status.NOT_RUNNING:
                logging.info(
                    "Job is not running. Running the command... "
                    "Logs will be synced to: %s/output/",
                    cfg.output_dir,
                )
                self._run_command()
            # Running, sleep and check back in a bit.
            elif status == TPURunnerJob.Status.RUNNING:
                # Note: in the multislice scenario, there will be num_slices of each worker ID.
                logging.info(
                    "Job is still running...\nTo view logs: gsutil ls %s/output/",
                    cfg.output_dir,
                )
                # Ensure VertexAI Tensorboard Uploader is running.
                if self._vertexai_tb_uploader:
                    self._vertexai_tb_uploader.upload()
                time.sleep(cfg.status_interval_seconds)
            # Job can have an unresolved status in a transitory period where workers have not all
            # completed a step. This is usually resolved by waiting.
            elif status == TPURunnerJob.Status.UNKNOWN:
                logging.info("Job has unknown status. Waiting to see if it resolves itself...")
                time.sleep(cfg.status_interval_seconds)


def with_tpu_training_defaults(
    cfg: TPUQRMJob.Config, *, flag_values: flags.FlagValues
) -> TPUQRMJob.Config:
    """Configures the job with TPU training defaults."""
    default_env = get_default_env(
        tpu_type=infer_tpu_type(flag_values.instance_type),
        num_tpu_slices=flag_values.num_replicas,
        job_name=cfg.name,
    )
    vertexai_tb_uploader = None
    if is_vertexai_tensorboard_configured(flag_values=flag_values):
        vertexai_tb_uploader = VertexAITensorboardUploader.from_flags(flag_values)

    return cfg.set(
        env_vars={**default_env, **cfg.env_vars},
        vertexai_tb_uploader=vertexai_tb_uploader,
        bundler=with_tpu_extras(cfg.bundler),
    )


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["start", "stop", "list"])

    if action == "list":
        print(tpu_info_table(list_tpu_info(tpu_resource(get_credentials()))))
        return

    if action == "start":
        if not flag_values.instance_type:
            raise app.UsageError("--instance_type is required.")

        # Use shlex join so that quoted commands (e.g. 'a && b') retain quotes.
        command = shlex.join(argv[2:])
        if not command:
            raise app.UsageError("Command is required.")

        cfg = TPURunnerJob.from_flags(flag_values)
        job: TPURunnerJob = cfg.set(command=command).instantiate()
        job.execute()
    elif action == "stop":
        flag_values.set_default("instance_type", "tpu")

        if not flag_values.name:
            raise app.UsageError("--name is required.")

        cfg = TPURunnerJob.from_flags(flag_values)
        job: TPURunnerJob = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
    else:
        # Unreachable -- `parse_action` will handle validation.
        raise app.UsageError(f"Unknown action {action}")


if __name__ == "__main__":
    TPURunnerJob.define_flags(FLAGS)
    configure_logging(logging.INFO)
    app.run(main)
