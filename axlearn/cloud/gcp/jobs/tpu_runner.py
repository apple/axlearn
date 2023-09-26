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

    # Simple launch. Everything after -- is treated as the command.
    axlearn gcp tpu start -- python3 my_script.py

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
import pathlib
import subprocess
import time
from typing import Dict, Optional, Sequence

from absl import app, flags, logging

from axlearn.cloud.common.bundler import DockerBundler, bundler_flags, get_bundler_config
from axlearn.cloud.common.utils import (
    configure_logging,
    generate_job_name,
    parse_action,
    parse_kv_flags,
)
from axlearn.cloud.gcp.bundler import GCSTarBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import TPUJob, docker_command
from axlearn.cloud.gcp.scopes import DEFAULT_TPU_SCOPES
from axlearn.cloud.gcp.tpu import (
    _qrm_resource,
    _tpu_resource,
    create_tpu,
    delete_tpu,
    format_tpu_info,
    get_queued_tpu_node,
    get_queued_tpu_node_status,
    get_tpu_node,
    get_tpu_node_status,
    infer_tpu_workers,
    list_tpu_info,
)
from axlearn.cloud.gcp.utils import catch_auth, common_flags, get_credentials, running_from_vm
from axlearn.cloud.gcp.vertexai_tensorboard import (
    VertexAITensorboardUploader,
    is_vertexai_tensorboard_configured,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.liveness_monitor import LivenessMonitor

FLAGS = flags.FLAGS
_COMMAND_SESSION_NAME = "command"


def launch_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bundler_flags(flag_values=flag_values)
    flag_values.set_default("project", gcp_settings("project", required=False))
    flag_values.set_default("zone", gcp_settings("zone", required=False))
    flag_values.set_default("bundler_type", GCSTarBundler.TYPE)
    # Note: don't use generate_job_name() here, as the VM may not have $USER.
    flags.DEFINE_string("name", None, "Job name.", flag_values=flag_values)
    flags.DEFINE_string("tpu_type", None, "Type of TPU to start.", flag_values=flag_values)
    flags.DEFINE_integer(
        "num_slices",
        1,
        "The number of slices of specified TPU type to start.",
        flag_values=flag_values,
    )
    flags.DEFINE_integer(
        "max_tries", 10, "Max attempts to launch the job.", flag_values=flag_values
    )
    flags.DEFINE_integer(
        "retry_interval", 60, "Interval in seconds between tries.", flag_values=flag_values
    )
    flags.DEFINE_multi_string(
        "env", [], "Env var in the format key:value.", flag_values=flag_values
    )
    flags.DEFINE_string(
        "output_dir",
        None,
        "If specified, the directory to store outputs (such as logs).",
        flag_values=flag_values,
    )
    flags.DEFINE_string(
        "service_account",
        None,
        "If specified, will run job as the service account. "
        "Otherwise will fallback to application-default credentials.",
        flag_values=flag_values,
    )


class TPURunnerJob(TPUJob):
    """Launches and monitors a TPU job."""

    @config_class
    class Config(TPUJob.Config):
        """Configures TPURunnerJob."""

        # Remote output directory. Should be a gs:// path.
        # TODO(markblee): Support other cloud storages using tf.io.
        output_dir: Required[str] = REQUIRED
        # Optional env vars to set.
        env_vars: Dict[str, str] = {}
        # Interval to poll status.
        status_interval_seconds: float = 30
        # A monitor that checks if a job is stalled/stuck.
        monitor: Optional[LivenessMonitor.Config] = None
        # Optional VertexAI Tensorboard Uploader.
        vertexai_tb_uploader: Optional[VertexAITensorboardUploader] = None

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg = super().from_flags(fv, **kwargs)
        cfg.name = cfg.name or generate_job_name()
        cfg.output_dir = (
            cfg.output_dir or f"gs://{gcp_settings('ttl_bucket')}/axlearn/jobs/{cfg.name}"
        )
        cfg.bundler = get_bundler_config(bundler_type=fv.bundler_type, spec=fv.bundler_spec)
        return cfg

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        cfg = self.config
        # Output directory on the TPU-VM.
        self._output_dir = pathlib.Path("/output") / cfg.name
        # Per-process status file.
        self._status_file = self._output_dir / "status"
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

    def _sync_outputs(self, *, session: str, src: str, dst: str, interval_s: int):
        """Starts a screen session to sync outputs to gs."""
        logging.info("Starting log sync...")
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

    def _wrap(self, cmd: str, *, env: Optional[Dict[str, str]] = None):
        """Wraps the command with env vars, and docker run if using docker bundler."""
        cfg: TPURunnerJob.Config = self.config
        if self._bundler.TYPE == DockerBundler.TYPE:
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

    def _prepare_env(self) -> Dict[str, str]:
        """Returns env vars to use in the command."""
        logging.info("Preparing env...")
        cfg = self.config
        # Make a copy of env vars.
        env_vars = {**cfg.env_vars}
        # Prepare environment variables for multislice training.
        # TODO(markblee,tom_gunter): Delete this when no longer necessary.
        if cfg.num_slices > 1:
            logging.info("Preparing environment on VMs for multislice training...")
            master_tpu_node = get_tpu_node(
                f"{cfg.name}-0",
                resource=_tpu_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
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
            f"sleep $((1 + $RANDOM % 30)) && {install_cmd} && {pip_freeze_cmd} | "
            f"tee -a {self._run_log}",
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
        credentials = self._get_job_credentials(DEFAULT_TPU_SCOPES)
        delete_tpu(cfg.name, credentials=credentials)
        create_tpu(
            name=cfg.name,
            tpu_type=cfg.tpu_type,
            bundler_type=self._bundler.TYPE,
            credentials=credentials,
            num_slices=cfg.num_slices,
            service_account=cfg.service_account,
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
        # Install the bundle.
        self._install_bundle()
        # Prepare command environment variables.
        env_vars = self._prepare_env()
        # Start syncing run log to GS.
        # TODO(markblee): Sync XLA logs.
        self._sync_outputs(
            session="sync_outputs",
            src=f"{self._output_dir}/",
            dst=f"{cfg.output_dir}/output/$HOSTNAME/",
            interval_s=60,
        )
        # Possibly wrap with docker run.
        cmd = self._wrap(cfg.command, env=env_vars)
        # Set env vars, run the command and pipe outputs to run log.
        # Depending on command returncode, emit either success or failure flag.
        # Note that we use PIPESTATUS[0] to check the returncode of the first command in the pipe.
        cmd = f"""mkdir -p {self._output_dir}; echo "Starting command..." >> {self._run_log};
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
        logging.info("Copying outputs from %s...", self._output_dir)
        self._copy_outputs(src=f"{self._output_dir}/*", dst=f"{cfg.output_dir}/output/$HOSTNAME/")
        logging.info("Start deleting TPU %s...", cfg.name)
        delete_tpu(cfg.name, credentials=self._get_job_credentials(DEFAULT_TPU_SCOPES))
        logging.info("Finished deleting %s.", cfg.name)

    def _num_workers(self) -> int:
        cfg: TPURunnerJob.Config = self.config
        return infer_tpu_workers(cfg.tpu_type) * cfg.num_slices

    def _get_status(self) -> Status:
        """Attempts to infer the status of the job.

        This checks all workers' statuses. We report a status only when all workers report and agree
        on the same status. Otherwise, we report UNKNOWN.

        TODO(markblee): If TPU util is low or idle core duration is high, return IDLE.
        """
        cfg: TPURunnerJob.Config = self.config
        credentials = self._get_job_credentials(DEFAULT_TPU_SCOPES)

        # If no TPU, or TPU not fully booted, return NOT_STARTED.
        num_booted = 0
        if cfg.num_slices > 1:
            node = get_queued_tpu_node(cfg.name, _qrm_resource(credentials))
            num_vms = cfg.num_slices * infer_tpu_workers(cfg.tpu_type)
            if node is not None:
                num_booted = get_queued_tpu_node_status(cfg.name, node=node)["num_booted"]
        else:
            node = get_tpu_node(cfg.name, _tpu_resource(credentials))
            num_vms = infer_tpu_workers(cfg.tpu_type)
            if node is not None:
                num_booted = get_tpu_node_status(cfg.name, node=node)["num_booted"]
        if num_booted < num_vms:
            logging.info("TPU doesn't exist or not fully booted: %d/%d", num_booted, num_vms)
            return TPURunnerJob.Status.NOT_STARTED

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
        valid_statuses = set(status.name for status in TPURunnerJob.Status)
        for proc in procs:
            if proc.returncode == 0:
                for line in proc.stdout.split("\n"):  # pytype: disable=attribute-error
                    if line.strip().startswith("STATUS_"):
                        _, worker_id, status = line.strip().split("_", maxsplit=2)
                        if status in valid_statuses:
                            statuses[worker_id] = TPURunnerJob.Status[status]
            else:
                logging.warning(
                    "Failed to get job status. stdout=%s, stderr=%s", proc.stdout, proc.stderr
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
            # TPU-VM doesn't exist -- create it and launch the command.
            elif status == TPURunnerJob.Status.NOT_STARTED:
                self._start()
                logging.info("TPU-VMs have started.")
            # TPU-VM is ready but not running command -- start the command.
            elif status == TPURunnerJob.Status.NOT_RUNNING:
                logging.info("Job is not running. Running the command...")
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


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["start", "stop", "list"])

    if action == "list":
        print(format_tpu_info(list_tpu_info(get_credentials())))
        return

    cfg = TPURunnerJob.from_flags(flag_values)

    if action == "start":
        if not flag_values.tpu_type:
            raise app.UsageError("--tpu_type is required.")

        command = " ".join(argv[2:])
        if not command:
            raise app.UsageError("Command is required.")

        default_env = dict(
            # Use a large refresh to mitigate DNS timeout issues until tf>2.12 upgrade.
            GCS_RESOLVE_REFRESH_SECS=600,
            TPU_TYPE=flag_values.tpu_type,
            NUM_TPU_SLICES=flag_values.num_slices,
            XLA_FLAGS=f"--xla_dump_to=/output/{flag_values.name}/xla",
            TF_CPP_MIN_LOG_LEVEL=0,
        )
        vertexai_tb_uploader = None
        if is_vertexai_tensorboard_configured():
            vertexai_tb_uploader = VertexAITensorboardUploader.default_config()

        cfg.set(
            env_vars={**default_env, **parse_kv_flags(flag_values.env)},
            command=command,
            vertexai_tb_uploader=vertexai_tb_uploader,
        )
        job: TPURunnerJob = cfg.instantiate()
        job.execute()
    elif action == "stop":
        job: TPURunnerJob = cfg.set(command="").instantiate()
        job._delete()  # pylint: disable=protected-access
    else:
        # Unreachable -- `parse_action` will handle validation.
        raise app.UsageError(f"Unknown action {action}")


if __name__ == "__main__":
    launch_flags()
    configure_logging(logging.INFO)
    app.run(main)
