# Copyright © 2023 Apple Inc.

"""Launches a bastion VM on Google Cloud Platform (GCP).

See `axlearn/cloud/common/bastion.py` for bastion details.

Possible actions: [create|delete|start|stop|submit|cancel|history]

    Create: creates the bastion VM, and runs "start" on it.
    Delete: deletes the bastion VM.
    Start: starts the bastion script locally. Typically not intended to be used directly.
    Stop: soft-stops the bastion VM, without deleting it.
    Submit: submits a job spec to the bastion VM for scheduling and execution.
    Cancel: cancels a running job managed by the bastion VM.
    History: prints history of a job or a project id.

Examples:

    # Create and start a bastion.
    #
    # Notes:
    #  - Only docker bundler_type is supported.
    #  - We assume the image is tagged with the same name as the bastion.
    #  - Unless configured in the settings, the default bastion name is <zone>-shared-bastion.
    #
    axlearn gcp bastion create --name=shared-bastion

    # Submit a command to be run by bastion.
    # Use `serialize_jobspec` to write a jobspec.
    axlearn gcp bastion submit --spec=/path/to/spec --name=shared-bastion

    # Check the job status/history.
    axlearn gcp bastion history --name=shared-bastion --job_name=my-job

    # If it is not running, check the project history to see the limit and the queue.
    axlearn gcp bastion history --name=shared-bastion <project_id>

    # Cancel a running job.
    axlearn gcp bastion cancel --job_name=my-job --name=shared-bastion

    # Soft-stop the bastion.
    #
    # Notes:
    # - The command will wait until bastion is stopped.
    # - On next create, bastion will pull the latest image and resume.
    #
    axlearn gcp bastion stop --name=shared-bastion

    # Delete the bastion.
    axlearn gcp bastion delete --name=shared-bastion

    # Build and push a bastion image.
    axlearn gcp bundle --bundler_type=artifactregistry \
        --name=shared-bastion \
        --bundler_spec=image=base \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=target=bastion

To test changes to bastion:

    # 1. Build a custom image.
    axlearn gcp bundle --bundler_type=docker \
        --name=$USER-bastion \
        --bundler_spec=image=base \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=target=bastion

    # 2. Create the bastion VM, if haven't already.
    axlearn gcp bastion create --name=$USER-bastion

    # 3. Submit a test job to $USER-bastion.
    axlearn gcp bastion submit --spec=... --name=$USER-bastion

    # 4. To iterate on changes, soft-stop the bastion and rerun steps 1 and 2.
    axlearn gcp bastion --name=$USER-bastion stop

    # Note: you may find debugging easier by SSHing into the bastion.
    axlearn gcp sshvm $USER-bastion
    tail -n 500 -f /var/tmp/logs/$USER-bastion  # Tail logs.
    docker stop $USER-bastion  # Soft-stop the bastion. Rerun step 2 to start.

    # 5. Once done testing, teardown the bastion.
    axlearn gcp bastion delete --name=$USER-bastion

On "start" vs "create":

    In order to run the bastion on remote compute, "create" does two things:
    1. Creates a remote VM.
    2. Runs "start" on the remote VM.

    In other words, "start" only runs the bastion "locally". This allows us to write the start logic
    in pure Python code (as opposed to remote SSH commands).

"""
# pylint: disable=consider-using-with,too-many-branches,too-many-instance-attributes,too-many-lines
import functools
import os
import re
import shlex
import subprocess
import time
from typing import Sequence

from absl import app, flags, logging
from tensorflow import io as tf_io

from axlearn.cloud.common.bastion import _LOG_DIR, Bastion, StartBastionJob
from axlearn.cloud.common.bastion import SubmitBastionJob as BaseSubmitBastionJob
from axlearn.cloud.common.bastion import bastion_job_flags, deserialize_jobspec
from axlearn.cloud.common.bundler import DockerBundler, get_bundler_config
from axlearn.cloud.common.quota import QUOTA_CONFIG_PATH, get_resource_limits
from axlearn.cloud.common.scheduler import JobScheduler
from axlearn.cloud.common.uploader import Uploader, with_interval
from axlearn.cloud.common.utils import configure_logging, infer_cli_name, parse_action
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import CPUJob, docker_command
from axlearn.cloud.gcp.tpu_cleaner import TPUCleaner
from axlearn.cloud.gcp.utils import catch_auth, common_flags, get_credentials
from axlearn.cloud.gcp.vm import _compute_resource, create_vm, delete_vm, get_vm_node
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function

_SHARED_BASTION_SUFFIX = "shared-bastion"

FLAGS = flags.FLAGS


def _private_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bastion_job_flags(flag_values=flag_values)
    flag_values.set_default("project", gcp_settings("project", required=False))
    flag_values.set_default("zone", gcp_settings("zone", required=False))

    flags.DEFINE_bool(
        "dry_run", False, "Whether to run with dry-run scheduling.", flag_values=flag_values
    )
    flags.DEFINE_string(
        "vm_type", "n2-highmem-128", "Machine spec to boot for VM.", flag_values=flag_values
    )
    flags.DEFINE_integer("disk_size", 256, "VM disk size in GB.", flag_values=flag_values)
    flags.DEFINE_integer(
        "max_tries", 1, "Max attempts to run the command.", flag_values=flag_values
    )
    flags.DEFINE_integer(
        "retry_interval", 60, "Interval in seconds between tries.", flag_values=flag_values
    )
    flags.DEFINE_multi_string(
        "bundler_spec",
        [],
        "Bundler spec provided as key=value. "
        "Refer to each bundler's `from_spec` method docstring for details.",
        flag_values=flag_values,
    )

    def _validate_name(name: str):
        # Must be a valid GCP VM name, as well as a valid docker tag name. For simplicity, check
        # that it's some letters followed by "-bastion", and that it's not too long (VM names are
        # capped at 63 chars).
        return len(name) < 64 and re.match("[a-z][a-z0-9-]*-bastion", name)

    flags.register_validator(
        "name",
        _validate_name,
        message="Must be < 64 chars and match <name>-bastion.",
        flag_values=flag_values,
    )
    flags.register_validator(
        "max_tries",
        lambda tries: tries > 0,
        message="Max tries must be positive.",
        flag_values=flag_values,
    )


def shared_bastion_name() -> str:
    # The zone-namespacing is necessary because of quirks with compute API. Specifically, even if
    # creating VMs within a specific zone, names are global. On the other hand, the list API only
    # returns VMs within a zone, so there's no easy way to check if a shared bastion already exists
    # in another zone.
    return gcp_settings(  # pytype: disable=bad-return-type
        "bastion_name",
        default=f"{gcp_settings('zone')}-{_SHARED_BASTION_SUFFIX}",
    )


def output_dir(bastion: str) -> str:
    """Directory in gs where jobs are recorded."""
    return os.path.join("gs://", gcp_settings("permanent_bucket"), bastion)


def _gsutil_rsync(
    *,
    src_dir: str,
    dst_dir: str,
    max_tries: int = 5,
    interval_s: float = 30,
    timeout_s: float = 5 * 60,
):
    """An upload fn that uses `gsutil rsync`."""

    for i in range(max_tries):
        # Ensure trailing slash, if not already present, for rsync.
        src = os.path.join(src_dir, "")
        dst = os.path.join(dst_dir, "")
        # Attempt to sync, raising TimeoutError on timeout.
        proc = subprocess.run(
            ["gsutil", "-m", "rsync", "-r", src, dst],
            check=False,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return
        logging.warning("Failed to rsync jobs: stdout=%s stderr=%s", proc.stdout, proc.stderr)
        # No need to sleep on last attempt.
        if i < max_tries - 1:
            time.sleep(interval_s)

    raise ValueError(f"Failed to sync jobs from {src}")


class CreateBastionJob(CPUJob):
    """A job to create and start the remote bastion."""

    @config_class
    class Config(CPUJob.Config):
        """Configures CreateBastionJob."""

        # Type of VM.
        vm_type: Required[str] = REQUIRED
        # Disk size in GB.
        disk_size: Required[int] = REQUIRED
        # Whether to launch bastion in dry-run mode.
        dry_run: bool = False

    @classmethod
    def default_config(cls) -> Config:
        return super().default_config().set(command="")

    def _delete(self):
        cfg = self.config
        delete_vm(cfg.name, credentials=self._get_job_credentials())

    def _execute(self):
        cfg: CreateBastionJob.Config = self.config
        # Create the bastion if it doesn't exist.
        create_vm(
            cfg.name,
            vm_type=cfg.vm_type,
            disk_size=cfg.disk_size,
            bundler_type=self._bundler.TYPE,
            credentials=self._get_job_credentials(),
        )

        # Command to start the bastion inside a docker container.
        # Bastion outputs will be piped to run_log.
        run_log = os.path.join(_LOG_DIR, cfg.name)
        image = self._bundler.id(cfg.name)
        # TODO(markblee): Instead of passing flags manually, consider serializing flags into a
        # flagfile, and reading that.
        run_command = docker_command(
            f"set -o pipefail; mkdir -p {_LOG_DIR}; "
            f"python3 -m axlearn.cloud.gcp.jobs.bastion_vm --name={cfg.name} "
            f"--project={cfg.project} --zone={cfg.zone} "
            f"--dry_run={cfg.dry_run} start 2>&1 | tee -a {run_log}",
            image=image,
            volumes={"/var/tmp": "/var/tmp"},
            detached_session=cfg.name,
        )
        # Command to setup the bastion. Along with the locking mechanism below, should be
        # idempotent. Setup outputs are also piped to run_log.
        start_cmd = f"""set -o pipefail;
            if [[ -z "$(docker ps -f "name={cfg.name}" -f "status=running" -q )" ]]; then
                mkdir -p {_LOG_DIR};
                {self._bundler.install_command(image)} 2>&1 | tee -a {run_log} && {run_command};
            fi"""
        # Run the start command on bastion.
        # Acquire a file lock '/root/start.lock' to guard against concurrent starts.
        # -nx indicates that we acquire an exclusive lock, exiting early if already acquired;
        # -E 0 indicates that early exits still return code 0;
        # -c indicates the command to execute, if we acquire the lock successfully.
        self._execute_remote_cmd(
            f"flock -nx -E 0 --verbose /root/start.lock -c {shlex.quote(start_cmd)}",
            detached_session="start_bastion",
            shell=True,
        )


class SubmitBastionJob(BaseSubmitBastionJob):
    """A job to submit a command to bastion.

    Main differences from base submit:
    - Emits gsutil commands to view logs.
    - Emits a warning if the bastion doesn't exist in GCE.
    """

    def _execute(self):
        cfg: SubmitBastionJob.Config = self.config
        node = get_vm_node(cfg.name, _compute_resource(get_credentials()))
        if node is None or node.get("status", None) != "RUNNING":
            logging.warning(
                "Bastion %s does not appear to be running yet. "
                "It will need to be running before jobs will execute.",
                cfg.name,
            )
        print(
            "\nView bastion outputs with:\n"
            f"gsutil cat {os.path.join(self.bastion_dir, 'logs', cfg.job_name)}\n"
            "\nCheck job history with:\n"
            f"{infer_cli_name()} gcp bastion history --name={cfg.name} --job_name={cfg.job_name}\n"
        )
        return super()._execute()


def _project_quotas_from_file(quota_file: str):
    """Returns a callable that fetches quota information."""
    return functools.partial(get_resource_limits, path=quota_file)


def _job_history(*, bastion_name: str, job_name: str) -> str:
    result = ""
    bastion_path = output_dir(bastion_name)
    spec_path_pattern = os.path.join(bastion_path, "jobs", "*", job_name)
    spec_paths = tf_io.gfile.glob(spec_path_pattern)
    if not spec_paths:
        raise ValueError(f"Job spec not found in {spec_path_pattern}")
    for spec_path in spec_paths:
        with tf_io.gfile.GFile(spec_path, mode="r") as f:
            spec = "".join(f.readlines())
            result += f"<spec path={spec_path}>\n{spec}\n</spec>\n"
    history_path = os.path.join(bastion_path, "history", "jobs", job_name)
    with tf_io.gfile.GFile(history_path, mode="r") as f:
        history = "".join(f.readlines())
        result += f"<history path={history_path}>\n{history}</history>\n"
    return result


def _project_history(*, bastion_name: str, project_id: str) -> str:
    path_pattern = os.path.join(
        output_dir(bastion_name),
        "history",
        "projects",
        project_id,
        "*",
    )
    paths = sorted(tf_io.gfile.glob(path_pattern))
    entries = []
    for path in paths[-2:]:
        with tf_io.gfile.GFile(path, mode="r") as f:
            entry = None
            for line in f:
                if re.search("^[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}", line):
                    # Timestamp line.
                    if entry:
                        entries.append(entry)
                    entry = line
                else:
                    entry += line
            if entry:
                entries.append(entry)
    if len(entries) > 3:
        # Only keep the last three entries.
        entries = entries[-3:]
    lines = "".join(entries)
    return f"<history project_id={project_id}>\n{lines}</history project_id={project_id}>"


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(
        argv, options=["create", "delete", "start", "stop", "submit", "cancel", "history"]
    )

    def quota_file() -> str:
        return os.path.join(
            "gs://",
            gcp_settings("private_bucket"),
            flag_values.name,
            QUOTA_CONFIG_PATH,
        )

    if action == "create":
        # Creates and starts the bastion on a remote VM.
        # Since users share the same bastion, we use docker instead of tar'ing the local dir.
        #
        # Note: The bundler here is only used for inferring the bundle ID. The actual image is built
        # separately, either through automation or with the bundle command (see docstring for
        # details).
        bundler_cfg = get_bundler_config(
            bundler_type=DockerBundler.TYPE, spec=flag_values.bundler_spec
        )
        cfg = CreateBastionJob.from_flags(flag_values).set(
            bundler=bundler_cfg.set(
                image=bundler_cfg.image or "base",
                repo=bundler_cfg.repo or gcp_settings("docker_repo", required=False),
                dockerfile=(
                    bundler_cfg.dockerfile or gcp_settings("default_dockerfile", required=False)
                ),
            ),
        )
        job = cfg.instantiate()
        job.execute()
    elif action == "delete":
        cfg = CreateBastionJob.from_flags(flag_values)
        job = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
    elif action == "start":
        # Start the bastion. This should run on the bastion itself.
        bastion_cfg = Bastion.default_config().set(
            output_dir=output_dir(flag_values.name),
            scheduler=JobScheduler.default_config().set(
                quota=config_for_function(_project_quotas_from_file).set(quota_file=quota_file()),
                dry_run=flag_values.dry_run,
            ),
            cleaner=TPUCleaner.default_config(),
            uploader=Uploader.default_config().set(
                upload_fn=config_for_function(with_interval).set(upload_fn=_gsutil_rsync),
            ),
        )
        cfg = StartBastionJob.from_flags(flag_values).set(max_tries=-1, bastion=bastion_cfg)
        job = cfg.instantiate()
        job.execute()
    elif action == "stop":
        # Stop the bastion. This typically runs locally.
        cfg = CPUJob.from_flags(flag_values).set(
            command=f"docker stop {flag_values.name}", max_tries=1
        )
        job = cfg.instantiate()
        job.execute()
    elif action == "submit":
        spec = deserialize_jobspec(flag_values.spec)
        # Construct a job for bastion to execute. This typically runs locally.
        # The spec file provided from the flags will be used and submitted to bastion vm.
        cfg = SubmitBastionJob.from_flags(flag_values).set(
            job_name=spec.name,
            job_spec_file=flag_values.spec,
            bastion_dir=output_dir(flag_values.name),
        )
        # Execute the job.
        job = cfg.instantiate()
        job.execute()
    elif action == "cancel":
        if not flag_values.job_name:
            raise app.UsageError("--job_name must be provided if running 'cancel'.")
        # Cancel a job that bastion is running (or planning to run).
        cfg = SubmitBastionJob.from_flags(flag_values).set(
            job_spec_file="", bastion_dir=output_dir(flag_values.name)
        )
        job = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
    elif action == "history":
        if flag_values.job_name:
            # Print job history.
            history = _job_history(bastion_name=flag_values.name, job_name=flag_values.job_name)
        elif len(argv) > 2:
            # Print project history.
            project_id = argv[2]
            history = _project_history(bastion_name=flag_values.name, project_id=project_id)
        else:
            limits = get_resource_limits(quota_file())
            raise app.UsageError(
                "The history command should be used with either a --job_name=<job> flag "
                "or a <project_id> arg, where <project_id> can be one of "
                f"{list(limits.project_resources.keys()) + ['none']}"
            )
        print(history)
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    app.run(main)
