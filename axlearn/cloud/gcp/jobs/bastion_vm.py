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
    #  - Unless configured in the settings, the default bastion name is <env_id>-shared-bastion.
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

    # Set runtime options. Will be read by the bastion on next update.
    OPTIONS=$(echo '{"scheduler": {"dry_run": true, "verbosity": 1}}' | jq -R .)
    axlearn gcp bastion set --name=shared-bastion --runtime_options=$OPTIONS

To test changes to bastion:

    # 1. Build a custom image.
    axlearn gcp bundle --bundler_type=artifactregistry \
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
import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Sequence
from typing import Optional

from absl import app, flags, logging

from axlearn.cloud.common.bastion import (
    _LOG_DIR,
    Bastion,
    BastionDirectory,
    bastion_job_flags,
    deserialize_jobspec,
    set_runtime_options,
)
from axlearn.cloud.common.bundler import DockerBundler, get_bundler_config
from axlearn.cloud.common.cleaner import UnschedulableCleaner
from axlearn.cloud.common.job import _with_retry
from axlearn.cloud.common.quota import QUOTA_CONFIG_PATH, get_resource_limits
from axlearn.cloud.common.scheduler import JobScheduler
from axlearn.cloud.common.uploader import Uploader, with_interval
from axlearn.cloud.common.utils import configure_logging, parse_action
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone, gcp_settings
from axlearn.cloud.gcp.event_queue import event_queue_from_config
from axlearn.cloud.gcp.job import CPUJob, docker_command
from axlearn.cloud.gcp.utils import GCPAPI, catch_auth, common_flags
from axlearn.cloud.gcp.vm import create_vm, delete_vm
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function
from axlearn.common.file_system import exists, glob
from axlearn.common.file_system import open as fs_open
from axlearn.common.file_system import readfile

FLAGS = flags.FLAGS


_RSYNC_DIR = os.path.join(_LOG_DIR, "..", "rsync")


# TODO(markblee): Use `define_flags` instead of redefining flags here.
def _private_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bastion_job_flags(flag_values=flag_values)
    flag_values.set_default("project", default_project())
    flag_values.set_default("zone", default_zone())
    flag_values.set_default("env_id", default_env_id())

    flags.DEFINE_string(
        "vm_type", "n2-highmem-128", "Machine spec to boot for VM.", flag_values=flag_values
    )
    flags.DEFINE_integer("disk_size", 1024, "VM disk size in GB.", flag_values=flag_values)
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
    flags.DEFINE_string(
        "runtime_options",
        None,
        "Runtime options provided as a json-serialized string. Will be merged recursively with "
        "existing runtime options.",
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


def shared_bastion_name(
    fv: Optional[flags.FlagValues], gcp_api: Optional[str] = None
) -> Optional[str]:
    # The env_id-namespacing is necessary because of quirks with compute API. Specifically, even if
    # creating VMs within a specific zone, names are global. On the other hand, the list API only
    # returns VMs within a zone, so there's no easy way to check if a shared bastion already exists
    # in another zone.
    # If env_id is not set, fall back to "zone" for backwards compatibility.
    env_id = gcp_settings("env_id", fv=fv, required=False) or gcp_settings("zone", fv=fv)
    if gcp_api is not None and gcp_api.lower() == GCPAPI.GKE.lower():
        default = f"{env_id}-gke-bastion"
    else:
        default = f"{env_id}-shared-bastion"
    bastion_name = gcp_settings(  # pytype: disable=bad-return-type
        "bastion_name",
        default=default,
        fv=fv,
    )
    return bastion_name


def bastion_root_dir(bastion: str, *, fv: Optional[flags.FlagValues]) -> str:
    """Directory in gs where jobs are recorded."""
    return os.path.join("gs://", gcp_settings("permanent_bucket", fv=fv), bastion)


def _gcloud_storage_rsync(
    *,
    src_dir: str,
    dst_dir: str,
    max_tries: int = 5,
    interval_s: float = 30,
    timeout_s: float = 5 * 60,
):
    """An upload fn that uses `gcloud storage rsync`."""

    for i in range(max_tries):
        # Ensure trailing slash from src, if not already present, for rsync.
        src = os.path.join(src_dir, "")
        # Ensure no trailing slash from dst.
        dst = dst_dir.rstrip("/")
        # Attempt to sync, raising TimeoutError on timeout.
        # Using gcloud storage instead of gsutil due to:
        # https://cloud.google.com/blog/products/storage-data-transfer/new-gcloud-storage-cli-for-your-data-transfers
        proc = subprocess.run(
            ["gcloud", "storage", "rsync", "-r", src, dst],
            check=False,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            # Avoid "No space left on device":
            # https://cloud.google.com/knowledge/kb/error-message-while-running-the-command-gsutil-rsync-000004577
            env={"TMPDIR": _RSYNC_DIR},
        )
        if proc.returncode == 0:
            return
        logging.warning("Failed to rsync jobs: stdout=%s stderr=%s", proc.stdout, proc.stderr)
        # No need to sleep on last attempt.
        if i < max_tries - 1:
            time.sleep(interval_s)

    raise ValueError(f"Failed to sync jobs from {src}")


class RemoteBastionJob(CPUJob):
    """A bastion job running on a remote VM.

    TODO(rpang): use CPURunnerJob for remote bastion.
    """

    @config_class
    class Config(CPUJob.Config):
        """Configures RemoteBastionJob."""

        # Type of VM.
        vm_type: Required[str] = REQUIRED
        # Disk size in GB.
        disk_size: Required[int] = REQUIRED
        # The root dir of the bastion.
        root_dir: Required[str] = REQUIRED

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg = super().from_flags(fv, **kwargs)
        cfg.root_dir = bastion_root_dir(cfg.name, fv=fv)
        return cfg

    @classmethod
    def default_config(cls) -> Config:
        return super().default_config().set(command="")

    def _delete(self):
        cfg = self.config
        delete_vm(cfg.name, credentials=self._get_job_credentials())

    def _execute(self):
        cfg: RemoteBastionJob.Config = self.config
        # Create the bastion if it doesn't exist.
        create_vm(
            cfg.name,
            vm_type=cfg.vm_type,
            disk_size=cfg.disk_size,
            bundler_type=self._bundler.TYPE,
            credentials=self._get_job_credentials(),
        )

        # Bastion outputs will be piped to run_log.
        run_log = os.path.join(cfg.root_dir, "logs", f"{cfg.name}-%Y%m%d")
        output_cmd = f"tee >(python3 -m axlearn.cloud.common.writer --output_path={run_log})"

        # Command to start the bastion inside a docker container.
        image = self._bundler.id(cfg.name)
        # TODO(markblee): Instead of passing flags manually, consider serializing flags into a
        # flagfile, and reading that.
        run_cmd = docker_command(
            f"python3 -m axlearn.cloud.gcp.jobs.bastion_vm --name={cfg.name} "
            f"--project={cfg.project} --env_id={cfg.env_id} start 2>&1 | {output_cmd}",
            image=image,
            volumes={"/var/tmp": "/var/tmp"},
            detached_session=cfg.name,
        )
        # Command to setup the bastion. Along with the locking mechanism below, should be
        # idempotent. Setup outputs are piped to setup_log.
        setup_log = os.path.join(_LOG_DIR, "setup.log")
        start_cmd = f"""set -o pipefail;
            mkdir -p {_LOG_DIR}; mkdir -p {_RSYNC_DIR};
            if [[ -z "$(docker ps -f "name={cfg.name}" -f "status=running" -q )" ]]; then
                {self._bundler.install_command(image)} 2>&1 | tee -a {setup_log} && \
                echo "Starting command..." >> {setup_log} && {run_cmd} && \
                echo "Command started." >> {setup_log};
            else
                echo "Already started." >> {setup_log};
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


def _project_quotas_from_file(quota_file: str):
    """Returns a callable that fetches quota information."""
    return functools.partial(get_resource_limits, path=quota_file)


def _job_history(*, job_name: str, root_dir: str) -> str:
    result = ""
    spec_path_pattern = os.path.join(root_dir, "jobs", "*", job_name)
    spec_paths = glob(spec_path_pattern)
    if not spec_paths:
        raise FileNotFoundError(f"Job spec not found in {spec_path_pattern}")
    for spec_path in spec_paths:
        result += f"<spec path={spec_path}>\n{readfile(spec_path)}\n</spec>\n"
    history_path = os.path.join(root_dir, "history", "jobs", job_name)
    result += f"<history path={history_path}>\n{readfile(history_path)}</history>\n"
    return result


def _project_history(*, root_dir: str, project_id: str) -> str:
    project_dir = os.path.join(
        root_dir,
        "history",
        "projects",
        project_id,
    )
    if not exists(project_dir):
        raise FileNotFoundError(f"Project {project_id} not found at {project_dir}")
    paths = sorted(glob(os.path.join(project_dir, "*")))
    entries = []
    for path in paths[-2:]:
        with fs_open(path, mode="r") as f:
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


def _stop_bastion(flag_values: flags.FlagValues):
    # Stop the bastion. This typically runs locally on the caller's machine; CPUJob will SSH into
    # the bastion to issue the `docker stop` command.
    logging.info("Stopping the bastion %s...", flag_values.name)
    cfg = CPUJob.from_flags(flag_values).set(command=f"docker stop {flag_values.name}", max_tries=1)
    job = cfg.instantiate()
    try:
        job.execute()
    except ValueError as e:
        # If we can determine that bastion is already stopped, no need to raise.
        cause = f"No such container: {flag_values.name}"
        curr_e = e
        while curr_e:
            if cause in str(curr_e):
                logging.info("Bastion is already stopped.")
                return
            curr_e = curr_e.__cause__
        raise e  # Else re-raise.


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(
        argv, options=["create", "delete", "start", "stop", "submit", "cancel", "history", "set"]
    )

    def quota_file() -> str:
        return os.path.join(
            "gs://",
            gcp_settings("private_bucket", fv=flag_values),
            flag_values.name,
            QUOTA_CONFIG_PATH,
        )

    def root_dir():
        return bastion_root_dir(flag_values.name, fv=flag_values)

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
        cfg = RemoteBastionJob.from_flags(flag_values).set(
            bundler=bundler_cfg.set(
                image=bundler_cfg.image or "base",
                repo=bundler_cfg.repo
                or gcp_settings("docker_repo", required=False, fv=flag_values),
                dockerfile=(
                    bundler_cfg.dockerfile
                    or gcp_settings("default_dockerfile", required=False, fv=flag_values)
                ),
            ),
        )
        job = cfg.instantiate()
        job.execute()
    elif action == "delete":
        cfg = RemoteBastionJob.from_flags(flag_values)
        job = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
    elif action == "start":
        # Start the bastion. This should run on the bastion itself.
        scheduler_cfg = JobScheduler.default_config().set(
            quota=config_for_function(_project_quotas_from_file).set(quota_file=quota_file()),
        )
        bastion_cfg = Bastion.default_config().set(
            output_dir=root_dir(),
            scheduler=scheduler_cfg,
            cleaner=UnschedulableCleaner.default_config().set(scheduler=scheduler_cfg),
            uploader=Uploader.default_config().set(
                upload_fn=config_for_function(with_interval).set(upload_fn=_gcloud_storage_rsync),
            ),
            quota=config_for_function(_project_quotas_from_file).set(quota_file=quota_file()),
            event_publisher=event_queue_from_config(flag_values=flag_values),
        )

        _with_retry(
            lambda: bastion_cfg.instantiate().execute(),
            interval=60,
            max_tries=-1,
        )
    # TODO(markblee): Split out 'internal' commands from user-facing ones.
    elif action == "stop":
        _stop_bastion(flag_values=flag_values)
    elif action == "submit":
        spec = deserialize_jobspec(flag_values.spec)
        # Construct a job for bastion to execute. This typically runs locally.
        # The spec file provided from the flags will be used and submitted to bastion vm.
        bastion_dir = BastionDirectory.default_config().set(root_dir=root_dir()).instantiate()
        bastion_dir.submit_job(spec.name, job_spec_file=flag_values.spec)
    elif action == "cancel":
        if not flag_values.job_name:
            raise app.UsageError("--job_name must be provided if running 'cancel'.")
        # Cancel a job that bastion is running (or planning to run).
        bastion_dir = BastionDirectory.default_config().set(root_dir=root_dir()).instantiate()
        bastion_dir.cancel_job(flag_values.name)
    elif action == "history":
        if flag_values.job_name:
            # Print job history.
            history = _job_history(root_dir=root_dir(), job_name=flag_values.job_name)
        else:
            # Print project history.
            if len(argv) > 2:
                project_id = argv[2]
            else:
                project_id = "none"
            try:
                history = _project_history(root_dir=root_dir(), project_id=project_id)
            except FileNotFoundError as e:
                limits = get_resource_limits(quota_file())
                raise FileNotFoundError(
                    f"Available projects are {list(limits.project_resources.keys()) + ['none']}"
                ) from e
        print(history)
    elif action == "set":
        try:
            options = json.loads(flag_values.runtime_options)
        except (TypeError, json.JSONDecodeError) as e:
            raise app.UsageError(f"--runtime_options should be a valid json string: {e}")
        set_runtime_options(root_dir(), **options)
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    app.run(main)
