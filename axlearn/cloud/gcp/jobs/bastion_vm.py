# Copyright © 2023 Apple Inc.

"""Launches a bastion VM and executes a command on it.

We use GCS as a registry for jobs that have been submitted to bastion. This ensures that if the
bastion is pre-empted, it can restart and resume monitoring the same jobs. We assume that jobs are
generally resumable via invoking the same command.

More concretely, the submit flow works as follows:
1. The bastion is started by using `axlearn gcp bastion create --name=...`.
    This provisions (creates) and starts the bastion on a remote VM, with the given name.
2. User submits a job using `axlearn gcp bastion submit --spec=...`.
    This simply uploads a job spec to a GCS bucket (serialized via `BastionJobSpec`).
3. Bastion pulls latest docker image (if just started), and polls the bucket in GCS. Each update, it
    syncs all new jobspecs from GCS and runs them asynchronously inside the docker container. Log
    outputs are emitted to GCS.
4. Once a job is completed, its corresponding jobspec is removed from the GCS job registry. Log
    outputs are also synced to GCS.
5. To stop a job, run `axlearn gcp bastion cancel --job_name=...`, which simply writes a
    "cancelling" statefile (serialized via `JobState`) to GCS. The job will be terminated, and any
    configured "cleanup command" will be run (e.g. to cleanup external resources). Writing a
    "cancelling" statefile is necessary to ensure that cleanup commands can be re-run if the bastion
    is pre-empted midway.

Note that emitted logs are "client-side" logs, i.e., logs visible from the bastion. Jobs that use
remote compute like TPUs are expected to sync their own artifacts/logs to a user-accessible location
(see tpu_runner for an example). Jobs are also expected to handle retries on their own.

The bastion itself is pre-emptible; on restart, it will sync in-flight jobs from GCS and run them.

Possible actions: [create|delete|start|stop|submit|cancel]

    Create: creates the bastion VM, and runs "start" on it.
    Delete: deletes the bastion VM.
    Start: starts the bastion script locally. Typically not intended to be used directly.
    Stop: soft-stops the bastion VM, without deleting it.
    Submit: submits a job spec to the bastion VM for scheduling and execution.
    Cancel: cancels a running job managed by the bastion VM.

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

The following paths may provide useful debugging information:

    BUCKET=gs://my-bucket/my-bastion-name

    Active jobspecs: $BUCKET/jobs/active/
    Complete jobspecs: $BUCKET/jobs/complete/

    Active job states: $BUCKET/jobs/states/
    User written job states: $BUCKET/jobs/user_states/

    Bastion logs: $BUCKET/logs/$BASTION
    Job logs: $BUCKET/logs/<job_name>
    Cleanup command logs: $BUCKET/logs/<job_name>.cleanup

    Job scheduling history: $BUCKET/history/jobs/<job_name>
    Project scheduling history: $BUCKET/history/projects/

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
import collections
import dataclasses
import enum
import functools
import json
import multiprocessing
import os
import re
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from subprocess import CalledProcessError
from typing import IO, Any, Dict, Optional, Union

import psutil
import tensorflow as tf
from absl import app, flags, logging

from axlearn.cloud.common.bundler import DockerBundler, get_bundler_config
from axlearn.cloud.common.quota import QUOTA_CONFIG_PATH
from axlearn.cloud.common.scheduler import JobMetadata, JobScheduler, ResourceMap, Scheduler
from axlearn.cloud.common.utils import configure_logging, parse_action
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import CPUJob, GCPJob, docker_command

# TODO(markblee): Switch to using tf.io everywhere, generalize for non-GCP environments, and move to
# axlearn/cloud/common.
from axlearn.cloud.gcp.storage import (
    blob_exists,
    delete_blob,
    download_blob,
    list_blobs,
    upload_blob,
)
from axlearn.cloud.gcp.tpu_cleaner import Cleaner, TPUCleaner
from axlearn.cloud.gcp.utils import common_flags
from axlearn.cloud.gcp.vm import _compute_resource, _get_vm_node, create_vm, delete_vm
from axlearn.common.config import REQUIRED, Required, config_class

_SHARED_BASTION_SUFFIX = "shared-bastion"
_LATEST_BASTION_VERSION = 1  # Determines job schema (see BastionJobSpec).
_LOG_DIR = "/var/tmp/logs"  # Use /var/tmp/ since /tmp/ is cleared every 10 days.
_JOB_DIR = "/var/tmp/jobs"


def _private_flags():
    common_flags()
    FLAGS.set_default("project", gcp_settings("project", required=False))
    FLAGS.set_default("zone", gcp_settings("zone", required=False))

    flags.DEFINE_string("name", None, "Name of bastion.", required=True)
    flags.DEFINE_string("job_name", None, "Name of job.")
    flags.DEFINE_string("vm_type", "n2-standard-128", "Machine spec to boot for VM.")
    flags.DEFINE_integer("disk_size", 256, "VM disk size in GB.")
    flags.DEFINE_integer("max_tries", 1, "Max attempts to run the command.")
    flags.DEFINE_integer("retry_interval", 60, "Interval in seconds between tries.")
    flags.DEFINE_string("spec", None, "Path to a job spec.")
    flags.DEFINE_bool("dry_run", False, "Whether to run with dry-run scheduling.")
    flags.DEFINE_multi_string(
        "bundler_spec",
        [],
        "Bundler spec provided as key=value. "
        "Refer to each bundler's `from_spec` method docstring for details.",
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
    )
    flags.register_validator(
        "max_tries", lambda tries: tries > 0, message="Max tries must be positive."
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


FLAGS = flags.FLAGS


# Subclass str to be JSON serializable: https://stackoverflow.com/a/51976841
class JobState(str, enum.Enum):
    """See BastionJob._update_job for state handling."""

    # Job is queued. Any running command will be forcefully terminated.
    PENDING = "PENDING"
    # Job is about to run, or currently running.
    ACTIVE = "ACTIVE"
    # Job is cancelling. Command is terminating.
    CANCELLING = "CANCELLING"
    # Job has completed/termianted the command, is running cleanup command (if any).
    CLEANING = "CLEANING"
    # Job is complete.
    COMPLETED = "COMPLETED"


@dataclasses.dataclass
class BastionJobSpec:
    """Represents a job that is executed by bastion."""

    # Version to handle schema changes.
    version: int
    # Name of the job (aka job_name).
    name: str
    # Command to run.
    command: str
    # Command to run when job completes (either normally or cancelled).
    cleanup_command: Optional[str]
    # Metadata related to a bastion job.
    metadata: JobMetadata


def new_jobspec(
    *,
    name: str,
    command: str,
    metadata: JobMetadata,
    cleanup_command: Optional[str] = None,
) -> BastionJobSpec:
    return BastionJobSpec(
        version=_LATEST_BASTION_VERSION,
        name=name,
        command=command,
        cleanup_command=cleanup_command,
        metadata=metadata,
    )


def serialize_jobspec(spec: BastionJobSpec, f: Union[str, IO]):
    """Writes job spec to filepath or file."""
    if isinstance(f, str):
        with open(f, "w", encoding="utf-8") as fd:
            serialize_jobspec(spec, fd)
            return

    json.dump(dataclasses.asdict(spec), f, default=str)
    f.flush()


def deserialize_jobspec(f: Union[str, IO]) -> BastionJobSpec:
    """Loads job spec from filepath or file."""
    if isinstance(f, str):
        with open(f, "r", encoding="utf-8") as fd:
            return deserialize_jobspec(fd)

    data = json.load(f)
    if data["version"] == _LATEST_BASTION_VERSION:
        data["metadata"]["creation_time"] = datetime.strptime(
            data["metadata"]["creation_time"], "%Y-%m-%d %H:%M:%S.%f"
        )
        return BastionJobSpec(
            version=data["version"],
            name=data["name"],
            command=data["command"],
            cleanup_command=data.get("cleanup_command", None),
            metadata=JobMetadata(**data["metadata"]),
        )
    raise ValueError(f"Unsupported version: {data['version']}")


def _download_jobspec(
    job_name: str, *, remote_dir: str, local_dir: str = _JOB_DIR
) -> BastionJobSpec:
    """Loads jobspec from gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    local_file = os.path.join(local_dir, job_name)
    download_blob(remote_file, local_file)
    return deserialize_jobspec(local_file)


def _upload_jobspec(spec: BastionJobSpec, *, remote_dir: str, local_dir: str = _JOB_DIR):
    """Uploads jobspec to gs path."""
    local_file = os.path.join(local_dir, spec.name)
    remote_file = os.path.join(remote_dir, spec.name)
    serialize_jobspec(spec, local_file)
    upload_blob(local_file, url=remote_file)


def _bastion_dir(bastion: str) -> str:
    """Directory in gs where jobs are recorded."""
    return os.path.join("gs://", gcp_settings("permanent_bucket"), bastion)


def _sync_dir(
    *, src: str, dst: str, max_tries: int = 5, interval_s: float = 30, timeout_s: float = 5 * 60
):
    """Syncs from src to dst."""
    for i in range(max_tries):
        # Ensure trailing slash, if not already present, for rsync.
        src = os.path.join(src, "")
        dst = os.path.join(dst, "")
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


@dataclasses.dataclass
class _PipedProcess:
    """A process with outputs piped to a file."""

    popen: subprocess.Popen
    fd: IO


def _piped_popen(cmd: str, f: str) -> _PipedProcess:
    """Runs cmd in the background, piping stdout+stderr to a file."""
    # Open with "a" to append to an existing logfile, if any.
    fd = open(f, "a", encoding="utf-8")
    popen = subprocess.Popen(shlex.split(cmd), stdout=fd, stderr=subprocess.STDOUT)
    return _PipedProcess(popen=popen, fd=fd)


def _is_proc_complete(proc: _PipedProcess) -> bool:
    """Returns True iff proc exited with returncode."""
    return proc.popen.poll() is not None


def _catch_with_error_log(fn, *args, **kwargs) -> Any:
    """Wraps a fn with try/except and log the error instead of raising.

    Some non-critical operations like uploading logs can be flaky and shouldn't break the bastion.
    If an exception is caught, the error will be logged and None will be returned. Otherwise, the
    function's outputs will be returned.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        logging.error("[Caught] %s failed with error: %s", fn, e)
    return None


def _kill_popen(popen: subprocess.Popen):
    """Kills the process (and child processes) with SIGKILL."""
    # Note: kill() might leave orphan processes if proc spawned child processes.
    # We use psutil to recursively kill() all children.
    try:
        parent = psutil.Process(popen.pid)
    except psutil.NoSuchProcess:
        return  # Nothing to do.
    for child in parent.children(recursive=True):
        child.kill()
    popen.kill()


@dataclasses.dataclass
class Job:
    spec: BastionJobSpec
    state: JobState
    # *_proc can be None prior to commands being started.
    command_proc: Optional[_PipedProcess]
    cleanup_proc: Optional[_PipedProcess]


def _download_job_state(job_name: str, *, remote_dir: str) -> JobState:
    """Loads job state from gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    if blob_exists(remote_file):
        with tempfile.NamedTemporaryFile("r+") as f:
            download_blob(remote_file, f.name)
            state = f.read().strip().upper()
            return JobState[state]
    # No job state, defaults to PENDING.
    return JobState.PENDING


def _upload_job_state(job_name: str, state: JobState, *, remote_dir: str, verbose: bool = True):
    """Uploads job state to gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(state.name)
        f.flush()
        upload_blob(f.name, url=remote_file, verbose=verbose)


def _start_command(job: Job, *, remote_log_dir: str):
    """Starts the given job.spec.command and sets `job.command_proc`."""
    if job.command_proc is not None:
        return  # Already running.
    # If a log dir exists for this job, download it. This can happen if a job is resumed.
    remote_log = os.path.join(remote_log_dir, job.spec.name)
    local_log = os.path.join(_LOG_DIR, job.spec.name)
    if blob_exists(remote_log):
        download_blob(remote_log, local_log)
    # Pipe all outputs to the local _LOG_DIR.
    job.command_proc = _piped_popen(job.spec.command, local_log)
    logging.info("Started command for the job %s: %s", job.spec.name, job.spec.command)


def _start_cleanup_command(job: Job):
    """Starts the given job.spec.cleanup_command."""
    if not job.spec.cleanup_command:
        logging.info("Job %s has no cleanup command.", job.spec.name)
    elif job.cleanup_proc is None:
        # Pipe all outputs to a local _LOG_DIR.
        job.cleanup_proc = _piped_popen(
            job.spec.cleanup_command, f"{os.path.join(_LOG_DIR, job.spec.name)}.cleanup"
        )
        logging.info(
            "Started cleanup command for the job %s: %s",
            job.spec.name,
            job.spec.cleanup_command,
        )


def download_job_batch(
    *,
    spec_dir: str,
    state_dir: str,
    user_state_dir: str,
    local_spec_dir: str = _JOB_DIR,
    verbose: bool = False,
) -> Dict[str, Job]:
    """Downloads a batch of jobs.

    Args:
        spec_dir: Directory to look for job specs.
        state_dir: Directory to look for job states.
        user_state_dir: Directory to look for user states.
        local_spec_dir: Directory to store downloaded job specs.
        verbose: Verbose logging.

    Returns:
        A mapping from job name to Job(spec, state).
    """
    # Figure out which statefiles to download from GCS. If user has written to "user_states",
    # those take precedence over bastion's statefiles. For example, user may write a
    # "cancelling" statefile.
    # TODO(markblee): Prevent two situations:
    # 1. Setting a user state to anything other than CANCELLING.
    # 2. Overriding a job state to CANCELLING when it's already in CLEANING/COMPLETED.
    user_states = {os.path.basename(f) for f in list_blobs(user_state_dir)}
    if verbose:
        logging.info("User states %s", user_states)
    job_names = []
    state_dirs = []
    for jobspec in list_blobs(spec_dir):
        job_name = os.path.basename(jobspec)
        # User states override bastion states.
        job_state_dir = user_state_dir if job_name in user_states else state_dir
        job_names.append(job_name)
        state_dirs.append(job_state_dir)
        if verbose:
            logging.info("Downloading %s from %s", job_name, job_state_dir)
    jobs = {}
    # max_workers matches urllib3's default connection pool size.
    with ThreadPoolExecutor(max_workers=10) as pool:
        download_spec_fn = functools.partial(
            _download_jobspec,
            remote_dir=spec_dir,
            local_dir=local_spec_dir,
        )
        spec_futs = [pool.submit(download_spec_fn, job_name) for job_name in job_names]
        state_futs = [
            pool.submit(_download_job_state, job_name, remote_dir=job_state_dir)
            for job_name, job_state_dir in zip(job_names, state_dirs)
        ]
        wait(spec_futs)
        wait(state_futs)
        for i, (job_name, spec_fut, state_fut) in enumerate(zip(job_names, spec_futs, state_futs)):
            try:
                # Mapping from job_name to (spec, state).
                jobs[job_name] = Job(
                    spec=spec_fut.result(),
                    state=state_fut.result(),
                    command_proc=None,
                    cleanup_proc=None,
                )
            except Exception as e:  # pylint: disable=broad-except
                # TODO(markblee): Distinguish transient vs non-transient errors.
                logging.warning(
                    "Failed to load job %s with spec from %s and state from %s: %s",
                    job_name,
                    spec_dir,
                    state_dirs[i],
                    e,
                )
    return jobs


class BastionJob(GCPJob):
    """A job that runs on a remote VM, which executes arbitrary user commands."""

    @config_class
    class Config(GCPJob.Config):
        """Configures BastionJob."""

        # Interval to sync and run jobs.
        update_interval_seconds: float = 30
        # Scheduler to decide whether to start/pre-empt jobs.
        scheduler: Required[Scheduler.Config] = REQUIRED
        # Cleaner to deprovision idle resources.
        cleaner: Required[Cleaner.Config] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Remote gs directory to emit logs.
        self._output_dir = _bastion_dir(cfg.name)
        # Remote log output dir. Ensure trailing slash.
        self._log_dir = os.path.join(self._output_dir, "logs")
        # Note: pathlib doesn't work well with gs:// prefix.
        self._job_dir = os.path.join(self._output_dir, "jobs")
        # Remote history dir. Ensure trailing slash.
        self._job_history_dir = os.path.join(self._output_dir, "history", "jobs")
        tf.io.gfile.makedirs(self._job_history_dir)
        self._project_history_dir = os.path.join(self._output_dir, "history", "projects")
        tf.io.gfile.makedirs(self._project_history_dir)
        # Mapping from project_id to previous job verdicts.
        self._project_history_previous_verdicts = {}
        # Jobs that have fully completed.
        self._complete_dir = os.path.join(self._job_dir, "complete")
        # Active jobs (all other jobs).
        self._active_dir = os.path.join(self._job_dir, "active")
        # All user states (e.g. "cancelling" written by user).
        self._user_state_dir = os.path.join(self._job_dir, "user_states")
        # All bastion-managed job states.
        self._state_dir = os.path.join(self._job_dir, "states")
        # Local active jobs (and respective commands, files, etc).
        # TODO(markblee): Rename this, as it includes more than just ACTIVE jobs (e.g. PENDING).
        self._active_jobs: Dict[str, Job] = {}
        # Log sync process.
        self._sync_log_proc = None

        # Instantiate children.
        self._scheduler = cfg.scheduler.instantiate()
        self._cleaner = cfg.cleaner.instantiate()

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        cfg.command = ""  # Unused.
        return cfg

    def _append_to_job_history(self, job: Job, msg: str):
        with tf.io.gfile.GFile(os.path.join(self._job_history_dir, f"{job.spec.name}"), "a") as f:
            curr_time = datetime.now(timezone.utc).strftime("%m%d %H:%M:%S")
            f.write(f"{curr_time} {msg}\n")

    def _append_to_project_history(
        self, jobs: Dict[str, JobMetadata], schedule_results: Scheduler.ScheduleResults
    ):
        now = datetime.now(timezone.utc)
        for project_id, limits in schedule_results.project_limits.items():
            job_verdicts = schedule_results.job_verdicts.get(project_id, {})
            verdicts = []
            for job_id, verdict in job_verdicts.items():
                verdicts.append((job_id, verdict.should_run()))
            verdicts = sorted(verdicts)
            previous_verdicts = self._project_history_previous_verdicts.get(project_id)
            if previous_verdicts == verdicts:
                # Nothing changed.
                continue
            self._project_history_previous_verdicts[project_id] = verdicts
            # Mapping from resource types to usage.
            project_usage = collections.defaultdict(lambda: 0)
            running_jobs = []
            queued_jobs = []
            for job_id, verdict in job_verdicts.items():
                if verdict.should_run():
                    running_jobs.append(job_id)
                    job_metadata = jobs[job_id]
                    for resource_type, demand in job_metadata.resources.items():
                        project_usage[resource_type] += demand
                else:
                    queued_jobs.append(job_id)

            def resource_str(resource_map: ResourceMap) -> str:
                return ", ".join(
                    sorted(
                        f"{resource_type}={quantity}"
                        for resource_type, quantity in resource_map.items()
                    )
                )

            project_dir = os.path.join(self._project_history_dir, project_id)
            tf.io.gfile.makedirs(project_dir)
            with tf.io.gfile.GFile(os.path.join(project_dir, now.strftime("%Y%m%d")), "a") as f:
                curr_time = now.strftime("%m%d %H:%M:%S")
                f.write(f"{curr_time}\n")
                f.write(f"Effective limits: {resource_str(limits)}\n")
                f.write(f"Usage: {resource_str(project_usage)}\n")
                f.write("Running jobs:\n")
                for job_id in running_jobs:
                    f.write(f"  {job_id}\n")
                f.write("Queued jobs:\n")
                for job_id in queued_jobs:
                    f.write(f"  {job_id}\n")

    def _sync_logs(self, interval_s: float = 10):
        """Periodically sync output logs to gs in a separate process.

        If the log sync process has died, it will be restarted.
        """

        def fn():
            sync_s = 0
            while True:
                src, dst = _LOG_DIR, self._log_dir
                logging.log_every_n(
                    logging.INFO, "Syncing outputs %s -> %s. Last duration: %s", 6, src, dst, sync_s
                )
                start = time.time()
                try:
                    _sync_dir(src=src, dst=dst, max_tries=1)
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning("Sync failed: %s", e)
                sync_s = time.time() - start
                if sync_s > interval_s:
                    logging.warning(
                        "Syncing outputs exceeded interval: %s > %s", sync_s, interval_s
                    )
                time.sleep(max(0, interval_s - sync_s))

        if self._sync_log_proc is not None:
            self._sync_log_proc.join(timeout=0)
            if not self._sync_log_proc.is_alive():
                logging.info("Log sync process died, removing...")
                self._sync_log_proc.kill()
                self._sync_log_proc.join()
                self._sync_log_proc = None
                logging.info("Log process removed. Will restart...")

        if self._sync_log_proc is None:
            logging.info("Starting log sync process.")
            self._sync_log_proc = multiprocessing.Process(target=fn, daemon=True)
            self._sync_log_proc.start()
            logging.info("Log sync started.")
        else:
            logging.info("Log sync is still running.")

    def _wait_and_close_proc(self, proc: _PipedProcess, kill: bool = False):
        """Cleans up the process/fds and upload logs to gs."""
        if kill:
            _kill_popen(proc.popen)
        # Note: proc should already be polled and completed, so wait is nonblocking.
        proc.popen.wait()
        proc.fd.close()
        # Upload outputs to log dir.
        _catch_with_error_log(
            upload_blob,
            proc.fd.name,
            url=os.path.join(self._log_dir, os.path.basename(proc.fd.name)),
        )
        # Remove the local output file.
        if os.path.exists(proc.fd.name):
            os.remove(proc.fd.name)

    def _sync_jobs(self):
        """Makes the local bastion state consistent with the remote GCS state.

        This function serves as a synchronization point for user-initiated state changes
        ("user_states") and state changes from a prior `_update_job` ("states"). Users should avoid
        writing to the "states" dir directly, as doing so can produce races with `_update_job`.

        More specifically, this function:
        1. Downloads all active jobspecs from GCS.
        2. Downloads all statefiles for active jobspecs from GCS:
            - If statefile exists in "user_states", use it;
            - Otherwise, if statefile exists in "states", use it;
            - Otherwise, default to ACTIVE.

        We use these jobspecs to update the local self._active_jobs.
        """
        active_jobs = download_job_batch(
            spec_dir=self._active_dir,
            state_dir=self._state_dir,
            user_state_dir=self._user_state_dir,
            verbose=True,
        )

        # Iterate over unique job names.
        # pylint: disable-next=use-sequence-for-iteration
        for job_name in {*active_jobs.keys(), *self._active_jobs.keys()}:
            # Detected new job: exists in GCS, but not local.
            if job_name not in self._active_jobs:
                logging.info("Detected new job %s.", job_name)
                self._active_jobs[job_name] = active_jobs[job_name]
            # Detected removed job: exists locally, but not in GCS.
            elif job_name not in active_jobs:
                job = self._active_jobs[job_name]
                if job.state != JobState.COMPLETED:
                    logging.warning("Detected orphaned job %s! Killing it...", job.spec.name)
                    if job.command_proc is not None:
                        self._wait_and_close_proc(job.command_proc, kill=True)
                    if job.cleanup_proc is not None:
                        self._wait_and_close_proc(job.cleanup_proc, kill=True)
                logging.info("Removed job %s.", job_name)
                del self._active_jobs[job_name]
            # Detected updated job: exists in both.
            else:
                curr_job = self._active_jobs[job_name]
                updated_job = active_jobs[job_name]
                curr_job.spec, curr_job.state = updated_job.spec, updated_job.state

    # pylint: disable-next=too-many-statements
    def _update_single_job(self, job: Job) -> Job:
        """Handles all state transitions for a single job.

        Assumptions:
        1. A jobspec file exists in GCS at the start of each call. The job.state provided to this
            function call is consistent with that state in GCS + any scheduling decisions.
        2. The function may be called by a freshly started bastion (recovering from pre-emption).
            Thus each condition must assume nothing about the local state.
        3. The function may be pre-empted at any point.
        4. Job commands/cleanup commands are resumable (can invoke the same command multiple times).

        Conditions that must be held at exit (either pre-emption or graceful):
        1. A jobspec must still exist in GCS.
        2. self._active_jobs must be unmodified, besides modifying `job` itself.
        """
        if job.state == JobState.PENDING:
            # Forcefully terminate the command proc and fd, if they exist, and sync logs to remote.
            # The forceful termination is similar to the behavior when bastion itself is pre-empted.
            #
            # We must also ensure that:
            # 1. command_proc is set to None, so we can resume in ACTIVE in a subsequent step
            #    (possibly the next step).
            # 2. Any job logs are sync'ed to remote log dir. The local log file cannot reliably be
            #    expected to be present if/when the job is resumed.
            if job.command_proc is not None:
                self._append_to_job_history(job, "PENDING: pre-empting")
                logging.info("Pre-empting job: %s", job.spec.name)
                self._wait_and_close_proc(job.command_proc, kill=True)
                job.command_proc = None
                logging.info("Job is pre-empted: %s", job.spec.name)

            job.state = JobState.PENDING

        elif job.state == JobState.ACTIVE:
            # Run the command if not already started. We attempt to run every time, in case bastion
            # got pre-empted.
            if job.command_proc is None:
                self._append_to_job_history(
                    job, f"ACTIVE: start process command: {job.spec.command}"
                )
            _start_command(job, remote_log_dir=self._log_dir)
            assert job.command_proc is not None

            # If command is completed, move to CLEANING. Otherwise, it's still RUNNING.
            if _is_proc_complete(job.command_proc):
                self._append_to_job_history(job, "CLEANING: process finished")
                logging.info(
                    "Job %s stopped gracefully: %s.",
                    job.spec.name,
                    job.command_proc.popen.returncode,
                )
                job.state = JobState.CLEANING

        elif job.state == JobState.CANCELLING:
            # If job is still running, terminate it. We stay in CANCELLING until it has fully
            # exited, after which we move to CLEANING.
            if job.command_proc is not None and not _is_proc_complete(job.command_proc):
                self._append_to_job_history(job, "CANCELLING: terminating the process")
                logging.info("Sending SIGTERM to job: %s", job.spec.name)
                job.command_proc.popen.terminate()
            else:
                self._append_to_job_history(job, "CLEANING: process terminated")
                job.state = JobState.CLEANING

        elif job.state == JobState.CLEANING:
            # If command exists, it must be fully stopped.
            assert job.command_proc is None or _is_proc_complete(job.command_proc)

            # Close the command proc and fd, if they exist.
            if job.command_proc is not None:
                self._wait_and_close_proc(job.command_proc)
                job.command_proc = None

            # Run the cleanup command if not already started (and if it exists). We attempt to run
            # every time, in case bastion got pre-empted.
            if job.spec.cleanup_command and not job.cleanup_proc:
                self._append_to_job_history(
                    job, f"CLEANING: start cleanup command: {job.spec.cleanup_command}"
                )
            _start_cleanup_command(job)

            # If job has no cleanup command, or cleanup command is complete, transition to
            # COMPLETED.
            if job.cleanup_proc is None or _is_proc_complete(job.cleanup_proc):
                self._append_to_job_history(job, "COMPLETED: cleanup finished")
                logging.info("Job %s finished running cleanup.", job.spec.name)
                if job.cleanup_proc is not None:
                    self._wait_and_close_proc(job.cleanup_proc)
                    job.cleanup_proc = None

                job.state = JobState.COMPLETED

        elif job.state == JobState.COMPLETED:
            # Copy the jobspec to "complete" dir.
            local_jobspec = os.path.join(_JOB_DIR, job.spec.name)
            if os.path.exists(local_jobspec):
                remote_jobspec = os.path.join(self._complete_dir, job.spec.name)
                if not blob_exists(remote_jobspec):
                    _upload_jobspec(job.spec, local_dir=_JOB_DIR, remote_dir=self._complete_dir)
                os.remove(local_jobspec)

        else:
            raise ValueError(f"Unexpected state: {job.state}")

        # Flush the state to GCS.
        # TODO(markblee): Skip the upload if we can detect the state hasn't changed.
        # State changes can come from user_states, scheduling, or above, so probably not worth the
        # complexity at the moment, given how small job states are.
        _upload_job_state(job.spec.name, job.state, remote_dir=self._state_dir, verbose=False)

        # Remove any remote "user_states" now that "states" dir has synchronized.
        user_state = os.path.join(self._user_state_dir, job.spec.name)
        if blob_exists(user_state):
            delete_blob(user_state)

        return job

    def _update_jobs(self):
        """Handles state transitions for all jobs.

        The scheduler is used to determine which jobs to resume/pre-empt based on job priority.
        The actual updates can be performed in any order (possibly in parallel).

        Note that this function should respect the conditions specified in `_update_single_job`.
        """
        logging.info("")
        logging.info("==Begin update step.")

        # Identify jobs which are schedulable.
        schedulable_jobs = {}
        for job_name, job in self._active_jobs.items():
            if job.state in {JobState.PENDING, JobState.ACTIVE}:
                schedulable_jobs[job_name] = job.spec.metadata

        # Decide which jobs to resume/pre-empt.
        schedule_results: Scheduler.ScheduleResults = self._scheduler.schedule(schedulable_jobs)
        self._append_to_project_history(schedulable_jobs, schedule_results)
        for verdicts in schedule_results.job_verdicts.values():
            for job_name, verdict in verdicts.items():
                if verdict.should_run():  # Resume/keep running.
                    self._active_jobs[job_name].state = JobState.ACTIVE
                else:  # Pre-empt/stay queued.
                    self._active_jobs[job_name].state = JobState.PENDING

        # TODO(markblee): Parallelize this.
        for job_name, job in self._active_jobs.items():
            try:
                self._update_single_job(job)
            except (CalledProcessError, RuntimeError) as e:
                logging.warning("Failed to execute %s: %s", job_name, e)

        logging.info("")
        logging.info("All job states:")
        for job_name, job in self._active_jobs.items():
            logging.info("%s: %s", job_name, job.state)

        logging.info("==End of update step.")
        logging.info("")

    def _gc_jobs(self):
        """Garbage collects idle jobs and completed specs.

        Note that this does not modify job state or self._active_jobs. Instead, fully gc'ed
        COMPLETED jobs will have their jobspecs removed. In the next _sync_jobs, self._active_jobs
        will be made consistent with GCS state.
        """
        logging.info("")
        logging.info("==Begin gc step.")
        cleaned = []

        # Identify jobs which are idle (i.e., can be garbage collected).
        jobs_to_clean = {}
        for job_name, job in self._active_jobs.items():
            if job.state in {JobState.PENDING, JobState.COMPLETED}:
                jobs_to_clean[job_name] = job.spec.metadata.resources

        # Note that this may contain PENDING jobs that have not yet started (since they will not be
        # associated with any resources yet).
        cleaned.extend(self._cleaner.sweep(jobs_to_clean))

        def _delete_jobspec(job_name: str):
            logging.info("Deleting jobspec for %s", job_name)
            # Delete jobspec before state. This ensures that we won't pickup the job again in
            # _sync_jobs, and that state files are always associated with a jobspec.
            delete_blob(os.path.join(self._active_dir, job_name))
            delete_blob(os.path.join(self._state_dir, job_name))
            logging.info("Job %s is complete.", job_name)

        # Remove remote jobspecs for COMPLETED jobs that finished gc'ing.
        # TODO(markblee): GC orphaned states (e.g. if delete gets pre-empted).
        cleaned_completed = [
            job_name
            for job_name in cleaned
            if self._active_jobs[job_name].state == JobState.COMPLETED
        ]
        logging.info("Fully cleaned COMPLETED jobs: %s", cleaned_completed)
        with ThreadPoolExecutor() as pool:
            pool.map(_delete_jobspec, cleaned_completed)

        logging.info("==End of gc step.")
        logging.info("")

    def _execute(self):
        """Provisions and launches a command on a VM."""
        cfg: BastionJob.Config = self.config
        os.makedirs(_LOG_DIR, exist_ok=True)
        os.makedirs(_JOB_DIR, exist_ok=True)
        while True:
            start = time.time()
            self._sync_logs()
            self._sync_jobs()
            self._update_jobs()
            self._gc_jobs()
            execute_s = time.time() - start
            if execute_s > cfg.update_interval_seconds:
                logging.warning(
                    "Execute step exceeded interval: %s > %s",
                    execute_s,
                    cfg.update_interval_seconds,
                )
            time.sleep(max(0, cfg.update_interval_seconds - execute_s))


# TODO(markblee): Add more unit tests.
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
        cfg = super().default_config()
        cfg.command = ""
        return cfg

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


# TODO(markblee): Add more unit tests.
class SubmitBastionJob(CPUJob):
    """A job to submit a command to bastion."""

    @config_class
    class Config(CPUJob.Config):
        """Configures SubmitBastionJob."""

        # Name of the job. Not to be confused with cfg.name, the bastion name.
        job_name: Required[str] = REQUIRED
        # Job spec file local path.
        job_spec_file: Required[str] = REQUIRED

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.command = ""
        return cfg

    def _output_dir(self):
        return _bastion_dir(self.config.name)

    def _job_dir(self):
        return os.path.join(self._output_dir(), "jobs")

    def _delete(self):
        cfg: SubmitBastionJob.Config = self.config
        try:
            jobspec = os.path.join(self._job_dir(), "active", cfg.job_name)
            if not blob_exists(jobspec):
                raise ValueError(f"Unable to locate jobspec {jobspec}")
            _upload_job_state(
                cfg.job_name,
                JobState.CANCELLING,
                remote_dir=os.path.join(self._job_dir(), "user_states"),
            )
            logging.info(
                "Job %s is cancelling.\nView bastion outputs with:\ngsutil cat %s",
                cfg.job_name,
                os.path.join(self._output_dir(), "logs", f"{cfg.job_name}.cleanup"),
            )
        except ValueError as e:
            logging.info("Failed with error: %s -- Has the job been cancelled already?", e)

    def _execute(self):
        cfg: SubmitBastionJob.Config = self.config
        node = _get_vm_node(cfg.name, _compute_resource(self._get_job_credentials()))
        if node is None or node.get("status", None) != "RUNNING":
            logging.warning(
                "Bastion %s does not appear to be running yet. "
                "It will need to be running before jobs will execute.",
                cfg.name,
            )
        logging.info("Submitting command to bastion: %s", cfg.command)
        dst = os.path.join(self._job_dir(), "active", cfg.job_name)
        if blob_exists(dst):
            logging.info("\n\nNote: Job is already running. To restart it, cancel the job first.\n")
        else:
            # Upload the job for bastion to pickup.
            upload_blob(cfg.job_spec_file, url=dst)

        print(
            "\nView bastion outputs with:\n"
            f"gsutil cat {os.path.join(self._output_dir(), 'logs', cfg.job_name)}",
        )


def main(argv):
    action = parse_action(argv, options=["create", "delete", "start", "stop", "submit", "cancel"])

    if action == "create":
        # Creates and starts the bastion on a remote VM.
        # Since users share the same bastion, we use docker instead of tar'ing the local dir.
        #
        # Note: The bundler here is only used for inferring the bundle ID. The actual image is built
        # separately, either through automation or with the bundle command (see docstring for
        # details).
        bundler_cfg = get_bundler_config(bundler_type=DockerBundler.TYPE, spec=FLAGS.bundler_spec)
        cfg = CreateBastionJob.from_flags(FLAGS)
        cfg.set(
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
        cfg = CreateBastionJob.from_flags(FLAGS)
        job = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
    elif action == "start":
        # Start the bastion. This should run on the bastion itself.
        quota_file = f"gs://{gcp_settings('private_bucket')}/{FLAGS.name}/{QUOTA_CONFIG_PATH}"
        cfg = BastionJob.from_flags(FLAGS).set(
            max_tries=-1,
            scheduler=JobScheduler.default_config().set(
                project_quota_file=quota_file,
                dry_run=FLAGS.dry_run,
            ),
            cleaner=TPUCleaner.default_config(),
        )
        job = cfg.instantiate()
        job.execute()
    elif action == "stop":
        # Stop the bastion. This typically runs locally.
        cfg = CPUJob.from_flags(FLAGS).set(
            command=f"docker stop {FLAGS.name}; rm -r {_JOB_DIR}",
            max_tries=1,
        )
        job = cfg.instantiate()
        job.execute()
    elif action == "submit":
        spec = deserialize_jobspec(FLAGS.spec)
        # Construct a job for bastion to execute. This typically runs locally.
        # The spec file provided from the flags will be used and submitted to bastion vm.
        cfg = SubmitBastionJob.from_flags(FLAGS).set(job_name=spec.name, job_spec_file=FLAGS.spec)
        # Execute the job.
        job = cfg.instantiate()
        job.execute()
    elif action == "cancel":
        if not FLAGS.job_name:
            raise app.UsageError("--job_name must be provided if running 'cancel'.")
        # Cancel a job that bastion is running (or planning to run).
        cfg = SubmitBastionJob.from_flags(FLAGS).set(job_spec_file="")
        job = cfg.instantiate()
        job._delete()  # pylint: disable=protected-access
        # Poll for jobspec to be removed.
        try:
            while True:
                # pylint: disable-next=protected-access
                dst = os.path.join(job._job_dir(), "active", cfg.job_name)
                if blob_exists(dst):
                    logging.info("Waiting for job to stop (use ctrl+c to stop waiting)...")
                    time.sleep(10)
                else:
                    break
            logging.info("Job is stopped.")
        except KeyboardInterrupt:
            logging.info("Job is stopping in the background.")
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    app.run(main)
