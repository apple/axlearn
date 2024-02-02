# Copyright © 2023 Apple Inc.

"""A simple cloud-agnostic job orchestrator.

The bastion is designed to have minimal dependencies:
1. It uses a cloud storage directory (such as GCS or S3) to store job states and logs.
2. It runs on a single VM.

The cloud storage directory has the following structure:

    ROOT=<cloud_storage_path>/<bastion_name>

    Active jobspecs: $ROOT/jobs/active/
    Complete jobspecs: $ROOT/jobs/complete/
    Active job states: $ROOT/jobs/states/
    User written job states: $ROOT/jobs/user_states/

    Bastion logs: $ROOT/logs/<bastion_name>
    Job logs: $ROOT/logs/<job_name>
    Cleanup command logs: $ROOT/logs/<job_name>.cleanup

    Job scheduling history: $ROOT/history/jobs/<job_name>
    Project scheduling history: $ROOT/history/projects/<project_name>/<date>

At a high level, the submit flow works as follows:
1. User submits a job to the bastion by uploading a job spec to the 'active jobspecs' path above
    (serialized via `JobSpec`).
2. Bastion polls the cloud storage directory. Each update, it syncs all new jobspecs from the
    directory and runs them asynchronously inside a docker container. Log outputs are emitted back
    to the directory.
3. Once a job is completed, its corresponding jobspec is removed from the cloud storage directory.
4. The bastion also supports user interaction. For instance, if a user wants to cancel a job, a
    "cancelling" state file (serialized via `JobState`) can be written to the cloud storage
    directory. The bastion will read these "user states" and terminate the jobs appropriately, as
    well as cleanup any processed "user state" files.

Bastion jobs should:
1. Be executable via invoking a bash command.
2. Be resumable via invoking the same command. This allows the bastion to be pre-emptible; when
    bastion restarts, it can simply resume the in-flight jobs by re-running each job command.
3. Be responsible for cleaning up any external resources that it creates (e.g. via the cleanup
    command in the jobspec).
4. Sync their own artifacts/logs to external storage (like a cloud storage directory), if persisting
    outputs is desired.
5. Handle retries internally, if retries are desired.
"""
# pylint: disable=consider-using-with,too-many-branches,too-many-instance-attributes,too-many-lines
import collections
import dataclasses
import enum
import functools
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from subprocess import CalledProcessError
from typing import IO, Any, Dict, List, Optional, Set, Tuple, Union

from absl import flags, logging
from tensorflow import errors as tf_errors
from tensorflow import io as tf_io
from tensorflow import nest as tf_nest

# tensorflow_io import is necessary for tf_io to understand s3:// scheme.
try:
    # pylint: disable-next=import-error,unused-import
    import tensorflow_io  # pytype: disable=import-error
except ModuleNotFoundError:
    logging.warning("tensorflow_io is not installed -- tf_io may not work with s3://")

from axlearn.cloud.common.cleaner import Cleaner
from axlearn.cloud.common.job import Job as CloudJob
from axlearn.cloud.common.scheduler import JobMetadata, ResourceMap, Scheduler
from axlearn.cloud.common.uploader import Uploader
from axlearn.cloud.common.utils import merge, send_signal
from axlearn.common.config import REQUIRED, Configurable, Required, config_class
from axlearn.common.utils import Nested

_LATEST_BASTION_VERSION = 1  # Determines job schema (see JobSpec).
_LOG_DIR = "/var/tmp/logs"  # Use /var/tmp/ since /tmp/ is cleared every 10 days.
_JOB_DIR = "/var/tmp/jobs"

FLAGS = flags.FLAGS


def bastion_job_flags(flag_values: flags.FlagValues = FLAGS):
    flags.DEFINE_string("name", None, "Name of bastion.", flag_values=flag_values, required=True)
    flags.DEFINE_string("job_name", None, "Name of job.", flag_values=flag_values)
    flags.DEFINE_string("spec", None, "Path to a job spec.", flag_values=flag_values)


# Subclass str to be JSON serializable: https://stackoverflow.com/a/51976841
class JobState(str, enum.Enum):
    """See Bastion._update_job for state handling."""

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
class JobSpec:
    """Represents a job that is executed by bastion."""

    # Version to handle schema changes.
    version: int
    # Name of the job (aka job_name).
    name: str
    # Command to run.
    command: str
    # Command to run when job completes (either normally or cancelled).
    cleanup_command: Optional[str]
    # Environment Variables. Will be merged into os.envrion and applied for both
    # command and cleanup_command.
    env_vars: Optional[Dict[str, str]]
    # Metadata related to a bastion job.
    metadata: JobMetadata


def new_jobspec(
    *,
    name: str,
    command: str,
    metadata: JobMetadata,
    cleanup_command: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> JobSpec:
    return JobSpec(
        version=_LATEST_BASTION_VERSION,
        name=name,
        command=command,
        cleanup_command=cleanup_command,
        env_vars=env_vars,
        metadata=metadata,
    )


def serialize_jobspec(spec: JobSpec, f: Union[str, IO]):
    """Writes job spec to filepath or file."""
    if isinstance(f, str):
        with open(f, "w", encoding="utf-8") as fd:
            serialize_jobspec(spec, fd)
            return

    json.dump(dataclasses.asdict(spec), f, default=str)
    f.flush()


def deserialize_jobspec(f: Union[str, IO]) -> JobSpec:
    """Loads job spec from filepath or file."""
    if isinstance(f, str):
        with open(f, "r", encoding="utf-8") as fd:
            return deserialize_jobspec(fd)

    data = json.load(f)
    if data["version"] == _LATEST_BASTION_VERSION:
        data["metadata"]["creation_time"] = datetime.strptime(
            data["metadata"]["creation_time"], "%Y-%m-%d %H:%M:%S.%f"
        )
        return JobSpec(
            version=data["version"],
            name=data["name"],
            command=data["command"],
            cleanup_command=data.get("cleanup_command", None),
            env_vars=data.get("env_vars", None),
            metadata=JobMetadata(**data["metadata"]),
        )
    raise ValueError(f"Unsupported version: {data['version']}")


def _download_jobspec(job_name: str, *, remote_dir: str, local_dir: str = _JOB_DIR) -> JobSpec:
    """Loads jobspec from gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    local_file = os.path.join(local_dir, job_name)
    tf_io.gfile.copy(remote_file, local_file, overwrite=True)
    return deserialize_jobspec(local_file)


def _upload_jobspec(spec: JobSpec, *, remote_dir: str, local_dir: str = _JOB_DIR):
    """Uploads jobspec to gs path."""
    local_file = os.path.join(local_dir, spec.name)
    remote_file = os.path.join(remote_dir, spec.name)
    serialize_jobspec(spec, local_file)
    tf_io.gfile.copy(local_file, remote_file, overwrite=True)


@dataclasses.dataclass
class _PipedProcess:
    """A process with outputs piped to a file."""

    popen: subprocess.Popen
    fd: IO


def _piped_popen(cmd: str, f: str, *, env_vars: Optional[Dict[str, str]] = None) -> _PipedProcess:
    """Runs cmd in the background, piping stdout+stderr to a file."""
    # Open with "a" to append to an existing logfile, if any.
    fd = open(f, "a", encoding="utf-8")
    # Make a copy of system env.
    env_vars_copy = os.environ.copy()
    if env_vars:
        # Inject supplied env.
        env_vars_copy.update(env_vars)

    popen = subprocess.Popen(
        shlex.split(cmd), stdout=fd, stderr=subprocess.STDOUT, env=env_vars_copy
    )
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


@dataclasses.dataclass
class Job:
    spec: JobSpec
    state: JobState
    # *_proc can be None prior to commands being started.
    command_proc: Optional[_PipedProcess]
    cleanup_proc: Optional[_PipedProcess]


def _download_job_state(job_name: str, *, remote_dir: str) -> JobState:
    """Loads job state from gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    try:
        # Note: tf_io.gfile.GFile seems to hit libcurl errors with ThreadPoolExecutor.
        with tempfile.NamedTemporaryFile("r+") as f:
            tf_io.gfile.copy(remote_file, f.name, overwrite=True)
            state = f.read().strip().upper()
            return JobState[state]
    except tf_errors.NotFoundError:
        # No job state, defaults to PENDING.
        return JobState.PENDING


def _upload_job_state(job_name: str, state: JobState, *, remote_dir: str, verbose: bool = True):
    """Uploads job state to gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    logging.log_if(logging.INFO, "Writing %s to %s.", verbose, state.name, remote_file)
    with tf_io.gfile.GFile(remote_file, mode="w") as f:
        f.write(state.name)


def _start_command(job: Job, *, remote_log_dir: str):
    """Starts the given job.spec.command and sets `job.command_proc`."""
    if job.command_proc is not None:
        return  # Already running.
    # If a log dir exists for this job, download it. This can happen if a job is resumed.
    remote_log = os.path.join(remote_log_dir, job.spec.name)
    local_log = os.path.join(_LOG_DIR, job.spec.name)
    try:
        tf_io.gfile.copy(remote_log, local_log, overwrite=True)
    except tf_errors.NotFoundError:
        pass
    # Pipe all outputs to the local _LOG_DIR.
    job.command_proc = _piped_popen(job.spec.command, local_log, env_vars=job.spec.env_vars)
    logging.info("Started command for the job %s: %s", job.spec.name, job.spec.command)


def _start_cleanup_command(job: Job):
    """Starts the given job.spec.cleanup_command."""
    if not job.spec.cleanup_command:
        logging.info("Job %s has no cleanup command.", job.spec.name)
    elif job.cleanup_proc is None:
        # Pipe all outputs to a local _LOG_DIR.
        job.cleanup_proc = _piped_popen(
            job.spec.cleanup_command,
            f"{os.path.join(_LOG_DIR, job.spec.name)}.cleanup",
            env_vars=job.spec.env_vars,
        )
        logging.info(
            "Started cleanup command for the job %s: %s",
            job.spec.name,
            job.spec.cleanup_command,
        )


def _listdir(path: str) -> List[str]:
    """Wraps tf_io.gfile.listdir by returning empty list if dir is not found."""
    try:
        return tf_io.gfile.listdir(path)
    except tf_errors.NotFoundError:
        return []


def _remove(path: str):
    """Wraps tf_io.gfile.remove by catching not found errors."""
    try:
        tf_io.gfile.remove(path)
    except tf_errors.NotFoundError:
        pass


def download_job_batch(
    *,
    spec_dir: str,
    state_dir: str,
    user_state_dir: str,
    local_spec_dir: str = _JOB_DIR,
    verbose: bool = False,
) -> Tuple[Dict[str, Job], Set[str]]:
    """Downloads a batch of jobs.

    Args:
        spec_dir: Directory to look for job specs.
        state_dir: Directory to look for job states.
        user_state_dir: Directory to look for user states.
        local_spec_dir: Directory to store downloaded job specs.
        verbose: Verbose logging.

    Returns:
        A mapping from job name to Job(spec, state), and
        A set of job names whose state originates from user_state_dir.
    """
    jobspecs = _listdir(spec_dir)
    user_states = _listdir(user_state_dir)
    if verbose:
        logging.info("User states %s", user_states)

    # Download all files from spec_dir, state_dir, and user_state_dir.
    with ThreadPoolExecutor() as pool:
        download_spec_fn = functools.partial(
            _download_jobspec,
            remote_dir=spec_dir,
            local_dir=local_spec_dir,
        )
        spec_futs = {job_name: pool.submit(download_spec_fn, job_name) for job_name in jobspecs}
        job_state_futs = {
            job_name: pool.submit(_download_job_state, job_name, remote_dir=state_dir)
            for job_name in jobspecs
        }
        user_state_futs = {
            job_name: pool.submit(_download_job_state, job_name, remote_dir=user_state_dir)
            for job_name in user_states
        }
        wait(spec_futs.values())
        wait(job_state_futs.values())
        wait(user_state_futs.values())
        # Construct Jobs for each spec. The state of the job depends on the following:
        # 1. User state must be CANCELLING. We ignore other user states, e.g., a user should not be
        #     able to bypass scheduling by initiating a state change to ACTIVE.
        # 2. Job state must not be CLEANING/COMPLETED, since it doesn't make sense to progress
        #     backwards to CANCELLING.
        #
        # If these conditions are met, we pick the user state; otherwise, we keep job state.
        # We also keep track of which jobs have a user state (whether it was used or not), so that
        # we can handle the appropriate cleanup in the update step.
        jobs = {}
        jobs_with_user_states = set()
        for job_name in jobspecs:
            try:
                spec = spec_futs[job_name].result()
                state = job_state_futs[job_name].result()
                if job_name in user_state_futs:
                    user_state = user_state_futs[job_name].result()
                else:
                    user_state = None
            except Exception as e:  # pylint: disable=broad-except
                # TODO(markblee): Distinguish transient vs non-transient errors.
                logging.warning("Failed to load job %s with error: %s", job_name, e)
                continue

            if user_state is not None:
                if user_state == JobState.CANCELLING and state not in (
                    JobState.CLEANING,
                    JobState.COMPLETED,
                ):
                    state = user_state
                else:
                    logging.warning(
                        "User state (%s) ignored for job %s (%s).", user_state, job_name, state
                    )
                # Even if user_state is ignored, we still want to clean it up.
                jobs_with_user_states.add(job_name)
            jobs[job_name] = Job(spec=spec, state=state, command_proc=None, cleanup_proc=None)
    return jobs, jobs_with_user_states


def _load_runtime_options(bastion_dir: str) -> Dict[str, Any]:
    """Loads runtime option(s) from file, or returns {} on failure."""
    flag_file = os.path.join(bastion_dir, "runtime_options")
    try:
        with tf_io.gfile.GFile(flag_file, "r") as f:
            return json.load(f)
    except (tf_errors.NotFoundError, json.JSONDecodeError) as e:
        logging.warning("Failed to load runtime options: %s", e)
    return {}


def set_runtime_options(bastion_dir: str, **kwargs) -> Nested[Any]:
    """Writes key, value pairs into runtime options file. None values are removed."""
    runtime_options = _load_runtime_options(bastion_dir)
    runtime_options = merge(runtime_options, kwargs)
    flag_file = os.path.join(bastion_dir, "runtime_options")
    with tf_io.gfile.GFile(flag_file, "w") as f:
        json.dump(runtime_options, f)
    logging.info("Updated runtime options: %s", runtime_options)
    return runtime_options


class Bastion(Configurable):
    """An orchestrator that schedules and executes jobs."""

    @config_class
    class Config(Configurable.Config):
        """Configures Bastion."""

        # Interval to sync and run jobs.
        update_interval_seconds: float = 30
        # Scheduler to decide whether to start/pre-empt jobs.
        scheduler: Required[Scheduler.Config] = REQUIRED
        # Cleaner to deprovision idle resources.
        cleaner: Required[Cleaner.Config] = REQUIRED
        # Utility to sync logs to output_dir.
        uploader: Required[Uploader.Config] = REQUIRED
        # Output directory. Must be compatible with tf_io.
        output_dir: Required[str] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Remote directory to emit logs.
        self._output_dir = cfg.output_dir
        # Remote log output dir. Ensure trailing slash.
        # Note: pathlib doesn't work well with gs:// prefix.
        self._log_dir = os.path.join(self._output_dir, "logs")
        self._job_dir = os.path.join(self._output_dir, "jobs")
        # Remote history dir. Ensure trailing slash.
        self._job_history_dir = os.path.join(self._output_dir, "history", "jobs")
        tf_io.gfile.makedirs(self._job_history_dir)
        self._project_history_dir = os.path.join(self._output_dir, "history", "projects")
        tf_io.gfile.makedirs(self._project_history_dir)
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
        # A set of job names which require cleanup of user states.
        self._jobs_with_user_states: Set[str] = set()
        # Runtime options.
        self._runtime_options = {}

        # Instantiate children.
        self._scheduler = cfg.scheduler.instantiate()
        self._cleaner = cfg.cleaner.instantiate()
        self._uploader = cfg.uploader.set(src_dir=_LOG_DIR, dst_dir=self._log_dir).instantiate()

    def _append_to_job_history(self, job: Job, msg: str):
        with tf_io.gfile.GFile(os.path.join(self._job_history_dir, f"{job.spec.name}"), "a") as f:
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
            tf_io.gfile.makedirs(project_dir)
            with tf_io.gfile.GFile(os.path.join(project_dir, now.strftime("%Y%m%d")), "a") as f:
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

    def _load_runtime_options(self):
        """Loads (updated) runtime options from remote."""
        self._runtime_options = _load_runtime_options(self._output_dir)

    def _get_runtime_options(self, key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Reads runtime options with the given key.

        The defaults will be used as a schema to validate the runtime options.
        If validation fails, we return the defaults.
        """
        options = self._runtime_options.get(key, {})
        try:
            # By default tf_nest does not check types of atoms/leaves.
            def check_leaves(x, y):
                assert type(x) == type(y)  # pylint: disable=unidiomatic-typecheck

            tf_nest.map_structure(check_leaves, options, defaults)
        except (TypeError, ValueError, AssertionError) as e:
            logging.warning("Ignoring invalid runtime options %s: %s, %s", key, options, e)
            options = defaults
        return options

    def _wait_and_close_proc(self, proc: _PipedProcess, kill: bool = False):
        """Cleans up the process/fds and upload logs to gs."""
        if kill:
            send_signal(proc.popen, sig=signal.SIGKILL)
        # Note: proc should already be polled and completed, so wait is nonblocking.
        proc.popen.wait()
        proc.fd.close()
        # Upload outputs to log dir.
        _catch_with_error_log(
            tf_io.gfile.copy,
            proc.fd.name,
            os.path.join(self._log_dir, os.path.basename(proc.fd.name)),
            overwrite=True,
        )
        # Remove the local output file.
        if os.path.exists(proc.fd.name):
            os.remove(proc.fd.name)

    def _sync_jobs(self):
        """Makes the local bastion state consistent with the remote state.

        This function serves as a synchronization point for user-initiated state changes
        ("user_states") and state changes from a prior `_update_job` ("states"). Users should avoid
        writing to the "states" dir directly, as doing so can produce races with `_update_job`.

        More specifically, this function:
        1. Downloads all active jobspecs from remote job dir.
        2. Downloads all statefiles for active jobspecs from remote state dir (see
            `download_job_batch` for details).

        We use these jobspecs to update the local self._active_jobs.
        """
        active_jobs, jobs_with_user_states = download_job_batch(
            spec_dir=self._active_dir,
            state_dir=self._state_dir,
            user_state_dir=self._user_state_dir,
            verbose=True,
        )
        self._jobs_with_user_states = jobs_with_user_states
        # Iterate over unique job names.
        # pylint: disable-next=use-sequence-for-iteration
        for job_name in {*active_jobs.keys(), *self._active_jobs.keys()}:
            # Detected new job: exists in remote, but not local.
            if job_name not in self._active_jobs:
                logging.info("Detected new job %s.", job_name)
                self._active_jobs[job_name] = active_jobs[job_name]
                self._append_to_job_history(active_jobs[job_name], "PENDING: detected jobspec")
            # Detected removed job: exists locally, but not in remote.
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
        1. A jobspec file exists in the remote job dir at the start of each call. The job.state
            provided to this function call is consistent with that state in the remote dir + any
            scheduling decisions.
        2. The function may be called by a freshly started bastion (recovering from pre-emption).
            Thus each condition must assume nothing about the local state.
        3. The function may be pre-empted at any point.
        4. Job commands/cleanup commands are resumable (can invoke the same command multiple times).

        Conditions that must be held at exit (either pre-emption or graceful):
        1. A jobspec must still exist in the remote job dir.
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
                _upload_jobspec(job.spec, local_dir=_JOB_DIR, remote_dir=self._complete_dir)
                os.remove(local_jobspec)

        else:
            raise ValueError(f"Unexpected state: {job.state}")

        # Flush the state to remote.
        # TODO(markblee): Skip the upload if we can detect the state hasn't changed.
        # State changes can come from user_states, scheduling, or above, so probably not worth the
        # complexity at the moment, given how small job states are.
        _upload_job_state(job.spec.name, job.state, remote_dir=self._state_dir, verbose=False)

        # Remove any remote "user_states" now that "states" dir has synchronized.
        if job.spec.name in self._jobs_with_user_states:
            _remove(os.path.join(self._user_state_dir, job.spec.name))

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
        schedule_options = self._get_runtime_options(
            "scheduler",
            defaults={"dry_run": False, "verbosity": 0},
        )
        schedule_results: Scheduler.ScheduleResults = self._scheduler.schedule(
            schedulable_jobs,
            dry_run=schedule_options["dry_run"],
            verbosity=schedule_options["verbosity"],
        )
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
        will be made consistent with remote state.
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
            _remove(os.path.join(self._active_dir, job_name))
            _remove(os.path.join(self._state_dir, job_name))
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

    def execute(self):
        """Starts the bastion."""
        cfg: Bastion.Config = self.config
        if os.path.exists(_JOB_DIR):
            shutil.rmtree(_JOB_DIR)
        os.makedirs(_LOG_DIR, exist_ok=True)
        os.makedirs(_JOB_DIR, exist_ok=True)
        while True:
            start = time.time()
            self._uploader()
            self._load_runtime_options()
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


class StartBastionJob(CloudJob):
    """A job that runs the bastion."""

    @config_class
    class Config(CloudJob.Config):
        """Configures StartBastionJob."""

        bastion: Required[Bastion.Config] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        return super().default_config().set(command="")

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._bastion: Bastion = cfg.bastion.instantiate()

    def _execute(self) -> Any:
        # Wraps bastion with retries.
        self._bastion.execute()


class SubmitBastionJob(CloudJob):
    """A job to submit a command to bastion."""

    @config_class
    class Config(CloudJob.Config):
        """Configures SubmitBastionJob."""

        # Name of the job. Not to be confused with cfg.name, the bastion name.
        job_name: Required[str] = REQUIRED
        # Job spec file local path.
        job_spec_file: Required[str] = REQUIRED
        # Output directory used by the bastion. Note that this must be consistent with the output
        # directory used when creating the bastion.
        bastion_dir: Required[str] = REQUIRED

    @classmethod
    def default_config(cls):
        return super().default_config().set(command="")

    @property
    def bastion_dir(self):
        return self.config.bastion_dir

    def _job_dir(self):
        return os.path.join(self.bastion_dir, "jobs")

    def _delete(self):
        cfg: SubmitBastionJob.Config = self.config
        try:
            jobspec = os.path.join(self._job_dir(), "active", cfg.job_name)
            if not tf_io.gfile.exists(jobspec):
                raise ValueError(f"Unable to locate jobspec {jobspec}")
            _upload_job_state(
                cfg.job_name,
                JobState.CANCELLING,
                remote_dir=os.path.join(self._job_dir(), "user_states"),
            )
            logging.info("Job %s is cancelling.", cfg.job_name)
            # Poll for jobspec to be removed.
            while tf_io.gfile.exists(jobspec):
                logging.info("Waiting for job to stop (which usually takes a few minutes)...")
                time.sleep(10)
            logging.info("Job is stopped.")
        except ValueError as e:
            logging.info("Failed with error: %s -- Has the job been cancelled already?", e)

    def _execute(self):
        cfg: SubmitBastionJob.Config = self.config
        dst = os.path.join(self._job_dir(), "active", cfg.job_name)
        if tf_io.gfile.exists(dst):
            logging.info("\n\nNote: Job is already running. To restart it, cancel the job first.\n")
        else:
            # Upload the job for bastion to pickup.
            tf_io.gfile.copy(cfg.job_spec_file, dst)
