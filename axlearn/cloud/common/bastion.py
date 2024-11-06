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
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from subprocess import CalledProcessError
from typing import IO, Any, Optional, Union

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
from axlearn.cloud.common.event_queue import BaseQueueClient, Event
from axlearn.cloud.common.quota import QuotaFn
from axlearn.cloud.common.scheduler import BaseScheduler, JobMetadata, JobScheduler, ResourceMap
from axlearn.cloud.common.types import JobSpec
from axlearn.cloud.common.uploader import Uploader
from axlearn.cloud.common.utils import merge, send_signal
from axlearn.common.config import (
    REQUIRED,
    Configurable,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)
from axlearn.common.utils import Nested

_LATEST_BASTION_VERSION = 1  # Determines job schema (see JobSpec).
_LOG_DIR = "/var/tmp/logs"  # Use /var/tmp/ since /tmp/ is cleared every 10 days.
_JOB_DIR = "/var/tmp/jobs"
_BASTION_SERIALIZED_JOBSPEC_ENV_VAR = "_BASTION_SERIALIZED_JOBSPEC"
BASTION_JOB_VERSION_ENV_VAR = "BASTION_JOB_VERSION"

FLAGS = flags.FLAGS

_VALID_NAME_CHARS = r"[!-~]+"  # match all printing ASCII characters except space
valid_name_re = re.compile(_VALID_NAME_CHARS)


def bastion_job_flags(flag_values: flags.FlagValues = FLAGS):
    flags.DEFINE_string("name", None, "Name of bastion.", flag_values=flag_values, required=True)
    flags.DEFINE_string("job_name", None, "Name of job.", flag_values=flag_values)
    flags.DEFINE_string("spec", None, "Path to a job spec.", flag_values=flag_values)


# The following functions, `_download`, `_readfile`, `_listdir`, and `_remove`, can be patched to
# support alternative storages that cannot be accessed via gfile.
#
# TODO(ruoming): refactor them to a `BastionDirStorage` class.
def _download(path: str, local_file: str):
    tf_io.gfile.copy(path, local_file, overwrite=True)


def _readfile(path: str) -> str:
    with tf_io.gfile.GFile(path, mode="r") as f:
        return f.read()


def _listdir(path: str) -> list[str]:
    """Wraps tf_io.gfile.listdir by returning empty list if dir is not found."""
    try:
        return tf_io.gfile.listdir(path)
    except tf_errors.NotFoundError:
        return []


def _remove(path: str):
    """Wraps tf_io.gfile.remove by catching not found errors."""
    try:
        if tf_io.gfile.isdir(path):
            tf_io.gfile.rmtree(path)
        else:
            tf_io.gfile.remove(path)
    except tf_errors.NotFoundError:
        pass


# Subclass str to be JSON serializable: https://stackoverflow.com/a/51976841
class JobStatus(str, enum.Enum):
    """See Bastion._update_job for state handling."""

    # Job is queued. Any running command will be forcefully terminated.
    PENDING = "PENDING"
    # Job is about to run, or currently running.
    ACTIVE = "ACTIVE"
    # Job is cancelling. Command is terminating.
    CANCELLING = "CANCELLING"
    # Job has completed/terminated the command, is running cleanup command (if any).
    CLEANING = "CLEANING"
    # Job is complete.
    COMPLETED = "COMPLETED"


# Subclass str to be JSON serializable: https://stackoverflow.com/a/51976841
class JobLifecycleState(str, enum.Enum):
    """Represents a lifecycle state for a job.

    The lifecycle state is meant for fine-grained reporting and tracking of job state transitions.
    For the states corresponding to bastion's internal state machine, see `JobStatus`.
    """

    # Job is queued. Bastion detects the new job's jobspec.
    QUEUED = "QUEUED"
    # Job is starting. Command is start to run.
    STARTING = "STARTING"
    # Job is running.
    RUNNING = "RUNNING"
    # Job is pre-empting.
    PREEMPTING = "PREEMPTING"
    # Job is rescheduling.
    RESCHEDULING = "RESCHEDULING"
    # Job is updating.
    UPDATING = "UPDATING"
    # Job is cancelling. Command is terminating.
    CANCELLING = "CANCELLING"
    # Job has completed/terminated the command, is running cleanup command (if any).
    CLEANING = "CLEANING"
    # Job is failed.
    FAILED = "FAILED"
    # Job finished successfully.
    SUCCEEDED = "SUCCEEDED"
    # Job is complete.
    COMPLETED = "COMPLETED"


@dataclasses.dataclass
class JobLifecycleEvent(Event):
    """Represents a lifecycle event for a job.

    Attributes:
        job_name: The name of the job associated with this event.
        state: The state of the job.
        details: The details of the state info.
        job_id: An optional identifier for the job. Defaults to None.
    """

    job_name: str
    state: JobLifecycleState
    details: str
    job_id: Optional[str] = None

    def serialize(self) -> str:
        """Serializes the job lifecycle event into a JSON string."""
        job_event = {
            "job_name": self.job_name,
            "job_id": self.job_id,
            "message": self.details,
            "state": self.state,
            "timestamp": time.time_ns(),
        }
        return json.dumps(job_event)

    def __repr__(self) -> str:
        """Custom string representation for logging."""
        return (
            f"JobLifecycleEvent(job_id={self.job_id}, job_name={self.job_name}, state={self.state})"
        )


class ValidationError(ValueError):
    """Validation failure (e.g. JobSpec deserialization)."""


def _validate_job_metadata(metadata: JobMetadata):
    """Validates the given metadata."""
    if not isinstance(metadata.user_id, str):
        raise ValidationError(f"Expected {metadata.user_id=} to be a string.")
    if not isinstance(metadata.project_id, str):
        raise ValidationError(f"Expected {metadata.project_id=} to be a string.")
    if not isinstance(metadata.resources, dict):
        raise ValidationError(f"Expected {metadata.resources=} to be a dict.")
    if not all(isinstance(k, str) and isinstance(v, int) for k, v in metadata.resources.items()):
        raise ValidationError(f"Expected {metadata.resources=} to have string keys and int values.")
    if not isinstance(metadata.priority, int):
        raise ValidationError(f"Expected {metadata.priority=} to be an int.")
    if metadata.version is not None and not isinstance(metadata.version, int):
        raise ValidationError(f"Expected {metadata.version=} to be None or an int.")


def _validate_jobspec(jobspec: JobSpec):
    """Validates the given jobspec.

    Note that type annotations are insufficient as the jobspec can be deserialized from json.
    """
    if not isinstance(jobspec.name, str):
        raise ValidationError(f"Expected {jobspec.name=} to be a string.")
    if not isinstance(jobspec.command, str):
        raise ValidationError(f"Expected {jobspec.command=} to be a string.")
    if not (jobspec.cleanup_command is None or isinstance(jobspec.cleanup_command, str)):
        raise ValidationError(f"Expected {jobspec.cleanup_command=} to be None or string.")

    # Validate env vars.
    if not (jobspec.env_vars is None or isinstance(jobspec.env_vars, dict)):
        raise ValidationError(f"Expected {jobspec.env_vars=} to be None or dict.")
    if jobspec.env_vars:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in jobspec.env_vars.items()):
            raise ValidationError(f"Expected {jobspec.env_vars=} to have string keys and values.")

    # Validate metadata.
    if not isinstance(jobspec.metadata, JobMetadata):
        raise ValidationError(f"Expected {jobspec.metadata=} to be JobMetadata.")
    _validate_job_metadata(jobspec.metadata)


def new_jobspec(
    *,
    name: str,
    command: str,
    metadata: JobMetadata,
    cleanup_command: Optional[str] = None,
    env_vars: Optional[dict[str, str]] = None,
    version: int = _LATEST_BASTION_VERSION,
) -> JobSpec:
    """Constructs a JobSpec with basic schema validation."""
    jobspec = JobSpec(
        version=version,
        name=name,
        command=command,
        cleanup_command=cleanup_command,
        env_vars=env_vars,
        metadata=metadata,
    )
    _validate_jobspec(jobspec)
    return jobspec


def serialize_jobspec(spec: JobSpec, f: Union[str, IO]):
    """Writes job spec to filepath or file."""
    if isinstance(f, str):
        with open(f, "w", encoding="utf-8") as fd:
            serialize_jobspec(spec, fd)
            return
    data = dataclasses.asdict(spec)
    # Use an explicit date format instead of str() to ensure that microseconds
    # are included even if they are 0.
    data["metadata"]["creation_time"] = data["metadata"]["creation_time"].strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    json.dump(data, f, default=str)
    f.flush()


def deserialize_jobspec(f: Union[str, IO]) -> JobSpec:
    """Loads job spec from filepath or file."""
    if isinstance(f, str):
        with open(f, encoding="utf-8") as fd:
            return deserialize_jobspec(fd)

    data: dict = json.load(f)
    if data["version"] == _LATEST_BASTION_VERSION:
        data["metadata"]["creation_time"] = datetime.strptime(
            data["metadata"]["creation_time"], "%Y-%m-%d %H:%M:%S.%f"
        )
        return new_jobspec(
            version=data["version"],
            name=data["name"],
            command=data["command"],
            cleanup_command=data.get("cleanup_command", None),
            env_vars=data.get("env_vars", None),
            metadata=JobMetadata(**data["metadata"]),
        )
    raise ValidationError(f"Unsupported version: {data['version']}")


def is_valid_job_name(name: str) -> bool:
    """Ensures job name is not path-like and only contains safe characters.

    This check should avoid making assumptions about the underlying compute environment.
    """
    return (
        bool(name)
        and ("/" not in name)
        and (name not in (".", ".."))
        and bool(valid_name_re.fullmatch(name))
    )


def _download_jobspec(job_name: str, *, remote_dir: str, local_dir: str = _JOB_DIR) -> JobSpec:
    """Loads jobspec from gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    local_file = os.path.join(local_dir, job_name)
    _download(remote_file, local_file)
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


def _piped_popen(cmd: str, f: str, *, env_vars: Optional[dict[str, str]] = None) -> _PipedProcess:
    """Runs cmd in the background, piping stdout+stderr to a file."""
    # Open with "a" to append to an existing logfile, if any.
    fd = open(f, "a", encoding="utf-8")
    # Make a copy of system env.
    env_vars_copy = os.environ.copy()
    if env_vars:
        # Inject supplied env.
        env_vars_copy.update(env_vars)

    # Ensure that all env var values are strings.
    env_vars_copy = {k: str(v) for k, v in env_vars_copy.items()}
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
class JobState:
    """Bastion job state.

    Attributes:
        status: Job status.
        metadata: Additional metadata.
    """

    status: JobStatus
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Job:
    """A bastion job.

    Attributes:
        spec: Job spec.
        state: Job state.
        command_proc: Optional process for the main command.
        cleanup_proc: Optional process for the cleanup command.
    """

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
        contents = _readfile(remote_file)
        try:
            state = json.loads(contents)
        except json.JSONDecodeError:
            # For backwards compatibility, interpret as status.
            state = dict(status=contents)
        state["status"] = JobStatus[state["status"].strip().upper()]
        return JobState(**state)
    except tf_errors.NotFoundError:
        # No job state, defaults to PENDING.
        return JobState(status=JobStatus.PENDING)


def _upload_job_state(job_name: str, state: JobState, *, remote_dir: str, verbose: bool = True):
    """Uploads job state to gs path."""
    remote_file = os.path.join(remote_dir, job_name)
    logging.log_if(logging.INFO, "Writing %s to %s.", verbose, state.status.name, remote_file)
    with tf_io.gfile.GFile(remote_file, mode="w") as f:
        json.dump(dataclasses.asdict(state), f)


def _start_command(job: Job, *, remote_log_dir: str, env_vars: dict):
    """Starts the given job.spec.command and sets `job.command_proc`."""
    if job.command_proc is not None:
        return  # Already running.
    # If a log dir exists for this job, download it. This can happen if a job is resumed.
    remote_log = os.path.join(remote_log_dir, job.spec.name)
    local_log = os.path.join(_LOG_DIR, job.spec.name)
    try:
        _download(remote_log, local_log)
    except tf_errors.NotFoundError:
        pass
    # Pipe all outputs to the local _LOG_DIR.
    job.command_proc = _piped_popen(
        job.spec.command, local_log, env_vars={**env_vars, **(job.spec.env_vars or {})}
    )
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


def download_job_batch(
    *,
    spec_dir: str,
    state_dir: str,
    user_state_dir: str,
    local_spec_dir: str = _JOB_DIR,
    verbose: bool = False,
    remove_invalid_job_specs: bool = False,
    quota: Optional[QuotaFn] = None,
) -> tuple[dict[str, Job], set[str]]:
    """Downloads a batch of jobs.

    Args:
        spec_dir: Directory to look for job specs.
        state_dir: Directory to look for job states.
        user_state_dir: Directory to look for user states.
        local_spec_dir: Directory to store downloaded job specs.
        verbose: Verbose logging.
        remove_invalid_job_specs: Whether to remove invalid job specs.
        quota: A thunk returning the UserQuotaInfo to use for validating
               user project membership.
               If None, do not validate project membership.

    Returns:
        A mapping from job name to Job(spec, state), and
        A set of job names whose state originates from user_state_dir.
    """
    jobspecs = []
    invalid_jobspecs = []
    user_states = []
    invalid_user_states = []

    for job_name in _listdir(spec_dir):
        if is_valid_job_name(job_name):
            jobspecs.append(job_name)
        else:
            invalid_jobspecs.append(job_name)

    for job_name in _listdir(user_state_dir):
        if is_valid_job_name(job_name):
            user_states.append(job_name)
        else:
            invalid_user_states.append(job_name)

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

        if quota is not None:
            quota_info = quota()
        else:
            quota_info = None

        wait(spec_futs.values())
        wait(job_state_futs.values())
        wait(user_state_futs.values())

        # Construct Jobs for each spec. The status of the job depends on the following:
        # 1. User state must be CANCELLING. We ignore other user states, e.g., a user should not be
        #     able to bypass scheduling by initiating a state change to ACTIVE.
        # 2. Job state must not be CLEANING/COMPLETED, since it doesn't make sense to progress
        #     backwards to CANCELLING.
        # 3. We ignore any user supplied metadata, e.g., a user should not be able to influence
        #     scheduling tiers.
        #
        # If these conditions are met, we pick the user status; otherwise, we keep job status.
        # We also keep track of which jobs have a user state (whether it was used or not), so that
        # we can handle the appropriate cleanup in the update step.
        jobs = {}
        jobs_with_user_states = set()
        for job_name in jobspecs:
            try:
                spec: JobSpec = spec_futs[job_name].result()
                state = job_state_futs[job_name].result()
                if job_name in user_state_futs:
                    user_state = user_state_futs[job_name].result()
                else:
                    user_state = None
                if quota_info is not None:
                    user_id = spec.metadata.user_id
                    project_id = spec.metadata.project_id
                    if project_id not in quota_info.user_projects(user_id):
                        # TODO(markblee): surface invalid specs in job history.
                        logging.warning(
                            "Job %s will be removed because user %s is not a member of %s",
                            job_name,
                            user_id,
                            project_id,
                        )
                        invalid_jobspecs.append(job_name)
                        continue
            except ValidationError as e:
                logging.warning("Job %s failed validation and will be removed: %s", job_name, e)
                invalid_jobspecs.append(job_name)
                continue
            except Exception as e:  # pylint: disable=broad-except
                # TODO(markblee): Distinguish transient vs non-transient errors.
                logging.warning("Failed to load job %s with error: %s", job_name, e)
                continue

            if user_state is not None:
                if user_state.status == JobStatus.CANCELLING and state.status not in (
                    JobStatus.CLEANING,
                    JobStatus.COMPLETED,
                ):
                    # Only copy the status, not the metadata.
                    state.status = user_state.status
                else:
                    logging.warning(
                        "User state (%s) ignored for job %s (%s).", user_state, job_name, state
                    )
                # Even if user_state is ignored, we still want to clean it up.
                jobs_with_user_states.add(job_name)
            jobs[job_name] = Job(spec=spec, state=state, command_proc=None, cleanup_proc=None)

        # Remove invalid jobspecs and user states.
        if remove_invalid_job_specs and (invalid_jobspecs or invalid_user_states):
            logging.info(
                "Removing invalid jobspecs: %s and user states: %s",
                invalid_jobspecs,
                invalid_user_states,
            )
            pool.map(
                _remove,
                [os.path.join(spec_dir, job_name) for job_name in invalid_jobspecs]
                + [os.path.join(user_state_dir, job_name) for job_name in invalid_user_states],
            )
    return jobs, jobs_with_user_states


def _load_runtime_options(bastion_dir: str) -> dict[str, Any]:
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
        scheduler: Required[JobScheduler.Config] = REQUIRED
        # Cleaner to deprovision idle resources.
        cleaner: Required[Cleaner.Config] = REQUIRED
        # Utility to sync logs to output_dir.
        uploader: Required[Uploader.Config] = REQUIRED
        # Output directory. Must be compatible with tf_io.
        output_dir: Required[str] = REQUIRED
        # The quota function to use for getting user group membership and group quota.
        quota: Required[QuotaFn] = REQUIRED
        # The event publisher sends events into queue.
        event_publisher: Optional[BaseQueueClient.Config] = None

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
        self._active_jobs: dict[str, Job] = {}
        # A set of job names which require cleanup of user states.
        self._jobs_with_user_states: set[str] = set()
        # Runtime options.
        self._runtime_options = {}
        # The QuotaFn that returns quota for scheduling.
        self._quota: QuotaFn = cfg.quota.instantiate()

        # Instantiate children.
        self._scheduler: JobScheduler = cfg.scheduler.set(
            quota=config_for_function(lambda: self._quota)
        ).instantiate()
        self._cleaner: Cleaner = cfg.cleaner.instantiate()
        self._uploader = cfg.uploader.set(src_dir=_LOG_DIR, dst_dir=self._log_dir).instantiate()
        self._event_publisher = maybe_instantiate(cfg.event_publisher)

    def _append_to_job_history(self, job: Job, *, msg: str, state: JobLifecycleState):
        with tf_io.gfile.GFile(os.path.join(self._job_history_dir, f"{job.spec.name}"), "a") as f:
            curr_time = datetime.now(timezone.utc).strftime("%m%d %H:%M:%S")
            f.write(f"{curr_time} {msg}\n")
        # Publish event into queue.
        if self._event_publisher:
            self._event_publisher.publish(
                JobLifecycleEvent(
                    job_name=job.spec.name,
                    job_id=job.spec.metadata.job_id,
                    state=state,
                    details=msg,
                )
            )

    def _append_to_project_history(
        self, jobs: dict[str, JobMetadata], schedule_results: BaseScheduler.ScheduleResults
    ):
        now = datetime.now(timezone.utc)
        for project_id, limits in schedule_results.project_limits.items():
            job_verdicts = schedule_results.job_verdicts.get(project_id, {})
            verdicts = []
            for job_id, verdict in job_verdicts.items():
                verdicts.append((job_id, verdict.should_run(), verdict.metadata))
            verdicts = sorted(verdicts)
            previous_verdicts = self._project_history_previous_verdicts.get(project_id)
            if previous_verdicts == verdicts:
                # Nothing changed.
                continue
            self._project_history_previous_verdicts[project_id] = verdicts
            # Mapping from resource types to usage.
            project_usage = collections.defaultdict(int)
            running_jobs = []
            queued_jobs = []
            for job_id, verdict in job_verdicts.items():
                if verdict.should_run():
                    running_jobs.append((job_id, verdict.metadata))
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
                for job_id, metadata in running_jobs:
                    f.write(f"  {job_id} ({metadata})\n")
                f.write("Queued jobs:\n")
                for job_id in queued_jobs:
                    f.write(f"  {job_id}\n")

    def _load_runtime_options(self):
        """Loads (updated) runtime options from remote."""
        self._runtime_options = _load_runtime_options(self._output_dir)

    def _get_runtime_options(self, key: str, defaults: dict[str, Any]) -> dict[str, Any]:
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

    def _kill_job(self, job: Job):
        """Terminates any processes with SIGKILL.

        Note that killing these processes does not affect the job state (i.e., this does not cause
        jobs to be cancelled). This is expected as bastion jobs are typically pre-emptible (see
        docstring for bastion job requirements).
        """
        if job.command_proc is not None:
            self._wait_and_close_proc(job.command_proc, kill=True)
        if job.cleanup_proc is not None:
            self._wait_and_close_proc(job.cleanup_proc, kill=True)

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
            remove_invalid_job_specs=True,
            quota=self._quota,
        )
        self._jobs_with_user_states = jobs_with_user_states
        # Iterate over unique job names.
        # pylint: disable-next=use-sequence-for-iteration
        for job_name in {*active_jobs.keys(), *self._active_jobs.keys()}:
            # Detected new job: exists in remote, but not local.
            if job_name not in self._active_jobs:
                logging.info("Detected new job %s.", job_name)
                self._active_jobs[job_name] = active_jobs[job_name]
                self._append_to_job_history(
                    active_jobs[job_name],
                    msg="PENDING: detected jobspec",
                    # When Bastion restarts, we will see this for every job.
                    # Leave to consumer to handle this case.
                    state=JobLifecycleState.QUEUED,
                )
            # Detected removed job: exists locally, but not in remote.
            elif job_name not in active_jobs:
                job = self._active_jobs[job_name]
                if job.state.status != JobStatus.COMPLETED:
                    logging.warning("Detected orphaned job %s! Killing it...", job.spec.name)
                    self._kill_job(job)
                logging.info("Removed job %s.", job_name)
                del self._active_jobs[job_name]
            # Detected updated job: exists in both.
            else:
                curr_job = self._active_jobs[job_name]
                updated_job = active_jobs[job_name]
                if updated_job.spec.metadata.version != curr_job.spec.metadata.version:
                    # When a new version is detected, add "updated" in the metadata to signal
                    # job state change and job relaunch.
                    # Note: "updated" is a transient state and should not be persisted.
                    updated_job.state.metadata["updated"] = True
                    logging.info("Detected a different version of job %s", job_name)
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
        if job.state.status == JobStatus.PENDING:
            # Forcefully terminate the command proc and fd, if they exist, and sync logs to remote.
            # The forceful termination is similar to the behavior when bastion itself is pre-empted.
            #
            # We must also ensure that:
            # 1. command_proc is set to None, so we can resume in ACTIVE in a subsequent step
            #    (possibly the next step).
            # 2. Any job logs are sync'ed to remote log dir. The local log file cannot reliably be
            #    expected to be present if/when the job is resumed.
            if job.command_proc is not None:
                self._append_to_job_history(
                    job, msg="PENDING: pre-empting", state=JobLifecycleState.PREEMPTING
                )
                logging.info("Pre-empting job: %s", job.spec.name)
                self._wait_and_close_proc(job.command_proc, kill=True)
                job.command_proc = None
                logging.info("Job is pre-empted: %s", job.spec.name)

        elif job.state.status == JobStatus.ACTIVE:
            # Run the command if not already started. We attempt to run every time, in case bastion
            # got pre-empted.
            if job.command_proc is None:
                self._append_to_job_history(
                    job,
                    msg=f"ACTIVE: start process command: {job.spec.command} "
                    f"with metadata: {job.state.metadata} and version: {job.spec.metadata.version}",
                    state=JobLifecycleState.STARTING,
                )
            env_vars = {f"BASTION_{k.upper()}": v for k, v in job.state.metadata.items()}

            if job.spec.metadata.version:
                # For backwards compatibility, only set the version in env when not None.
                env_vars.update({BASTION_JOB_VERSION_ENV_VAR: job.spec.metadata.version})

            serialized_jobspec = io.StringIO()
            serialize_jobspec(job.spec, serialized_jobspec)
            env_vars |= {_BASTION_SERIALIZED_JOBSPEC_ENV_VAR: serialized_jobspec.getvalue()}
            _start_command(
                job,
                remote_log_dir=self._log_dir,
                env_vars=env_vars,
            )
            assert job.command_proc is not None

            # If command is completed, move to CLEANING. Otherwise, it's still RUNNING.
            if _is_proc_complete(job.command_proc):
                self._append_to_job_history(
                    job, msg="CLEANING: process finished", state=JobLifecycleState.CLEANING
                )
                logging.info(
                    "Job %s stopped gracefully: %s.",
                    job.spec.name,
                    job.command_proc.popen.returncode,
                )
                job.state.status = JobStatus.CLEANING

        elif job.state.status == JobStatus.CANCELLING:
            # If job is still running, terminate it. We stay in CANCELLING until it has fully
            # exited, after which we move to CLEANING.
            if job.command_proc is not None and not _is_proc_complete(job.command_proc):
                self._append_to_job_history(
                    job,
                    msg="CANCELLING: terminating the process",
                    state=JobLifecycleState.CANCELLING,
                )
                logging.info("Sending SIGTERM to job: %s", job.spec.name)
                job.command_proc.popen.terminate()
            else:
                self._append_to_job_history(
                    job, msg="CLEANING: process terminated", state=JobLifecycleState.CLEANING
                )
                job.state.status = JobStatus.CLEANING

        elif job.state.status == JobStatus.CLEANING:
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
                    job,
                    msg=f"CLEANING: start cleanup command: {job.spec.cleanup_command}",
                    state=JobLifecycleState.CLEANING,
                )
            _start_cleanup_command(job)

            # If job has no cleanup command, or cleanup command is complete, transition to
            # COMPLETED.
            if job.cleanup_proc is None or _is_proc_complete(job.cleanup_proc):
                self._append_to_job_history(
                    job, msg="COMPLETED: cleanup finished", state=JobLifecycleState.COMPLETED
                )
                logging.info("Job %s finished running cleanup.", job.spec.name)
                if job.cleanup_proc is not None:
                    self._wait_and_close_proc(job.cleanup_proc)
                    job.cleanup_proc = None

                job.state.status = JobStatus.COMPLETED

        elif job.state.status == JobStatus.COMPLETED:
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
            if job.state.status in {JobStatus.PENDING, JobStatus.ACTIVE}:
                schedulable_jobs[job_name] = job.spec.metadata

        # Decide which jobs to resume/pre-empt.
        schedule_options = self._get_runtime_options(
            "scheduler",
            defaults={"dry_run": False, "verbosity": 0},
        )
        schedule_results: BaseScheduler.ScheduleResults = self._scheduler.schedule(
            schedulable_jobs,
            dry_run=schedule_options["dry_run"],
            verbosity=schedule_options["verbosity"],
        )
        self._append_to_project_history(schedulable_jobs, schedule_results)
        for verdicts in schedule_results.job_verdicts.values():
            for job_name, verdict in verdicts.items():
                job = self._active_jobs[job_name]
                assert job.state.status in {JobStatus.PENDING, JobStatus.ACTIVE}

                if verdict:
                    old_tier = job.state.metadata.get("tier")
                    new_tier = verdict.metadata.get("tier")
                    changed_tiers = old_tier != new_tier

                    jobspec_changed = job.state.metadata.get("updated")

                    # Jobspec changed, trigger a restart of the runner.
                    if jobspec_changed:
                        self._append_to_job_history(
                            job,
                            msg="UPDATING: Detected updated jobspec. Will restart the runner "
                            "by sending to PENDING state",
                            state=JobLifecycleState.UPDATING,
                        )
                        job.state.status = JobStatus.PENDING
                    elif job.state.status == JobStatus.PENDING or not changed_tiers:
                        # Resume if not running, or keep running if scheduling tier did not change.
                        job.state.status = JobStatus.ACTIVE
                    else:
                        # Job changed scheduling tiers, and must be restarted on the new tier.
                        # NOTE: this can possibly lead to thrashing of jobs that frequently switch
                        # tiers. One option is track per-job tier changes and hold off on promoting
                        # low priority to high priority if it was demoted recently.
                        # TODO(markblee): Add instrumentation to track frequency of tier changes to
                        # see whether this is necessary.
                        assert job.state.status == JobStatus.ACTIVE and changed_tiers
                        self._append_to_job_history(
                            job,
                            msg=f"Rescheduling at a different tier from {old_tier} to {new_tier}",
                            state=JobLifecycleState.RESCHEDULING,
                        )
                        job.state.status = JobStatus.PENDING
                else:
                    # Pre-empt/stay queued.
                    if job.command_proc is not None and _is_proc_complete(job.command_proc):
                        # As a slight optimization, we avoid pre-empting ACTIVE jobs that are
                        # complete, since we can directly transition to CLEANING.
                        job.state.status = JobStatus.ACTIVE
                    else:
                        job.state.status = JobStatus.PENDING
                        # Pending jobs which are not rescheduled should have no tier information.
                        verdict.metadata.pop("tier", None)

                job.state.metadata = verdict.metadata

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
        # These are jobs that are fully completed or fully pending. Note that some jobs may be
        # pending due to rescheduling tiers, which should not be cleaned -- instead, we let the
        # runner decide how/when to recreate the resources.
        jobs_to_clean = {}
        for job_name, job in self._active_jobs.items():
            if job.state.status == JobStatus.COMPLETED or (
                job.state.status == JobStatus.PENDING
                and job.state.metadata.get("tier", None) is None
            ):
                jobs_to_clean[job_name] = job.spec

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
            if self._active_jobs[job_name].state.status == JobStatus.COMPLETED
        ]
        logging.info("Fully cleaned COMPLETED jobs: %s", cleaned_completed)
        with ThreadPoolExecutor() as pool:
            pool.map(_delete_jobspec, cleaned_completed)

        logging.info("==End of gc step.")
        logging.info("")

    def _execute(self):
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

    def execute(self):
        """Starts the bastion."""
        try:
            self._execute()
        except Exception:
            logging.error("Caught exception, will cleanup all child jobs.")
            for job in self._active_jobs.values():
                try:
                    self._kill_job(job)
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning("Fail to kill a job with error: %s", e)
            self._active_jobs = {}
            self._uploader.cleanup()
            raise  # Re-raise.


class BastionDirectory(Configurable):
    """A directory watched by a Bastion.

    submit_job() writes the job spec to the active job dir watched by the bastion, which will
    start a process on the bastion VM to run the command when resources are ready. The on-bastion
    process usually further creates and watches remote VMs.

    cancel_job() writes a job state file to the cancel job dir watched by the bastion, which will
    terminate the on-bastion process.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures BastionDirectory."""

        # Directory watched by the bastion. Note that this must be consistent with the output
        # directory used when creating the bastion.
        root_dir: Required[str] = REQUIRED

    @property
    def root_dir(self):
        return self.config.root_dir

    def __str__(self) -> str:
        return self.root_dir

    @property
    def logs_dir(self):
        return os.path.join(self.root_dir, "logs")

    @property
    def active_job_dir(self):
        return os.path.join(self.root_dir, "jobs", "active")

    @property
    def complete_job_dir(self):
        return os.path.join(self.root_dir, "jobs", "complete")

    @property
    def job_states_dir(self):
        return os.path.join(self.root_dir, "jobs", "states")

    @property
    def user_states_dir(self):
        return os.path.join(self.root_dir, "jobs", "user_states")

    def list_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs, _ = download_job_batch(
                spec_dir=self.active_job_dir,
                state_dir=self.job_states_dir,
                user_state_dir=self.user_states_dir,
                local_spec_dir=tmpdir,
                remove_invalid_job_specs=False,
            )
            jobs: dict[str, Job] = dict(sorted(jobs.items(), key=lambda kv: kv[0]))
            return jobs

    def cancel_job(self, job_name: str):
        try:
            jobspec = os.path.join(self.active_job_dir, job_name)
            if not tf_io.gfile.exists(jobspec):
                raise ValueError(f"Unable to locate jobspec {jobspec}")
            _upload_job_state(
                job_name,
                JobState(status=JobStatus.CANCELLING),
                remote_dir=self.user_states_dir,
            )
            logging.info("Job %s is cancelling.", job_name)
            # Poll for jobspec to be removed.
            while tf_io.gfile.exists(jobspec):
                logging.info("Waiting for job to stop (which usually takes a few minutes)...")
                time.sleep(10)
            logging.info("Job is stopped.")
        except ValueError as e:
            logging.info("Failed with error: %s -- Has the job been cancelled already?", e)

    def submit_job(self, job_name: str, *, job_spec_file: str):
        if not is_valid_job_name(job_name):
            raise ValueError(f"{job_name} is not a valid job name.")
        dst = os.path.join(self.active_job_dir, job_name)
        if tf_io.gfile.exists(dst):
            logging.info("\n\nNote: Job is already running. To restart it, cancel the job first.\n")
        else:
            # Upload the job for bastion to pickup.
            tf_io.gfile.copy(job_spec_file, dst)

    def get_job(self, job_name: str) -> JobSpec:
        job_path = os.path.join(self.active_job_dir, job_name)
        if not tf_io.gfile.exists(job_path):
            raise ValueError(f"Unable to locate jobspec {job_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            job_spec = _download_jobspec(job_name, remote_dir=self.active_job_dir, local_dir=tmpdir)
            return job_spec

    def update_job(self, job_name: str, *, job_spec: JobSpec) -> JobSpec:
        dst = os.path.join(self.active_job_dir, job_name)
        if not tf_io.gfile.exists(dst):
            raise ValueError(f"Unable to locate jobspec {dst}")

        with tempfile.NamedTemporaryFile("w") as f:
            serialize_jobspec(job_spec, f)
            # Upload the job for bastion to pickup.
            tf_io.gfile.copy(f.name, dst, overwrite=True)
        logging.info("Job %s is updating.", job_name)

        return job_spec
