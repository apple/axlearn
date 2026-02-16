"""Utilities for FT manager.

This module provides utility functions and helpers for the fault tolerance (FT) training system.

Environment Variables Used:
    HOSTNAME: Current worker's hostname
    MEGASCALE_SLICE_ID: Current worker's slice/replica ID (0-based)
    TPU_WORKER_ID: Current worker's ID within the replica (0-based)
    TPU_WORKER_HOSTNAMES: Comma-separated list of all worker hostnames in current replica
    NUM_TPU_SLICES: Total number of slices/replicas in the training job

Hostname Format:
    TPU worker hostnames follow the pattern: {job_name}-"job"-{replica_id}-{worker_id}.{job_name}
    Example: mymodel-job-0-1.mymodel-job for replica 0, worker 1
"""

import logging
import os
import pathlib
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, Optional

from google.protobuf import timestamp_pb2

from axlearn.ft import manager_pb2

# Port configuration constants
DEFAULT_MANAGER_PORT = 8901

# Client timeout in seconds
DEFAULT_CLIENT_TIMEOUT = 10.0

# Compiled regex patterns for hostname parsing
_WORKER_ID_PATTERN = re.compile(r"^.+?-job-\d+-(\d+)\..+")
_JOB_NAME_PATTERN = re.compile(r"^(.+?)-job-\d+-\d+\.(\1)(?:\.|$)")


@dataclass
class WorkerIdentity:
    """Worker identity information."""

    hostname: str
    replica_id: int
    worker_id: int

    @property
    def is_replica_manager(self) -> bool:
        """True if this worker is a replica manager (worker_id == 0)."""
        return self.worker_id == 0

    @property
    def is_global_manager(self) -> bool:
        """True if this worker is the global manager (replica_id == 0 and worker_id == 0)."""
        return self.replica_id == 0 and self.worker_id == 0


@dataclass
class WorkerStatusRecord:
    """Record of a worker's status in the registry."""

    worker_identity: WorkerIdentity
    training_step: int
    last_update: float
    timestamp: Any  # protobuf timestamp object
    tensorcore_util: float = -1.0  # Tensor core utilization (0.0-1.0), -1.0 if unavailable


class TrainerProcessController:
    """Controller for managing trainer subprocess lifecycle.

    Provides thread-safe interface for terminating and tracking trainer processes.
    Used to coordinate restart requests between gRPC server and supervisor process.
    """

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.termination_requested = False
        self.termination_reason = ""
        self.lock = threading.Lock()

    def set_process(self, process: subprocess.Popen):
        """Set the current trainer process."""
        with self.lock:
            self.current_process = process
            self.termination_requested = False
            self.termination_reason = ""

    def _cleanup_tpu_lock_files(self) -> None:
        """Clean up TPU lock files.

        See: https://github.com/jax-ml/jax/issues/10192#issuecomment-1509814942
        """
        for lock_file in pathlib.Path("/tmp").glob("libtpu_lockfile*"):
            try:
                lock_file.unlink()
                logging.info("Cleaned up stale lock file: %s", lock_file)
            except FileNotFoundError:
                pass  # File already removed, this is fine
            except OSError as e:
                logging.warning("Failed to remove TPU lock file %s: %s", lock_file, e)

    def _do_terminate(self) -> bool:
        """Execute the actual process termination sequence.

        Uses process group killing to ensure all threads are terminated.
        The trainer subprocess should be started with start_new_session=True.

        Returns:
            bool: True if termination succeeded, False otherwise
        """
        pid = self.current_process.pid

        # Try to get process group ID
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            logging.info("Process %d already dead", pid)
            self._cleanup_tpu_lock_files()
            return True

        # If trainer has its own process group, kill the entire group
        if pgid == pid:
            logging.warning("Sending SIGKILL to process group %d", pgid)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                logging.info("Process group %d already dead", pgid)
            except PermissionError as e:
                logging.warning("Cannot kill process group %d: %s, killing process only", pgid, e)
                self.current_process.kill()
        else:
            logging.warning("Sending SIGKILL to process %d", pid)
            self.current_process.kill()

        try:
            self.current_process.wait(timeout=10)
            logging.info(
                "Process killed successfully, exit code: %s", self.current_process.returncode
            )
        except subprocess.TimeoutExpired:
            logging.error("Process still alive after SIGKILL, pid=%d", pid)
            return False

        self._cleanup_tpu_lock_files()
        return True

    def terminate_training(self, reason: str = "Restart requested"):
        """Terminate the current trainer process.

        Args:
            reason: Reason for termination

        Returns:
            bool: True if termination was initiated, False if no process to terminate
        """
        with self.lock:
            if not self.current_process:
                logging.warning("No trainer process to terminate")
                return False

            logging.warning("Terminating trainer process: %s", reason)
            self.termination_requested = True
            self.termination_reason = reason

            try:
                return self._do_terminate()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Failed to terminate process: %s", e)
                return False

    def check_termination_requested(self):
        """Check if termination was requested and get reason.

        Returns:
            tuple: (was_requested: bool, reason: str)
        """
        with self.lock:
            if self.termination_requested:
                reason = self.termination_reason
                # Reset the flag after checking
                self.termination_requested = False
                self.termination_reason = ""
                return True, reason
            return False, ""

    def clear_process(self):
        """Clear the current process reference."""
        with self.lock:
            self.current_process = None


def retry(max_attempts: int = 3, retry_interval: float = 1.0):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        retry_interval: Initial retry interval in seconds
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    if attempt == max_attempts - 1:  # Last attempt
                        raise e
                    time.sleep(retry_interval)  # Simple delay between retries

        return wrapper

    return decorator


def get_all_worker_hostnames() -> List[str]:
    """Get all worker hostnames from env TPU_WORKER_HOSTNAMES.

    TPU_WORKER_HOSTNAMES contains a comma-separated list of hostnames in the same replica.
    The hostname format is: {job_name}-job-{replica_id}-{worker_id}.{job_name}
    """
    hostnames_str = os.environ.get("TPU_WORKER_HOSTNAMES", "localhost")
    return [h.strip() for h in hostnames_str.split(",") if h.strip()]


def get_replica_id() -> int:
    """Get replica ID from environment variable with validation.

    Returns:
        The replica ID as an integer. Defaults to 0 for single-slice jobs.

    Raises:
        ValueError: If MEGASCALE_SLICE_ID is not a valid integer.
    """
    # Default to "0" for single-slice jobs where MEGASCALE_SLICE_ID is not set
    replica_id_str = os.environ.get("MEGASCALE_SLICE_ID", "0")

    try:
        return int(replica_id_str)
    except ValueError as e:
        raise ValueError(
            f"MEGASCALE_SLICE_ID must be a valid integer, got: '{replica_id_str}'"
        ) from e


def get_worker_id() -> int:
    """Get worker ID from environment variable with validation.

    Returns:
        The worker ID as an integer.

    Raises:
        ValueError: If TPU_WORKER_ID is not set or not a valid integer.
    """
    worker_id_str = os.environ.get("TPU_WORKER_ID")
    if worker_id_str is None:
        raise ValueError("TPU_WORKER_ID environment variable is not set")

    try:
        return int(worker_id_str)
    except ValueError as e:
        raise ValueError(f"TPU_WORKER_ID must be a valid integer, got: '{worker_id_str}'") from e


def get_num_replicas() -> int:
    """Get total number of replicas from environment variable with validation.

    Returns:
        The total number of replicas as an integer. Defaults to 1 for single-slice jobs.

    Raises:
        ValueError: If NUM_TPU_SLICES is not a valid positive integer.
    """
    # Default to "1" for single-slice jobs where NUM_TPU_SLICES is not set
    num_replicas_str = os.environ.get("NUM_TPU_SLICES", "1")

    try:
        num_replicas = int(num_replicas_str)
        if num_replicas < 1:
            raise ValueError(f"NUM_TPU_SLICES must be at least 1, got: {num_replicas}")
        return num_replicas
    except ValueError as e:
        raise ValueError(
            f"NUM_TPU_SLICES must be a valid positive integer, got: '{num_replicas_str}'"
        ) from e


def get_worker_identity() -> WorkerIdentity:
    """Get worker identity from environment variables."""
    hostname = os.environ.get("HOSTNAME", "unknown")
    replica_id = get_replica_id()
    worker_id = get_worker_id()

    identity = WorkerIdentity(
        hostname=hostname,
        replica_id=replica_id,
        worker_id=worker_id,
    )

    # Log the identity for debugging
    logging.debug(
        "Worker identity: hostname=%s, replica_id=%d, worker_id=%d, "
        "is_replica_manager=%s, is_global_manager=%s",
        identity.hostname,
        identity.replica_id,
        identity.worker_id,
        identity.is_replica_manager,
        identity.is_global_manager,
    )

    return identity


def create_worker_identity_proto() -> manager_pb2.WorkerIdentity:
    """Create WorkerIdentity protobuf from environment variables.

    Returns:
        WorkerIdentity protobuf message
    """
    identity_data = get_worker_identity()
    identity = manager_pb2.WorkerIdentity()
    identity.hostname = identity_data.hostname
    identity.replica_id = identity_data.replica_id
    identity.worker_id = identity_data.worker_id
    return identity


def get_job_name() -> str:
    """Extract job name from hostname.

    Returns:
        Job name extracted from hostname, or "unknown" if parsing fails
    """
    hostnames = get_all_worker_hostnames()
    if not hostnames:
        return "unknown"

    hostname = hostnames[0]

    # Match pattern: {job_name}-job-{replica_id}-{worker_id}.{job_name}
    match = _JOB_NAME_PATTERN.match(hostname)

    if match:
        return match.group(1)

    # Fallback: extract domain after first dot (assumes no period in job_name)
    if "." in hostname:
        job_name = hostname.split(".", 1)[1]
        if "." not in job_name:  # Validate no additional domain parts
            return job_name

    return "unknown"


def get_replica_head_hostname(replica_id: int) -> str:
    """Build replica manager hostname for given replica_id.

    Args:
        replica_id: The replica ID to build hostname for

    Returns:
        Hostname of the replica manager for the given replica
    """
    job_name = get_job_name()
    return f"{job_name}-job-{replica_id}-0.{job_name}"


def get_global_manager_hostname() -> str:
    """Get hostname of global manager (replica 0, worker 0).

    Returns:
        Hostname of the global manager
    """
    return get_replica_head_hostname(0)


def get_replica_manager_hostname() -> str:
    """Get hostname of replica manager for current worker's replica.

    Returns:
        Hostname of this worker's replica manager
    """
    replica_id = get_worker_identity().replica_id
    hostname = get_replica_head_hostname(replica_id)

    return hostname


def create_current_timestamp() -> timestamp_pb2.Timestamp:
    """Create a timestamp protobuf with current time.

    Returns:
        Timestamp protobuf set to current time
    """
    now = timestamp_pb2.Timestamp()
    now.GetCurrentTime()
    return now


def worker_identity_to_proto(identity: WorkerIdentity) -> manager_pb2.WorkerIdentity:
    """Convert WorkerIdentity dataclass to protobuf message.

    Args:
        identity: WorkerIdentity dataclass

    Returns:
        WorkerIdentity protobuf message
    """
    proto_identity = manager_pb2.WorkerIdentity()
    proto_identity.hostname = identity.hostname
    proto_identity.replica_id = identity.replica_id
    proto_identity.worker_id = identity.worker_id
    return proto_identity


def extract_worker_id_from_hostname(hostname: str) -> int:
    """Extract worker ID from TPU worker hostname.

    Hostnames follow the pattern: {job_name}-job-{replica_id}-{worker_id}.{job_name}

    Args:
        hostname: Worker hostname to parse

    Returns:
        Worker ID extracted from hostname

    Raises:
        ValueError: If hostname doesn't match expected pattern
    """
    # Match pattern: {job_name}-job-{replica_id}-{worker_id}.{job_name}
    match = _WORKER_ID_PATTERN.match(hostname)

    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot extract worker_id from hostname: {hostname}")


def worker_status_record_to_proto(record: WorkerStatusRecord) -> manager_pb2.WorkerStatusEntry:
    """Convert WorkerStatusRecord dataclass to WorkerStatusEntry protobuf.

    Args:
        record: WorkerStatusRecord dataclass

    Returns:
        WorkerStatusEntry protobuf message
    """
    proto_entry = manager_pb2.WorkerStatusEntry()

    # Convert worker identity
    proto_entry.worker_identity.CopyFrom(worker_identity_to_proto(record.worker_identity))

    # Set worker status
    proto_entry.worker_status.training_step = record.training_step
    proto_entry.worker_status.tensorcore_util = record.tensorcore_util

    # Set timestamp from float
    proto_entry.last_update.FromSeconds(int(record.last_update))
    proto_entry.last_update.nanos = int((record.last_update % 1) * 1e9)

    return proto_entry
