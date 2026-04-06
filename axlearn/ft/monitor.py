"""FT trainer status monitor.

Monitors training status and reports to FT manager system for fault tolerance.
Provides hang detection and automatic restart capabilities.
"""

import logging
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from axlearn.ft.manager import Manager
from axlearn.ft.utils import (
    REPORT_INTERVAL_SECONDS,
    STRAGGLER_CONFIRMED,
    STRAGGLER_POTENTIAL,
    TENSORCORE_UTIL_UNAVAILABLE,
)

# libtpu module is only useable on TPU machines
try:
    from libtpu import sdk  # type: ignore[import]
except ImportError:
    sdk = None


@dataclass
class _WorkerData:
    """Worker info and snapshot of metrics."""

    replica_id: int
    worker_id: int
    hostname: str
    tensorcore_util: float


class StragglerMonitor:
    """Monitors straggler workers slowing down training efficiency

    1. Tensorcore straggler detection:

    Uses Median Absolute Deviation (MAD) to identify workers whose tensorcore
    utilization deviates significantly from the rest of workers. More robust to outliers
    than standard deviation. Only triggers after a worker is flagged as a straggler for a
    sustained duration.

    TODO: Add additional straggler detection methods

    Args:
        enabled: Enable straggler detection.
        sensitivity: Number of MAD-scaled deviations from the median to
            consider a straggler. Lower = more sensitive. Default 3.0.
        sustained_duration_seconds: How long a worker must be continuously
            straggling before a warning is logged. Default 300 (5 minutes).
    """

    # Consistency constant that makes the MAD-based score comparable to
    # standard deviation units for normally distributed data.
    _CONSISTENCY_CONSTANT = 0.6745

    def __init__(
        self,
        enabled: bool = False,
        sensitivity: float = 8.0,
        sustained_duration_seconds: int = 300,
    ):
        self.enabled = enabled
        self.sensitivity = sensitivity
        self.sustained_duration_seconds = sustained_duration_seconds
        # Maps (replica_id, worker_id) -> timestamp when first flagged as straggler.
        self._straggler_since: dict[tuple[int, int], float] = {}

    def reset(self):
        """Clear all straggler tracking state."""
        self._straggler_since = {}

    def _log_tensorcore_straggler_worker(
        self,
        *,
        confirmed: bool,
        replica_id: int,
        worker_id: int,
        hostname: str,
        tensorcore_util: float,
        median_tensorcore_util: float,
        job_avg: float,
        duration: float,
        modified_z: float,
    ) -> None:
        """Log a warning for a straggler worker with anomalous
        tensorcore utilization .

        Args:
            confirmed: True if the worker has been a straggler for the full
                sustained duration, False for the early (50%) warning.
        """
        if confirmed:
            label = STRAGGLER_CONFIRMED
            detail = (
                f"threshold: {self.sustained_duration_seconds}s, "
                f"modified z-score: {modified_z:.1f}, "
                f"sensitivity: {self.sensitivity:.1f}"
            )
        else:
            label = STRAGGLER_POTENTIAL
            detail = f"modified z-score: {modified_z:.1f}, " f"sensitivity: {self.sensitivity:.1f}"

        logging.warning(
            "FT Monitor: %s - replica %d worker %d (%s)\n"
            "    tensorcore util %.1f%% vs job median %.1f%% (job avg %.1f%%)\n"
            "    straggler for %d seconds (%s)",
            label,
            replica_id,
            worker_id,
            hostname,
            tensorcore_util,
            median_tensorcore_util,
            job_avg,
            int(duration),
            detail,
        )

    def run_tensorcore_straggler_check(self, detailed_status: dict) -> None:
        """Check for straggler workers in the job based on tensorcore utilization

        Args:
            detailed_status: The detailed global status dict from the FT manager,
                containing per-replica, per-worker tensorcore utilization data.
        """
        if not self.enabled:
            return

        # 1. Collect valid worker tensorcore util readings.
        workers_data: list[_WorkerData] = []
        for replica_id, replica_info in detailed_status.get("replicas", {}).items():
            for worker in replica_info.get("workers", []):
                tensorcore_util = worker.get("tensorcore_util", TENSORCORE_UTIL_UNAVAILABLE)
                if tensorcore_util < 0:
                    continue
                identity = worker.get("worker_identity", {})
                workers_data.append(
                    _WorkerData(
                        replica_id=(
                            int(replica_id) if not isinstance(replica_id, int) else replica_id
                        ),
                        worker_id=identity.get("worker_id", -1),
                        hostname=identity.get("hostname", "unknown"),
                        tensorcore_util=tensorcore_util,
                    )
                )

        # 2. Need at least 4 valid workers for meaningful straggler detection.
        if len(workers_data) < 4:
            return

        tensorcore_values = [worker_data.tensorcore_util for worker_data in workers_data]

        # 3. Compute median.
        median_tensorcore_util = statistics.median(tensorcore_values)

        # 4. Compute MAD = median(|xi - median|).
        abs_deviations = [abs(v - median_tensorcore_util) for v in tensorcore_values]
        mad = statistics.median(abs_deviations)

        # 5. If MAD == 0 all workers are identical — no outliers possible.
        if mad == 0:
            self._straggler_since.clear()
            return

        # 6. Compute job average for logging context.
        job_avg = statistics.mean(tensorcore_values)

        now = time.time()
        current_stragglers: set[tuple[int, int]] = set()

        for worker_data in workers_data:
            deviation = worker_data.tensorcore_util - median_tensorcore_util
            modified_z = self._CONSISTENCY_CONSTANT * deviation / mad
            if abs(modified_z) < self.sensitivity:
                continue

            key = (worker_data.replica_id, worker_data.worker_id)
            current_stragglers.add(key)

            if key not in self._straggler_since:
                self._straggler_since[key] = now

            duration = now - self._straggler_since[key]
            half_threshold = self.sustained_duration_seconds / 2

            if duration >= self.sustained_duration_seconds:
                self._log_tensorcore_straggler_worker(
                    confirmed=True,
                    replica_id=worker_data.replica_id,
                    worker_id=worker_data.worker_id,
                    hostname=worker_data.hostname,
                    tensorcore_util=worker_data.tensorcore_util,
                    median_tensorcore_util=median_tensorcore_util,
                    job_avg=job_avg,
                    duration=duration,
                    modified_z=modified_z,
                )
            elif duration >= half_threshold:
                self._log_tensorcore_straggler_worker(
                    confirmed=False,
                    replica_id=worker_data.replica_id,
                    worker_id=worker_data.worker_id,
                    hostname=worker_data.hostname,
                    tensorcore_util=worker_data.tensorcore_util,
                    median_tensorcore_util=median_tensorcore_util,
                    job_avg=job_avg,
                    duration=duration,
                    modified_z=modified_z,
                )

        # 7. Remove workers that are no longer stragglers.
        for key in list(self._straggler_since):
            if key not in current_stragglers:
                del self._straggler_since[key]


class StatusMonitor:
    """Monitor training status and report to FT manager.

    Provides automatic status reporting, hang detection, and restart capabilities
    for distributed training fault tolerance.
    """

    def __init__(
        self,
        trainer=None,
        process_controller=None,
        hang_threshold_in_seconds: int = 600,
        # TODO(ruhan-prasad): Reduce interval once Google fixes race condition with libtpu sdk
        metrics_sampling_interval_seconds: float = 30.0,
        straggler_monitor: Optional[StragglerMonitor] = None,
    ):
        """Initialize status monitor with FT manager.

        Args:
            trainer: Optional trainer instance to get step directly from trainer.step
            process_controller: Optional process controller for restart functionality
            hang_threshold_in_seconds: Hang detection threshold in seconds (default: 10 minutes)
            metrics_sampling_interval_seconds: How often to sample tensorcore util (default: 3s)
            straggler_monitor: Optional straggler monitor
        """
        self.manager = Manager(process_controller=process_controller)
        self.trainer = trainer
        self.current_step = -1
        self.last_step_time = time.time()
        self.stop_event = threading.Event()
        self.step_pattern = re.compile(r"step\s+(\d+)")

        # Hang detection settings
        self.hang_detection_enabled = True
        self.hang_threshold_seconds = hang_threshold_in_seconds
        self.last_hang_check = time.time()

        # Tensor core utilization tracking
        self.current_tensorcore_util = TENSORCORE_UTIL_UNAVAILABLE

        # High-frequency tensorcore sampling.
        # Buffer is drained every REPORT_INTERVAL_SECONDS in monitor_loop.
        self._metrics_sampling_interval = metrics_sampling_interval_seconds
        self._tensorcore_util_samples: list[float] = []

        # Straggler detection
        self._straggler_monitor = straggler_monitor or StragglerMonitor()

    @property
    def is_global_manager(self) -> bool:
        """Check if this is a global manager."""
        return (
            hasattr(self.manager, "_identity")
            and self.manager._identity.is_global_manager  # pylint: disable=protected-access
        )

    @property
    def is_replica_manager(self) -> bool:
        """Check if this is a replica manager."""
        return (
            hasattr(self.manager, "_identity")
            and self.manager._identity.is_replica_manager  # pylint: disable=protected-access
        )

    def parse_step_from_log(self, log_line: str) -> Optional[int]:
        """Extract step number from trainer log line. This is a fallback method."""
        match = self.step_pattern.search(log_line)
        if match:
            return int(match.group(1))
        return None

    def update_step(self, step: int):
        """Update current step from any source (logs, callbacks, etc)."""
        if step > self.current_step:
            self.current_step = step
            self.last_step_time = time.time()

    def get_tensorcore_utilization(self) -> float:
        """Get current tensor core utilization.

        Returns:
            Utilization as float 0.0-1.0, or TENSORCORE_UTIL_UNAVAILABLE if unavailable
        """
        if sdk is None:
            return TENSORCORE_UTIL_UNAVAILABLE
        try:
            metric = sdk.tpumonitoring.get_metric("tensorcore_util")
            float_data = [float(x) for x in metric.data()]
            if not float_data:
                return TENSORCORE_UTIL_UNAVAILABLE
            worker_average = sum(float_data) / len(float_data)
            return worker_average

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug("Failed to get tensor core utilization: %s", e)
            return TENSORCORE_UTIL_UNAVAILABLE

    def _get_buffered_tensorcore_average(self) -> float:
        """Drain the sample buffer and return the average valid tensorcore utilization.

        Filters out unavailable readings. Returns TENSORCORE_UTIL_UNAVAILABLE if no valid samples.
        """
        samples = self._tensorcore_util_samples
        self._tensorcore_util_samples = []
        valid = [s for s in samples if s >= 0]
        if not valid:
            return TENSORCORE_UTIL_UNAVAILABLE
        return sum(valid) / len(valid)

    def detect_global_training_hang(self) -> bool:
        """Detect if training is hanging across the training job (global manager only)."""
        if not self.is_global_manager:
            return False

        try:
            detailed_status = self.manager.get_detailed_global_status()
            replicas = detailed_status.get("replicas", {})

            if not replicas:
                logging.warning("No replicas reporting to global manager")
                return False

            hanging_replicas_count = self._count_hanging_replicas(replicas)

            if hanging_replicas_count > 0:
                logging.error(
                    "Training hang detected: %d/%d replicas hanging",
                    hanging_replicas_count,
                    len(replicas),
                )
                return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error detecting training hang: %s", e)

        return False

    def _count_hanging_replicas(self, replicas: dict) -> int:
        """Count number of replicas which are hanging."""
        hanging_count = 0
        current_time = time.time()

        for replica_id, replica_info in replicas.items():
            workers = replica_info.get("workers", [])
            if not workers:
                continue

            # Check if any worker in this replica is making progress
            replica_hanging = all(
                current_time - worker.get("last_update", 0) >= self.hang_threshold_seconds
                for worker in workers
            )

            if replica_hanging:
                hanging_count += 1
                logging.warning(
                    "Replica %d are hanging, no worker progress in %d seconds",
                    replica_id,
                    self.hang_threshold_seconds,
                )

        return hanging_count

    def trigger_global_restart(self, reason: str) -> bool:
        """Trigger restart of all replicas (global manager only)."""
        if not self.is_global_manager:
            logging.error("Only global manager can trigger global restart")
            return False

        try:
            logging.warning("Triggering global restart: %s", reason)
            results = self.manager.send_restart_to_all_replicas(reason)

            successful_restarts = sum(1 for success in results.values() if success)

            if successful_restarts > 0:
                logging.info(
                    "Global restart initiated: %d/%d replicas acknowledged",
                    successful_restarts,
                    len(results),
                )
                return True
            else:
                logging.error("Global restart failed: no replicas acknowledged")
                return False

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error triggering global restart: %s", e)
            return False

    def monitor_loop(self):
        """Main monitoring loop.

        Wakes every ``_metrics_sampling_interval`` seconds to sample tensorcore
        utilization, and performs the full reporting cycle every
        ``REPORT_INTERVAL_SECONDS``.
        """
        last_report_time = time.monotonic()
        while not self.stop_event.wait(self._metrics_sampling_interval):
            sample = self.get_tensorcore_utilization()
            self._tensorcore_util_samples.append(sample)
            if time.monotonic() - last_report_time >= REPORT_INTERVAL_SECONDS:
                last_report_time = time.monotonic()
                try:
                    self._update_current_step()
                    self._report_worker_status()

                    if self.is_replica_manager:
                        self._report_replica_status()
                    if self.is_global_manager:
                        self._handle_global_manager_tasks()

                except Exception as e:  # pylint: disable=broad-exception-caught
                    logging.error("FT Monitor: Failed to report status: %s", e)

    def _update_current_step(self):
        """Update current step from trainer if available."""
        if self.trainer is not None:
            self.current_step = self.trainer.step

    def _report_worker_status(self):
        """Report worker status to replica manager."""
        # Use buffered average if available, fall back to single read.
        buffered = self._get_buffered_tensorcore_average()
        self.current_tensorcore_util = (
            buffered if buffered >= 0 else self.get_tensorcore_utilization()
        )

        # Report status with utilization
        self.manager.report_status(
            step=self.current_step, tensorcore_util=self.current_tensorcore_util
        )

        if self.current_tensorcore_util >= 0:
            logging.debug(
                "FT Monitor: Reported step %d, TensorCore util %.1f%%",
                self.current_step,
                self.current_tensorcore_util,
            )
        else:
            logging.debug("FT Monitor: Reported step %d", self.current_step)

    def _report_replica_status(self):
        """Report replica status to global manager."""
        try:
            self.manager.report_replica_status()
            logging.debug("FT Monitor: Reported replica status")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("FT Monitor: Failed to report replica status: %s", e)

    def _handle_global_manager_tasks(self):
        """Handle global manager specific tasks."""
        try:
            detailed_status = self._log_global_status()

            # Check for hang detection
            if self.hang_detection_enabled:
                self._check_and_handle_hangs()

            # Check for straggler workers
            if self._straggler_monitor.enabled:
                self._straggler_monitor.run_tensorcore_straggler_check(detailed_status)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("FT Monitor: Failed to handle global manager tasks: %s", e)

    def _log_replica_details(self, replica_id, replica_info):
        workers = replica_info.get("workers", [])
        if workers:
            worker_details = []
            tensorcore_utils = []

            for worker in workers:
                hostname = worker.get("worker_identity", {}).get("hostname", "unknown")
                worker_id = worker.get("worker_identity", {}).get("worker_id", -1)
                step = worker.get("training_step", 0)
                tensorcore_util = worker.get("tensorcore_util", TENSORCORE_UTIL_UNAVAILABLE)

                # Build worker detail string
                detail = f"{hostname}:w{worker_id}:s{step}"
                if tensorcore_util >= 0:
                    detail += f":tensorcore{tensorcore_util:.0f}%"
                    tensorcore_utils.append(tensorcore_util)
                worker_details.append(detail)

            # Calculate average tensorcore utilization for replica
            if tensorcore_utils:
                avg_tensorcore_util = sum(tensorcore_utils) / len(tensorcore_utils)
                logging.debug(
                    "FT Monitor: Replica %d workers - %s (avg tensorcore util: %.1f%%)",
                    replica_id,
                    ", ".join(worker_details),
                    avg_tensorcore_util,
                )
            else:
                logging.debug(
                    "FT Monitor: Replica %d workers - %s", replica_id, ", ".join(worker_details)
                )
        else:
            logging.debug("FT Monitor: Replica %d - no workers reported", replica_id)

    def _log_global_status(self) -> dict:
        """Log global status information.

        Returns:
            The detailed global status dict, for reuse by callers.
        """
        status = self.manager.get_global_status()
        detailed_status = self.manager.get_detailed_global_status()

        logging.debug(
            "FT Monitor: Global status - total replicas: %d, total workers: %d/%d, step: %d",
            status.get("total_replicas", 0),
            status.get("total_reported_workers", 0),
            status.get("total_workers", 0),
            self.current_step,
        )

        # Log detailed worker steps and tensor core utilization for each replica
        for replica_id, replica_info in detailed_status.get("replicas", {}).items():
            self._log_replica_details(replica_id, replica_info)

        return detailed_status

    def _check_and_handle_hangs(self):
        """Check for training hangs and trigger restart if needed."""
        current_time = time.time()
        if current_time - self.last_hang_check >= 60:  # Check every minute
            self.last_hang_check = current_time

            if self.detect_global_training_hang():
                reason = f"Training hang detected after {self.hang_threshold_seconds}s"
                restart_success = self.trigger_global_restart(reason)
                if restart_success:
                    logging.warning("Global restart triggered due to training hang")
                else:
                    logging.error("Failed to trigger global restart")

    def start(self):
        """Start monitoring thread."""
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.debug("FT Monitor: Started monitoring thread")

    def stop(self):
        """Stop monitoring thread."""
        self.stop_event.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=5)
        self.manager.cleanup()
        logging.debug("FT Monitor: Stopped monitoring thread")

    # Convenience methods for easier integration
    def update_step_from_callback(self, step: int):
        """Alias for update_step() for backward compatibility."""
        self.update_step(step)

    def get_restart_status(self) -> tuple[bool, str]:
        """Check if restart was requested through FT system.

        Returns:
            tuple: (restart_requested: bool, reason: str)
        """
        return self.manager.get_restart_status()

    def _forward_trainer_log_line(self, line: str) -> None:
        """Forward a trainer subprocess log line to stdout for Cloud Logging.

        Args:
            line: A single line of text from the trainer's stdout (may include
                trailing newline).
        """
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
        sys.stdout.flush()

    def monitor_training_process(self, process: "subprocess.Popen") -> int:
        """Monitor training process with integrated restart handling.

        Args:
            process: The training subprocess to monitor.

        Returns:
            int: Process return code
        """
        try:
            # Stream output in real-time and parse steps
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break

                    # Parse step from log line and update monitor
                    step = self.parse_step_from_log(line)
                    if step is not None:
                        self.update_step(step)

                    # Forward trainer log line preserving original metadata
                    self._forward_trainer_log_line(line)

                    # Check if restart was requested through FT system
                    termination_requested, reason = self.get_restart_status()
                    if termination_requested:
                        logging.warning("FT Monitor: Termination requested: %s", reason)
                        break

            # Wait for completion
            return process.wait()

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("FT Monitor: Error monitoring process: %s", e)
            return 1
