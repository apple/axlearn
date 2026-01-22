"""FT trainer status monitor.

Monitors training status and reports to FT manager system for fault tolerance.
Provides hang detection and automatic restart capabilities.
"""

import logging
import re
import subprocess
import threading
import time
from typing import Optional

from axlearn.ft.manager import Manager

# libtpu module is only useable on TPU machines
try:
    from libtpu import sdk  # type: ignore[import]
except ImportError:
    sdk = None


class StatusMonitor:
    """Monitor training status and report to FT manager.

    Provides automatic status reporting, hang detection, and restart capabilities
    for distributed training fault tolerance.
    """

    def __init__(self, trainer=None, process_controller=None, hang_threshold_in_seconds: int = 600):
        """Initialize status monitor with FT manager.

        Args:
            trainer: Optional trainer instance to get step directly from trainer.step
            process_controller: Optional process controller for restart functionality
            hang_threshold_in_seconds: Hang detection threshold in seconds (default: 10 minutes)
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
        self.current_tensorcore_util = -1.0

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
            Utilization as float 0.0-1.0, or -1.0 if unavailable
        """
        if sdk is None:
            return -1.0
        try:
            metric = sdk.monitoring.get_metric("tensorcore_util")
            float_data = [float(x) for x in metric.data()]
            if not float_data:
                return -1.0
            worker_average = sum(float_data) / len(float_data)
            return worker_average

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to get tensor core utilization: %s", e)
            return -1.0

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
        """Main monitoring loop - runs every 30 seconds."""
        while not self.stop_event.wait(30):  # Wait 30 seconds or until stop
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
        # Get tensor core utilization
        self.current_tensorcore_util = self.get_tensorcore_utilization()

        # Report status with utilization
        self.manager.report_status(
            step=self.current_step, tensorcore_util=self.current_tensorcore_util
        )

        if self.current_tensorcore_util >= 0:
            logging.info(
                "FT Monitor: Reported step %d, TC util %.1f%%",
                self.current_step,
                self.current_tensorcore_util,
            )
        else:
            logging.info("FT Monitor: Reported step %d", self.current_step)

    def _report_replica_status(self):
        """Report replica status to global manager."""
        try:
            self.manager.report_replica_status()
            logging.info("FT Monitor: Reported replica status")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("FT Monitor: Failed to report replica status: %s", e)

    def _handle_global_manager_tasks(self):
        """Handle global manager specific tasks."""
        try:
            self._log_global_status()

            # Check for hang detection
            if self.hang_detection_enabled:
                self._check_and_handle_hangs()

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("FT Monitor: Failed to handle global manager tasks: %s", e)

    def _log_replica_details(self, replica_id, replica_info):
        workers = replica_info.get("workers", [])
        if workers:
            worker_details = []
            tc_utils = []

            for worker in workers:
                hostname = worker.get("worker_identity", {}).get("hostname", "unknown")
                worker_id = worker.get("worker_identity", {}).get("worker_id", -1)
                step = worker.get("training_step", 0)
                tc_util = worker.get("tensorcore_util", -1.0)

                # Build worker detail string
                detail = f"{hostname}:w{worker_id}:s{step}"
                if tc_util >= 0:
                    detail += f":tc{tc_util:.0f}%"
                    tc_utils.append(tc_util)
                worker_details.append(detail)

            # Calculate average TC utilization for replica
            if tc_utils:
                avg_tc_util = sum(tc_utils) / len(tc_utils)
                logging.info(
                    "FT Monitor: Replica %d workers - %s (avg TC util: %.1f%%)",
                    replica_id,
                    ", ".join(worker_details),
                    avg_tc_util,
                )
            else:
                logging.info(
                    "FT Monitor: Replica %d workers - %s", replica_id, ", ".join(worker_details)
                )
        else:
            logging.info("FT Monitor: Replica %d - no workers reported", replica_id)

    def _log_global_status(self):
        """Log global status information."""
        status = self.manager.get_global_status()
        detailed_status = self.manager.get_detailed_global_status()

        logging.info(
            "FT Monitor: Global status - total replicas: %d, total workers: %d/%d, step: %d",
            status.get("total_replicas", 0),
            status.get("total_reported_workers", 0),
            status.get("total_workers", 0),
            self.current_step,
        )

        # Log detailed worker steps and tensor core utilization for each replica
        for replica_id, replica_info in detailed_status.get("replicas", {}).items():
            self._log_replica_details(replica_id, replica_info)

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
        logging.info("FT Monitor: Started monitoring thread")

    def stop(self):
        """Stop monitoring thread."""
        self.stop_event.set()
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=5)
        self.manager.cleanup()
        logging.info("FT Monitor: Stopped monitoring thread")

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

    def monitor_training_process(self, process: "subprocess.Popen") -> int:
        """Monitor training process with integrated restart handling.

        Args:
            process: The training subprocess to monitor

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

                    # Output with FT prefix
                    print(f"[FT_TRAINER] {line}", end="", flush=True)

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
