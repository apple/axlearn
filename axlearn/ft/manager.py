"""FT trainer manager.

Fault tolerance trainer manager for distributed training, supports functions including:
- Regular workers: Report status to replica managers
- Replica managers: Manage workers and report to global manager
- Global manager: Coordinate all replicas
"""

import atexit
import logging
from typing import Dict, Optional

from axlearn.ft import manager_pb2
from axlearn.ft.manager_client import ManagerClient
from axlearn.ft.manager_server import ManagerServer
from axlearn.ft.utils import (
    DEFAULT_CLIENT_TIMEOUT,
    DEFAULT_MANAGER_PORT,
    get_global_manager_hostname,
    get_num_replicas,
    get_replica_head_hostname,
    get_replica_manager_hostname,
    get_worker_identity,
)


class Manager:
    """FT manager that adapts based on worker role."""

    def __init__(
        self,
        port: Optional[int] = None,
        timeout: int = 600,
        client_timeout: float = DEFAULT_CLIENT_TIMEOUT,
        process_controller=None,
    ):
        """Initialize and start manager components.

        Args:
            port: gRPC port (auto-detected if None)
            timeout: Timeout for waiting operations
            client_timeout: gRPC client timeout
            process_controller: Process controller for managing trainer subprocess
        """
        self._identity = get_worker_identity()
        self.port = port or DEFAULT_MANAGER_PORT
        self.timeout = timeout
        self.client_timeout = client_timeout
        self._server = None
        self._client = None

        atexit.register(self.cleanup)

        # Initialize components based on role
        logging.info(
            "Initializing manager in worker (replica=%d, worker=%d, port=%d)",
            self._identity.replica_id,
            self._identity.worker_id,
            self.port,
        )

        try:
            self._client = ManagerClient(timeout=client_timeout, port=DEFAULT_MANAGER_PORT)

            self._server = ManagerServer(port=self.port, process_controller=process_controller)
            self._server.start_server()

            logging.info("Manager initialized.")
        except Exception as e:
            logging.error("Failed to initialize manager: %s", e)
            self.cleanup()
            raise

    def report_status(self, step: int = 0, tensorcore_util: float = -1.0):
        """Report training step and tensor core utilization to replica manager.

        Args:
            step: Current training step
            tensorcore_util: Tensor core utilization (0.0-1.0), -1.0 if unavailable
        """

        status = manager_pb2.WorkerStatus(training_step=step, tensorcore_util=tensorcore_util)
        replica_manager_host = get_replica_manager_hostname()

        success = self._client.report_status(replica_manager_host, status)
        if not success:
            raise RuntimeError(f"Failed to report status to {replica_manager_host}")

    def report_replica_status(self):
        """Report replica status to global manager (replica managers only)."""
        self._check_role("replica_manager", "report replica status")

        status = self._server.get_replica_status()
        global_manager_host = get_global_manager_hostname()

        success = self._client.report_replica_status(
            global_manager_host, self._identity.replica_id, status
        )
        if not success:
            raise RuntimeError(f"Failed to report replica status to {global_manager_host}")

    def get_replica_status(self):
        """Get replica status (replica managers only)."""
        self._check_role("replica_manager", "get replica status")
        if not self._server:
            raise RuntimeError("Server not available for this role")
        return self._server.get_replica_status()

    def get_global_status(self):
        """Get global status (global manager only)."""
        self._check_role("global_manager", "get global status")
        if not self._server:
            raise RuntimeError("Server not available for this role")
        return self._server.get_global_status()

    def get_detailed_global_status(self):
        """Get detailed global status including individual worker information.

        Only available for global manager.
        """
        self._check_role("global_manager", "get detailed global status")
        if not self._server:
            raise RuntimeError("Server not available for this role")
        return self._server.get_detailed_global_status()

    def _send_restart_to_replica(self, replica_id: int, reason: str) -> bool:
        """Send restart request to a specific replica manager (global manager only).

        Args:
            replica_id: Target replica ID
            reason: Reason for restart

        Returns:
            bool: True if restart was acknowledged, False otherwise
        """
        self._check_role("global_manager", "send restart requests")
        if not self._client:
            raise RuntimeError("Client not available for this role")

        target_hostname = get_replica_head_hostname(replica_id)
        return self._client.restart_replica(
            target_hostname, replica_id, reason, DEFAULT_MANAGER_PORT
        )

    def send_restart_to_all_replicas(self, reason: str) -> Dict[int, bool]:
        """Send restart requests to all known replicas (global manager only).

        Args:
            reason: Reason for restart

        Returns:
            dict: Map of replica_id -> success status
        """
        self._check_role("global_manager", "send restart requests")
        if not self._server:
            raise RuntimeError("Server not available for global manager")

        try:
            num_replicas = get_num_replicas()
        except ValueError as e:
            logging.error("Failed to get number of replicas: %s", e)
            return {}

        results = {}

        for replica_id in range(num_replicas):
            logging.info("Sending restart request to replica %d: %s", replica_id, reason)
            success = self._send_restart_to_replica(replica_id, reason)
            results[replica_id] = success

            if success:
                logging.info("Replica %d restart request acknowledged", replica_id)
            else:
                logging.error("Replica %d restart request failed", replica_id)

        return results

    def report_pod_shutdown(self, reason: str = "Pod termination") -> bool:
        """Report pod shutdown to global manager.

        Args:
            reason: Reason for shutdown

        Returns:
            bool: True if shutdown was reported successfully
        """
        if not self._client:
            raise RuntimeError("Client not available for pod shutdown reporting")

        return self._client.report_pod_shutdown(reason)

    def restart_local_training(self, reason: str = "Manual restart") -> bool:
        """Restart local training process if available.

        Args:
            reason: Reason for restart

        Returns:
            bool: True if restart was initiated successfully
        """
        if (
            not self._server
            or not hasattr(self._server, "process_controller")
            or not self._server.process_controller
        ):
            logging.warning("No process controller available for local restart")
            return False

        return self._server.process_controller.terminate_training(reason)

    def get_restart_status(self) -> tuple[bool, str]:
        """Check if restart was requested through FT system.

        Returns:
            tuple: (restart_requested: bool, reason: str)
        """
        if (
            self._server
            and hasattr(self._server, "process_controller")
            and self._server.process_controller
        ):
            return self._server.process_controller.check_termination_requested()
        return False, ""

    def _check_role(self, required_role: str, action: str):
        """Check if current role can perform action."""
        if required_role == "replica_manager" and not self._identity.is_replica_manager:
            raise RuntimeError(f"Only replica managers can {action}")
        if required_role == "global_manager" and not self._identity.is_global_manager:
            raise RuntimeError(f"Only global manager can {action}")

    def cleanup(self):
        """Clean up resources."""
        if self._client:
            self._client.close()
        if self._server:
            self._server.stop_server()
        logging.info("Manager cleanup completed")
