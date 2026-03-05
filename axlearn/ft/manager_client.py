"""FT manager gRPC client.

Provides client functionality for FT manager operations including
worker status reporting, replica status reporting, and restart requests.
"""

import logging
import threading
from typing import Any, Optional

import grpc

from axlearn.ft import manager_pb2, manager_pb2_grpc
from axlearn.ft.utils import (
    DEFAULT_MANAGER_PORT,
    create_current_timestamp,
    create_worker_identity_proto,
    get_global_manager_hostname,
    retry,
)


class ManagerClient:
    """FT trainer manager gRPC client with simplified, consistent interface."""

    def __init__(self, timeout: float = 10.0, port: int = DEFAULT_MANAGER_PORT):
        """Initialize manager client.

        Args:
            timeout: gRPC request timeout in seconds
            port: Default port for manager connections
        """
        self.timeout = timeout
        self.port = port
        self._channels: dict[tuple[str, int], grpc.Channel] = {}
        self._channels_lock = threading.Lock()
        logging.debug("ManagerClient initialized: timeout=%.1fs, port=%d", timeout, port)

    def _get_stub(self, hostname: str, port: int) -> manager_pb2_grpc.ManagerServiceStub:
        """Return a cached gRPC stub for the target host, creating the channel if needed."""
        key = (hostname, port)
        with self._channels_lock:
            if key not in self._channels:
                self._channels[key] = grpc.insecure_channel(f"{hostname}:{port}")
            return manager_pb2_grpc.ManagerServiceStub(self._channels[key])

    def _grpc_call(
        self,
        hostname: str,
        port: int,
        method_name: str,
        request,
        log_prefix: Optional[str] = None,
    ) -> Any:
        """Execute a gRPC call with uniform debug logging.

        Args:
            hostname: Target hostname.
            port: Target port.
            method_name: Name of the gRPC stub method to invoke (e.g. "ReportStatus",
                "RestartReplicaTraining"). Must match the RPC name in the proto service definition.
            request: Protobuf request message.
            log_prefix: Short string used in log messages; defaults to method_name.

        Returns:
            The response message. Raises exceptions on failure.
        """
        prefix = log_prefix or method_name
        logging.debug("%s: sending to %s:%d", prefix, hostname, port)
        stub = self._get_stub(hostname, port)
        response = getattr(stub, method_name)(request, timeout=self.timeout)
        logging.debug(
            "%s: response from %s:%d acknowledged=%s",
            prefix,
            hostname,
            port,
            response.acknowledged,
        )
        return response

    @retry(max_attempts=3)
    def report_status(
        self, target_host: str, worker_status: manager_pb2.WorkerStatus, port: Optional[int] = None
    ) -> bool:
        """Send worker status update to replica manager.

        Args:
            target_host: Hostname of the replica manager
            worker_status: Worker status protobuf containing training step
            port: Port for the replica manager service (uses default_port if None)

        Returns:
            bool: True if status was acknowledged, False otherwise
        """
        if port is None:
            port = self.port
        try:
            request = manager_pb2.StatusUpdate()
            request.worker_identity.CopyFrom(create_worker_identity_proto())
            request.worker_status.CopyFrom(worker_status)
            request.timestamp.CopyFrom(create_current_timestamp())

            response = self._grpc_call(target_host, port, "ReportStatus", request)
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug(
                "Status report failed: target=%s:%d, step=%d, error=%s",
                target_host,
                port,
                worker_status.training_step,
                str(e),
            )
            return False

    @retry(max_attempts=3)
    def report_replica_status(
        self,
        target_host: str,
        replica_id: int,
        replica_status: manager_pb2.ReplicaStatus,
        port: Optional[int] = None,
    ) -> bool:
        """Send replica status update to global manager.

        Args:
            target_host: Hostname of the global manager
            replica_id: ID of the replica reporting status
            replica_status: Replica status protobuf
            port: Port for the global manager service (uses default_port if None)

        Returns:
            bool: True if status was acknowledged, False otherwise
        """
        if port is None:
            port = self.port
        try:
            # Create request
            request = manager_pb2.ReplicaStatusUpdate()
            request.replica_id = replica_id
            request.replica_status.CopyFrom(replica_status)
            request.timestamp.CopyFrom(create_current_timestamp())

            response = self._grpc_call(target_host, port, "ReportReplicaStatus", request)
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
                "Replica status failed: replica_id=%d, target=%s:%d, error=%s",
                replica_id,
                target_host,
                port,
                str(e),
            )
            return False

    @retry(max_attempts=3)
    def restart_replica(
        self, target_hostname: str, replica_id: int, reason: str, port: Optional[int] = None
    ) -> bool:
        """Send restart request to a replica manager (Global -> Replica level).

        Args:
            target_hostname: Hostname of the replica manager
            replica_id: Target replica ID
            reason: Reason for restart
            port: Port of the replica manager (uses default_port if None)

        Returns:
            bool: True if restart was acknowledged, False otherwise
        """
        if port is None:
            port = self.port
        try:
            request = manager_pb2.RestartReplicaRequest()
            request.replica_id = replica_id
            request.reason = reason
            request.timestamp.CopyFrom(create_current_timestamp())

            response = self._grpc_call(
                target_hostname,
                port,
                "RestartReplicaTraining",
                request,
                f"RestartReplicaTraining(replica={replica_id}, reason='{reason}')",
            )
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug(
                "RestartReplicaTraining failed: target=%s:%d, replica_id=%d, "
                "reason='%s', error=%s",
                target_hostname,
                port,
                replica_id,
                reason,
                str(e),
            )
            return False

    @retry(max_attempts=3)
    def restart_worker(
        self,
        target_hostname: str,
        replica_id: int,
        reason: str,
        worker_id: int,
        port: Optional[int] = None,
    ) -> bool:
        """Send restart request to a specific worker (Replica -> Worker level).

        Args:
            target_hostname: Hostname of the worker
            replica_id: Replica ID of the target worker
            reason: Reason for restart
            worker_id: Worker ID of the target worker
            port: Port of the worker (uses default_port if None)

        Returns:
            bool: True if restart was acknowledged, False otherwise
        """
        if port is None:
            port = self.port
        try:
            request = manager_pb2.RestartRequest()
            request.worker_identity.hostname = target_hostname
            request.worker_identity.replica_id = replica_id
            request.worker_identity.worker_id = worker_id
            request.reason = reason
            request.timestamp.CopyFrom(create_current_timestamp())

            response = self._grpc_call(
                target_hostname,
                port,
                "RestartTraining",
                request,
                f"RestartTraining(worker={worker_id}, reason='{reason}')",
            )
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.debug(
                "RestartTraining failed: target=%s:%d, replica_id=%d, "
                "worker_id=%s, reason='%s', error=%s",
                target_hostname,
                port,
                replica_id,
                worker_id,
                reason,
                str(e),
            )
            return False

    @retry(max_attempts=3)
    def report_pod_shutdown(
        self, reason: str = "Pod termination", port: Optional[int] = None
    ) -> bool:
        """Report pod shutdown to global manager.

        Args:
            reason: Reason for shutdown
            port: Port of the global manager (uses default_port if None)

        Returns:
            bool: True if shutdown was acknowledged, False otherwise
        """
        target_hostname = get_global_manager_hostname()
        if port is None:
            port = self.port
        try:
            request = manager_pb2.PodShutdownRequest()
            request.worker_identity.CopyFrom(create_worker_identity_proto())
            request.reason = reason
            request.timestamp.CopyFrom(create_current_timestamp())

            logging.warning(
                "Reporting pod shutdown to global manager: target=%s:%d, reason='%s'",
                target_hostname,
                port,
                reason,
            )

            response = self._grpc_call(target_hostname, port, "ReportPodShutdown", request)
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
                "Pod shutdown report failed: target=%s:%d, reason='%s', error=%s",
                target_hostname,
                port,
                reason,
                str(e),
            )
            return False

    def close(self):
        """Close all cached gRPC channels."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False
