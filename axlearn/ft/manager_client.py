"""FT manager gRPC client.

Provides client functionality for FT manager operations including
worker status reporting, replica status reporting, and restart requests.
"""

import logging
from typing import Optional

import grpc

from axlearn.ft import manager_pb2, manager_pb2_grpc
from axlearn.ft.utils import (
    DEFAULT_MANAGER_PORT,
    create_current_timestamp,
    create_worker_identity_proto,
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
        logging.info("ManagerClient initialized: timeout=%.1fs, port=%d", timeout, port)

    def _create_stub(self, hostname: str, port: int) -> manager_pb2_grpc.ManagerServiceStub:
        """Create a gRPC stub for the target host."""
        channel = grpc.insecure_channel(f"{hostname}:{port}")
        return manager_pb2_grpc.ManagerServiceStub(channel)

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
            # Create request
            request = manager_pb2.StatusUpdate()
            request.worker_identity.CopyFrom(create_worker_identity_proto())
            request.worker_status.CopyFrom(worker_status)
            request.timestamp.CopyFrom(create_current_timestamp())

            logging.debug(
                "Sending status update: target=%s:%d, worker=%s, step=%d",
                target_host,
                port,
                request.worker_identity.hostname,
                worker_status.training_step,
            )

            # Send request
            stub = self._create_stub(target_host, port)
            response = stub.ReportStatus(request, timeout=self.timeout)

            logging.info(
                "Status report successful: target=%s:%d, step=%d, acknowledged=%s",
                target_host,
                port,
                worker_status.training_step,
                response.acknowledged,
            )
            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
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

            logging.debug(
                "Sending replica status: replica_id=%d, target=%s:%d, workers=%d/%d",
                replica_id,
                target_host,
                port,
                replica_status.reported_workers,
                replica_status.total_workers,
            )

            # Send request
            stub = self._create_stub(target_host, port)
            response = stub.ReportReplicaStatus(request, timeout=self.timeout)

            logging.info(
                "Replica status sent: replica_id=%d, target=%s:%d, acknowledged=%s",
                replica_id,
                target_host,
                port,
                response.acknowledged,
            )
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
            # Create request
            request = manager_pb2.RestartReplicaRequest()
            request.replica_id = replica_id
            request.reason = reason
            request.timestamp.CopyFrom(create_current_timestamp())

            logging.info(
                "Sending RestartReplicaTraining: target=%s:%d, replica_id=%d, reason='%s'",
                target_hostname,
                port,
                replica_id,
                reason,
            )

            # Send request
            stub = self._create_stub(target_hostname, port)
            response = stub.RestartReplicaTraining(request, timeout=self.timeout)

            logging.info(
                "RestartReplicaTraining response: target=%s:%d, replica_id=%d, "
                "acknowledged=%s, message='%s'",
                target_hostname,
                port,
                replica_id,
                response.acknowledged,
                response.message,
            )

            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
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
        """Send restart request to a specific worker (Replica → Worker level).

        Args:
            target_hostname: Hostname of the worker
            replica_id: Replica ID of the target worker
            reason: Reason for restart
            worker_id: Worker ID (derived from hostname if None)
            port: Port of the worker (uses default_port if None)

        Returns:
            bool: True if restart was acknowledged, False otherwise
        """
        if port is None:
            port = self.port
        try:
            # Create request
            request = manager_pb2.RestartRequest()
            request.worker_identity.hostname = target_hostname
            request.worker_identity.replica_id = replica_id
            request.worker_identity.worker_id = worker_id
            request.reason = reason
            request.timestamp.CopyFrom(create_current_timestamp())

            logging.info(
                "Sending RestartTraining: target=%s:%d, worker_id=%d, reason='%s'",
                target_hostname,
                port,
                worker_id,
                reason,
            )

            # Send request
            stub = self._create_stub(target_hostname, port)
            response = stub.RestartTraining(request, timeout=self.timeout)

            logging.info(
                "RestartTraining response: target=%s:%d, worker_id=%d, "
                "acknowledged=%s, message='%s'",
                target_hostname,
                port,
                worker_id,
                response.acknowledged,
                response.message,
            )

            return response.acknowledged

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
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

    def close(self):
        """Close client resources (no-op for simplified implementation)."""
        pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False
