"""FT trainer gRPC server.

Provides server functionality for FT trainer manager including:
- Worker status reporting (replica managers)
- Replica status reporting (global manager)
- Restart request handling (both levels)
"""

import logging
import threading
import time
from concurrent import futures
from typing import Dict, Optional

import grpc

from axlearn.ft import manager_pb2, manager_pb2_grpc
from axlearn.ft.manager_client import ManagerClient
from axlearn.ft.utils import (
    DEFAULT_MANAGER_PORT,
    TrainerProcessController,
    WorkerIdentity,
    WorkerStatusRecord,
    create_current_timestamp,
    get_all_worker_hostnames,
    get_worker_id,
    get_worker_identity,
    worker_status_record_to_proto,
)


class ManagerServer(manager_pb2_grpc.ManagerServiceServicer):
    """FT trainer manager server.

    Handles both replica manager and global manager responsibilities based on worker identity.
    Provides gRPC server functionality with role-based request routing.
    """

    def __init__(
        self,
        port: int = DEFAULT_MANAGER_PORT,
        max_worker_threads: int = 10,
        process_controller: Optional[TrainerProcessController] = None,
    ):
        """Initialize manager server.

        Args:
            port: Port to listen on
            max_worker_threads: Maximum number of worker threads for gRPC server
            process_controller: Controller for trainer process (replica managers only)
        """
        self._worker_identity = get_worker_identity()
        self.port = port
        self.max_worker_threads = max_worker_threads
        self.server: Optional[grpc.Server] = None
        self.process_controller = process_controller
        self._registry_lock = threading.Lock()

        # Initialize registries based on role
        if self._worker_identity.is_replica_manager:
            self._status_registry: Dict[str, WorkerStatusRecord] = {}
            logging.info(
                "Server initialized as replica manager for replica %d",
                self._worker_identity.replica_id,
            )

        if self._worker_identity.is_global_manager:
            self._replica_registry: Dict[int, Dict] = {}
            logging.info("Server initialized as global manager")

    def _validate_role(self, required_role: str, operation: str) -> None:
        """Validate worker role for operation.

        Args:
            required_role: Required role ('replica_manager' or 'global_manager')
            operation: Description of operation for error message

        Raises:
            grpc.RpcError: If role validation fails
        """
        if required_role == "replica_manager" and not self._worker_identity.is_replica_manager:
            raise grpc.RpcError(f"Only replica managers can {operation}")
        elif required_role == "global_manager" and not self._worker_identity.is_global_manager:
            raise grpc.RpcError(f"Only global managers can {operation}")

    def _create_error_response(self, context, code: grpc.StatusCode, message: str) -> None:
        """Create standardized error response.

        Args:
            context: gRPC context
            code: gRPC status code
            message: Error message
        """
        logging.error(message)
        context.set_code(code)
        context.set_details(message)

    def _create_success_response(self, response_type, **kwargs):
        """Create standardized success response.

        Args:
            response_type: Response message type
            **kwargs: Additional response fields

        Returns:
            Configured response message
        """
        response = response_type()
        response.acknowledged = True
        response.timestamp.CopyFrom(create_current_timestamp())

        for key, value in kwargs.items():
            if hasattr(response, key):
                setattr(response, key, value)

        return response

    def start_server(self):
        """Start the gRPC server."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_worker_threads))
        self.server.add_insecure_port(f"[::]:{self.port}")

        logging.info(
            "Starting server on port %d (replica=%d, worker=%d)",
            self.port,
            self._worker_identity.replica_id,
            self._worker_identity.worker_id,
        )

        # Add the servicer to the server
        manager_pb2_grpc.add_ManagerServiceServicer_to_server(self, self.server)

        self.server.start()
        return self.server

    def stop_server(self, grace_period: int = 10):
        """Stop the gRPC server."""
        if self.server:
            self.server.stop(grace_period)
            logging.info("Manager server stopped")

    def ReportStatus(
        self, request: manager_pb2.StatusUpdate, context
    ) -> manager_pb2.StatusUpdateResponse:
        """Handle worker status reports (replica managers only)."""
        try:
            self._validate_role("replica_manager", "handle worker status reports")
            return self._handle_worker_status(request, context)
        except grpc.RpcError as e:
            self._create_error_response(context, grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise
        except Exception as e:
            self._create_error_response(context, grpc.StatusCode.INTERNAL, f"Unexpected error: {e}")
            raise

    def ReportReplicaStatus(
        self, request: manager_pb2.ReplicaStatusUpdate, context
    ) -> manager_pb2.ReplicaStatusResponse:
        """Handle replica status reports (global managers only)."""
        try:
            self._validate_role("global_manager", "handle replica status reports")
            return self._handle_replica_status(request, context)
        except grpc.RpcError as e:
            self._create_error_response(context, grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise
        except Exception as e:
            self._create_error_response(context, grpc.StatusCode.INTERNAL, f"Unexpected error: {e}")
            raise

    def _handle_worker_status(
        self, request: manager_pb2.StatusUpdate, context  # pylint: disable=unused-argument
    ) -> manager_pb2.StatusUpdateResponse:
        """Handle worker status reports when running as replica manager."""
        worker_identity = WorkerIdentity(
            hostname=request.worker_identity.hostname,
            replica_id=request.worker_identity.replica_id,
            worker_id=request.worker_identity.worker_id,
        )

        with self._registry_lock:
            self._status_registry[request.worker_identity.hostname] = WorkerStatusRecord(
                worker_identity=worker_identity,
                training_step=request.worker_status.training_step,
                last_update=time.time(),
                timestamp=request.timestamp,
            )

        logging.debug(
            "Worker status received: hostname=%s, replica=%d, worker=%d, step=%d, registry_size=%d",
            request.worker_identity.hostname,
            request.worker_identity.replica_id,
            request.worker_identity.worker_id,
            request.worker_status.training_step,
            len(self._status_registry),
        )

        return self._create_success_response(manager_pb2.StatusUpdateResponse)

    def _handle_replica_status(
        self, request: manager_pb2.ReplicaStatusUpdate, context  # pylint: disable=unused-argument
    ) -> manager_pb2.ReplicaStatusResponse:
        """Handle replica status reports from replica heads."""
        replica_id = request.replica_id

        # Extract worker status entries
        worker_statuses = []
        for worker_entry in request.replica_status.worker_statuses:
            worker_statuses.append(
                {
                    "worker_identity": {
                        "replica_id": worker_entry.worker_identity.replica_id,
                        "worker_id": worker_entry.worker_identity.worker_id,
                        "hostname": worker_entry.worker_identity.hostname,
                    },
                    "training_step": worker_entry.worker_status.training_step,
                    "last_update": worker_entry.last_update.seconds
                    + worker_entry.last_update.nanos / 1e9,
                }
            )

        # Update replica status registry
        with self._registry_lock:
            self._replica_registry[replica_id] = {
                "replica_id": replica_id,
                "replica_status": {
                    "total_workers": request.replica_status.total_workers,
                    "reported_workers": request.replica_status.reported_workers,
                    "worker_statuses": worker_statuses,
                },
                "last_update": time.time(),
                "timestamp": request.timestamp,
            }

        logging.info(
            "Replica status received: replica=%d, workers=%d/%d, "
            "individual_workers=%d, registry_size=%d",
            replica_id,
            request.replica_status.reported_workers,
            request.replica_status.total_workers,
            len(worker_statuses),
            len(self._replica_registry),
        )

        return self._create_success_response(manager_pb2.ReplicaStatusResponse)

    def RestartReplicaTraining(
        self, request: manager_pb2.RestartReplicaRequest, context
    ) -> manager_pb2.RestartReplicaResponse:
        """Handle replica restart requests from global manager."""
        try:
            self._validate_role("replica_manager", "handle RestartReplicaTraining")

            # Verify replica ID matches
            if request.replica_id != self._worker_identity.replica_id:
                error_msg = (
                    f"Replica ID mismatch: request={request.replica_id}, "
                    f"actual={self._worker_identity.replica_id}"
                )
                self._create_error_response(context, grpc.StatusCode.INVALID_ARGUMENT, error_msg)
                raise grpc.RpcError(error_msg)

            logging.warning(
                "RestartReplicaTraining request received: replica_id=%d, reason='%s'",
                request.replica_id,
                request.reason,
            )

            # Forward restart request to all workers in this replica
            forward_results = self._forward_restart_to_workers(request.reason)
            successful_forwards = sum(1 for result in forward_results.values() if result)
            total_forwards = len(forward_results)

            logging.info(
                "Replica %d restart: workers=%d/%d",
                request.replica_id,
                successful_forwards,
                total_forwards,
            )

            success = successful_forwards > 0
            message = (
                f"Replica {request.replica_id}: workers={successful_forwards}/{total_forwards}"
            )

            return self._create_success_response(
                manager_pb2.RestartReplicaResponse, acknowledged=success, message=message
            )

        except grpc.RpcError:
            raise
        except Exception as e:
            self._create_error_response(
                context, grpc.StatusCode.INTERNAL, f"RestartReplicaTraining error: {e}"
            )
            raise

    def RestartTraining(
        self, request: manager_pb2.RestartRequest, context
    ) -> manager_pb2.RestartResponse:
        """Handle worker restart requests from replica manager."""
        try:
            worker_identity = request.worker_identity

            logging.warning(
                "RestartTraining request received: target=%s (replica=%d, worker=%d), reason='%s'",
                worker_identity.hostname,
                worker_identity.replica_id,
                worker_identity.worker_id,
                request.reason,
            )

            logging.info("Worker processing restart request")

            # Attempt to terminate the trainer process
            success = False
            message = ""

            if self.process_controller:
                success = self.process_controller.terminate_training(request.reason)
                if success:
                    message = f"Worker {worker_identity.hostname} training termination initiated"
                    logging.info(message)
                else:
                    message = f"Worker {worker_identity.hostname} has no active training process"
                    logging.warning(message)
            else:
                message = f"Worker {worker_identity.hostname} has no process controller"
                logging.error(message)
                success = False

            return self._create_success_response(
                manager_pb2.RestartResponse, acknowledged=success, message=message
            )

        except grpc.RpcError:
            raise
        except Exception as e:
            self._create_error_response(
                context, grpc.StatusCode.INTERNAL, f"RestartTraining error: {e}"
            )
            raise

    def _forward_restart_to_workers(self, reason: str) -> Dict[str, bool]:
        """Forward restart request to all workers in this replica (replica manager only).

        Args:
            reason: Reason for restart

        Returns:
            dict: Map of hostname -> success status
        """
        if not self._worker_identity.is_replica_manager:
            logging.error("Only replica managers can forward restart requests")
            return {}

        results = {}
        worker_hostnames = get_all_worker_hostnames()

        with ManagerClient() as client:
            for hostname in worker_hostnames:
                try:
                    logging.info("Forwarding restart to worker: %s", hostname)

                    worker_id = get_worker_id()

                    success = client.restart_worker(
                        hostname, self._worker_identity.replica_id, reason, worker_id
                    )
                    results[hostname] = success

                    if success:
                        logging.info("Worker %s acknowledged restart request", hostname)
                    else:
                        logging.warning("Worker %s failed to acknowledge restart request", hostname)

                except Exception as e:  # pylint: disable=broad-exception-caught
                    logging.error("Failed to forward restart to worker %s: %s", hostname, e)
                    results[hostname] = False

        return results

    def get_replica_status(self) -> manager_pb2.ReplicaStatus:
        """Generate current replica status (only available for replica managers)."""
        if not self._worker_identity.is_replica_manager:
            raise ValueError("get_replica_status only available for replica managers")

        with self._registry_lock:
            status = manager_pb2.ReplicaStatus()
            status.total_workers = len(get_all_worker_hostnames())
            status.reported_workers = len(self._status_registry)

            # Populate individual worker statuses
            for worker_record in self._status_registry.values():
                worker_status_entry = worker_status_record_to_proto(worker_record)
                status.worker_statuses.append(worker_status_entry)

            return status

    def get_global_status(self) -> Dict[str, int]:
        """Get global status summary (only available for global manager)."""
        if not self._worker_identity.is_global_manager:
            raise ValueError("get_global_status only available for global manager")

        with self._registry_lock:
            total_workers = 0
            total_reported_workers = 0

            for replica_data in self._replica_registry.values():
                replica_status = replica_data.get("replica_status", {})
                total_workers += replica_status.get("total_workers", 0)
                total_reported_workers += replica_status.get("reported_workers", 0)

            return {
                "total_replicas": len(self._replica_registry),
                "total_workers": total_workers,
                "total_reported_workers": total_reported_workers,
            }

    def get_detailed_global_status(self) -> Dict:
        """Get detailed global status including individual worker information.

        Only available for global manager.
        """
        if not self._worker_identity.is_global_manager:
            raise ValueError("get_detailed_global_status only available for global manager")

        with self._registry_lock:
            detailed_status = {"total_replicas": len(self._replica_registry), "replicas": {}}

            for replica_id, replica_data in self._replica_registry.items():
                replica_status = replica_data.get("replica_status", {})
                detailed_status["replicas"][replica_id] = {
                    "total_workers": replica_status.get("total_workers", 0),
                    "reported_workers": replica_status.get("reported_workers", 0),
                    "last_update": replica_data.get("last_update"),
                    "workers": replica_status.get("worker_statuses", []),
                }

            return detailed_status
