"""
Module for GCP Workload Monitoring, enabling performance and heartbeat metrics
to be sent to Google Cloud Monitoring.
"""

import time

from absl import app, logging
from google.api import metric_pb2, monitored_resource_pb2
from google.api_core.exceptions import GoogleAPIError
from google.cloud import monitoring_v3

from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import get_gcp_metadata


class GCPWorkloadMonitoring:
    """A class to send metrics to Google Cloud Monitoring."""

    def __init__(self, project_id, zone, workload_id, replica_id):
        self.project_id = project_id
        self.zone = zone
        self.workload_id = workload_id
        self.replica_id = replica_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{self.project_id}"

    def check_connectivity(self):
        """Checks connectivity to the specified GCP project and zone."""
        try:
            # Check if the project is accessible
            resource_descriptors = self.client.list_monitored_resource_descriptors(
                name=self.project_name
            )
            if resource_descriptors:
                logging.info("Successfully connected to GCP project %s.", self.project_id)
        except GoogleAPIError as e:
            raise ValueError(f"Unable to connect to GCP project {self.project_id}: {e}") from e

    def send_performance_metric(self, perf_metric: float):
        """Send performance metric to Google Cloud Monitoring."""

        metric_type = "compute.googleapis.com/workload/performance"
        resource_type = "compute.googleapis.com/Workload"

        try:
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)

            # Create a TimeSeries object for the step time metric
            series = monitoring_v3.TimeSeries(
                metric=metric_pb2.Metric(
                    type=metric_type,
                ),
                resource=monitored_resource_pb2.MonitoredResource(
                    type=resource_type,
                    labels={
                        "location": self.zone,
                        "workload_id": self.workload_id,
                        "replica_id": self.replica_id,
                    },
                ),
                points=[
                    monitoring_v3.Point(
                        interval=monitoring_v3.TimeInterval(
                            end_time={"seconds": seconds, "nanos": nanos}
                        ),
                        value=monitoring_v3.TypedValue(double_value=perf_metric),
                    ),
                ],
            )

            # Send data to Google Cloud Monitoring
            self.client.create_time_series(
                request={"name": self.project_name, "time_series": [series]}
            )
            logging.info(
                "Perf metric (%.3f) successfully sent to GCP resource %s.",
                perf_metric,
                resource_type,
            )
        except GoogleAPIError as e:
            logging.error("Failed to send metric to GCP. Metric: %s, Error: %s", metric_type, e)
        except Exception as e:
            logging.error("Unexpected Error. Metric: %s, Error: %s", metric_type, e)

    def send_heartbeat_metric(self, acc_index: str, jax_process_index: str):
        """Send heartbeat metric to Google Cloud Monitoring."""

        is_alive = True
        metric_type = "compute.googleapis.com/workload_process/heartbeat"
        resource_type = "compute.googleapis.com/WorkloadProcess"

        try:
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)

            # Create a TimeSeries object for the heartbeat metric
            series = monitoring_v3.TimeSeries(
                metric=metric_pb2.Metric(
                    type=metric_type,
                    labels={
                        "gpu_index": acc_index,
                        "instance_id": get_gcp_metadata(category="instance", attribute="id"),
                    },
                ),
                resource=monitored_resource_pb2.MonitoredResource(
                    type=resource_type,
                    labels={
                        "project_id": self.project_id,
                        "location": self.zone,
                        "workload_id": self.workload_id,
                        "replica_id": self.replica_id,
                        "process_id": jax_process_index,
                    },
                ),
                points=[
                    monitoring_v3.Point(
                        interval=monitoring_v3.TimeInterval(
                            end_time={"seconds": seconds, "nanos": nanos}
                        ),
                        value=monitoring_v3.TypedValue(bool_value=is_alive),
                    ),
                ],
            )

            # Send data to Google Cloud Monitoring
            self.client.create_time_series(
                request={"name": self.project_name, "time_series": [series]}
            )
            logging.info(
                "Heartbeat metric successfully sent to GCP with value %s for resource %s.",
                is_alive,
                resource_type,
            )
        except GoogleAPIError as e:
            logging.error("Failed to send metric to GCP. Metric: %s, Error: %s", metric_type, e)
        except Exception as e:
            logging.error("Unexpected Error. Metric: %s, Error: %s", metric_type, e)


def main(argv):
    del argv  # Unused argv

    # Initialize the monitoring class
    enable_gcp_workload_monitoring = gcp_settings("enable_gcp_workload_monitoring", default=False)

    if enable_gcp_workload_monitoring:
        monitor = GCPWorkloadMonitoring(
            project_id=gcp_settings("project", required=True),
            zone=gcp_settings("zone", required=True),
            workload_id=gcp_settings("workload_id", required=True),
            replica_id=gcp_settings("replica_id", default="0"),
        )
        # Check Connectivity
        monitor.check_connectivity()
        # Example: Send metrics
        monitor.send_performance_metric(perf_metric=0.123)
        monitor.send_heartbeat_metric(acc_index="0", jax_process_index="0")


if __name__ == "__main__":
    app.run(main)
