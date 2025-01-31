"""
Unit tests for GCPWorkloadMonitoring in axlearn.cloud.gcp.monitoring.monitor_workload.

This module provides tests for verifying the behavior of GCPWorkloadMonitoring.
The tests ensure connectivity, performance metrics, and heartbeat metrics are handled correctly.
"""

import unittest
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import GoogleAPIError

from axlearn.cloud.gcp.monitoring.monitor_workload import GCPWorkloadMonitoring


class TestGCPWorkloadMonitoring(unittest.TestCase):
    """
    Test suite for the GCPWorkloadMonitoring class.

    It tests:
    - Connectivity validation.
    - Performance metric submission.
    - Heartbeat metric submission.
    """

    def setUp(self):
        self.project_id = "test-project"
        self.zone = "test-zone"
        self.workload_id = "test-workload"
        self.replica_id = "test-replica"

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_check_connectivity_success(self, mock_metric_service_client):
        # Mock the client and its response
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.list_monitored_resource_descriptors.return_value = [MagicMock()]

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        self.monitor.check_connectivity()
        mock_metric_service_client.assert_called_once()
        mock_client_instance.list_monitored_resource_descriptors.assert_called_once_with(
            name=f"projects/{self.project_id}"
        )

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_check_connectivity_failure(self, mock_metric_service_client):
        # Mock the client and simulate an error
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.list_monitored_resource_descriptors.side_effect = GoogleAPIError(
            "API Error"
        )

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        with self.assertRaises(ValueError):
            self.monitor.check_connectivity()

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_send_performance_metric_success(self, mock_metric_service_client):
        # Mock the client and its response
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.create_time_series.return_value = None

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        self.monitor.send_performance_metric(perf_metric=0.123)
        mock_metric_service_client.assert_called_once()
        mock_client_instance.create_time_series.assert_called_once()

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_send_performance_metric_failure(self, mock_metric_service_client):
        # Mock the client and simulate an error
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.create_time_series.side_effect = GoogleAPIError("API Error")

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        with patch(
            "axlearn.cloud.gcp.monitoring.monitor_workload.logging.error"
        ) as mock_logging_error:
            self.monitor.send_performance_metric(perf_metric=0.123)
            mock_logging_error.assert_called()

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.get_gcp_metadata")
    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_send_heartbeat_metric_success(self, mock_metric_service_client, mock_get_gcp_metadata):
        # Mock the client and its response
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.create_time_series.return_value = None

        # Mock metadata fetching
        mock_get_gcp_metadata.return_value = "test-instance-id"

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        self.monitor.send_heartbeat_metric(local_rank="0", global_rank="1")
        mock_metric_service_client.assert_called_once()
        mock_client_instance.create_time_series.assert_called_once()

    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.get_gcp_metadata")
    @patch("axlearn.cloud.gcp.monitoring.monitor_workload.monitoring_v3.MetricServiceClient")
    def test_send_heartbeat_metric_failure(self, mock_metric_service_client, mock_get_gcp_metadata):
        # Mock the client and simulate an error
        mock_client_instance = MagicMock()
        mock_metric_service_client.return_value = mock_client_instance
        mock_client_instance.create_time_series.side_effect = GoogleAPIError("API Error")

        # Mock metadata fetching
        mock_get_gcp_metadata.return_value = "test-instance-id"

        # Reinitialize monitor after patching
        self.monitor = GCPWorkloadMonitoring(
            project_id=self.project_id,
            zone=self.zone,
            workload_id=self.workload_id,
            replica_id=self.replica_id,
        )

        with patch(
            "axlearn.cloud.gcp.monitoring.monitor_workload.logging.error"
        ) as mock_logging_error:
            self.monitor.send_heartbeat_metric(local_rank="0", global_rank="1")
            mock_logging_error.assert_called()


if __name__ == "__main__":
    unittest.main()
