# Copyright © 2024 Apple Inc.

"""Test class for tpu_device_mon.py."""
import time

from absl.testing import parameterized
from tpu_info import device

from axlearn.cloud.gcp.monitoring import tpu_client
from axlearn.cloud.gcp.monitoring.tpu_client_test import DummyTpuMetricV2Server
from axlearn.cloud.gcp.monitoring.tpu_device_monitor import TPUMonitorClient
from axlearn.common.monitoring.device_monitor import DeviceMonitor


class TestMetrics(parameterized.TestCase):
    """Test class for TPUMonitorClient and DeviceMonitor."""

    def test_tpu_client(self):
        """Test the TPUMonitorClient."""
        expected_usage = [
            tpu_client.Usage(
                device_id=i,
                hbm_memory_total_bytes=int(1.02803439616e11),
                hbm_memory_usage_bytes=int(6.5e10),
                tensorcore_duty_cycle_percent=100.0,
                tensorcore_utilization=1.0 * (1 + i),
                hbm_memory_bandwidth_utilization=30.0,
            )
            for i in range(4)
        ]
        # Test the case where the metrics are supported.
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client_cfg = TPUMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            metric_list=list(tpu_client.MetricV2Name),
            addr=metric_server_addr,
        )
        tpu_monitor_client = tpu_monitor_client_cfg.instantiate()
        chip_metrics = tpu_monitor_client.collect_metrics()
        self.assertListEqual(chip_metrics, expected_usage)
        self.assertFalse(tpu_monitor_client.is_host_idle(chip_metrics))

    def test_tpu_client_no_metric_supported(self):
        """Test the TPUMonitorClient when no metric is supported."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client_cfg = TPUMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            metric_list=list(tpu_client.MetricName),
            addr=metric_server_addr,
        )
        tpu_monitor_client = tpu_monitor_client_cfg.instantiate()
        chip_metrics = tpu_monitor_client.collect_metrics()
        self.assertListEqual(chip_metrics, [])
        self.assertFalse(tpu_monitor_client.is_host_idle(chip_metrics))

    def test_device_monitor(self):
        """Test the TPUMonitorClient."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2()
        tpu_monitor_client = TPUMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client=tpu_monitor_client,
            check_interval_in_sec=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertFalse(device_monitor.is_host_idle())

    def test_device_monitor_idle(self):
        """Test the TPUMonitorClient."""
        metric_server_addr = DummyTpuMetricV2Server.fake_tpu_metrics_v2(
            metric_file="sample_metrics_idle.txt"
        )
        tpu_monitor_client = TPUMonitorClient.default_config().set(
            chip_type=device.TpuChip.V5P,
            addr=metric_server_addr,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client=tpu_monitor_client,
            check_interval_in_sec=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertTrue(device_monitor.is_host_idle())
