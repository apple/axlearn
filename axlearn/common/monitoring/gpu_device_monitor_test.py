# Copyright © 2025 Apple Inc.

"""Tests for gpu_device_monitor.py"""

import time
from unittest import mock, skipIf

import jax
from absl.testing import parameterized

from axlearn.common.monitoring.device_monitor import DeviceMonitor
from axlearn.common.monitoring.gpu_device_monitor import GPUMonitorClient


@skipIf(jax.default_backend() != "gpu", "Skip due to not a gpu backend")
class TestGPUDeviceMonitor(parameterized.TestCase):
    """
    Test class for GPUDeviceMonitor methods
    """

    def setUp(self):
        self.nvml_metrics_patch = mock.patch(
            "axlearn.common.monitoring.gpu_device_monitor.NVMLMetrics"
        )
        self.nvml_metrics_mock = self.nvml_metrics_patch.start()

        self.nvml_metrics_mock.get_gpu_device_count.return_value = 8

        self.nvml_metrics_mock.get_gpu_device_utilization.return_value = 10.0
        self.nvml_metrics_mock.get_gpu_device_memory.return_value = (100.0, 200.0)
        self.nvml_metrics_mock.get_gpu_device_memory_utilization.return_value = 15.0

    def tearDown(self):
        self.nvml_metrics_patch.stop()

    def test_gpu_device_monitor(self):
        cfg = GPUMonitorClient.default_config()
        device_monitor = cfg.instantiate()

        metrics = device_monitor.collect_metrics()

        self.assertLen(metrics, 8)
        for i in range(8):
            self.assertEqual(metrics[i].device_id, i)
            self.assertEqual(metrics[i].device_utilization, 10.0)
            self.assertEqual(metrics[i].hbm_memory_usage_bytes, 100.0)
            self.assertEqual(metrics[i].hbm_memory_total_bytes, 200.0)
            self.assertEqual(metrics[i].hbm_memory_bandwidth_utilization, 15.0)
            self.assertEqual(metrics[i].device_duty_cycle_percent, None)

        self.nvml_metrics_mock.get_gpu_device_count.assert_called_once()
        self.nvml_metrics_mock.get_gpu_device_utilization.assert_has_calls(
            [mock.call(device_id=i) for i in range(8)]
        )
        self.nvml_metrics_mock.get_gpu_device_memory.assert_has_calls(
            [mock.call(device_id=i) for i in range(8)]
        )
        self.nvml_metrics_mock.get_gpu_device_memory_utilization.assert_has_calls(
            [mock.call(device_id=i) for i in range(8)]
        )

    def test_gpu_device_monitor_is_idle_false(self):
        cfg = GPUMonitorClient.default_config()
        device_monitor = cfg.instantiate()

        usages = device_monitor.collect_metrics()

        self.assertFalse(device_monitor.is_host_idle(usages))

    def test_gpu_device_monitor_is_idle_true(self):
        cfg = GPUMonitorClient.default_config()
        device_monitor = cfg.instantiate()

        self.nvml_metrics_mock.get_gpu_device_utilization.return_value = 0.1
        self.nvml_metrics_mock.get_gpu_device_memory_utilization.return_value = 0.1

        usages = device_monitor.collect_metrics()

        self.assertTrue(device_monitor.is_host_idle(usages))

    def test_gpu_device_monitor_start_monitoring(self):
        gpu_monitor_cfg = GPUMonitorClient.default_config()
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client=gpu_monitor_cfg,
            check_interval_in_sec=0.1,
            log_every_n=1,
        )

        self.nvml_metrics_mock.get_gpu_device_utilization.return_value = 0.1
        self.nvml_metrics_mock.get_gpu_device_memory_utilization.return_value = 0.1

        device_monitor = device_monitor_cfg.instantiate()

        with device_monitor.start_monitoring():
            time.sleep(0.3)
            self.assertTrue(device_monitor.is_host_idle())
