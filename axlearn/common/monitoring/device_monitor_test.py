# Copyright © 2024 Apple Inc.

"""Test class for device_mon.py."""
import time

from absl.testing import parameterized

from axlearn.common.config import config_class
from axlearn.common.monitoring.device_monitor import DeviceMonitor, DeviceMonitorClient
from axlearn.common.utils import DeviceUsage as Usage


class MockDeviceMonitorClient(DeviceMonitorClient):
    """Mock Client for fetching metrics from devices."""

    @config_class
    class Config(DeviceMonitorClient.Config):
        """Configures DeviceMonitorClient."""

        fake_usage: list[Usage] = []

    def __init__(self, cfg: Config):
        """Initialize the DeviceMonitorClient."""
        super().__init__(cfg)
        cfg = self.config
        self._platform = cfg.platform

    def collect_metrics(self) -> list[Usage]:
        """Collect metrics from the device, it should be empty."""
        return self.config.fake_usage

    def is_host_idle(self, usages: list[Usage]) -> bool:
        """Check if the devices on the host are idle, always return False."""
        # Make sure the usages are empty.
        return (
            usages[0].hbm_memory_bandwidth_utilization <= 0.1
            and usages[0].tensorcore_utilization <= 0.1
        )


class TestDeviceMonitor(parameterized.TestCase):
    """Test class for MonitorClient and DeviceMonitor."""

    def test_client(self):
        """Test the Busu MonitorClient."""
        fake_usage = [
            Usage(
                device_id=0,
                tensorcore_duty_cycle_percent=100.0,
                tensorcore_utilization=1.0,
                hbm_memory_total_bytes=100,
                hbm_memory_usage_bytes=50,
                hbm_memory_bandwidth_utilization=30.0,
            )
        ]
        mock_monitor_client = MockDeviceMonitorClient.default_config().set(
            fake_usage=fake_usage,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client=mock_monitor_client,
            check_interval_in_sec=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertFalse(device_monitor.is_host_idle())

    def test_client_idle(self):
        """Test the Idle MonitorClient."""
        fake_usage = [
            Usage(
                device_id=0,
                tensorcore_duty_cycle_percent=0.0,
                tensorcore_utilization=0.0,
                hbm_memory_total_bytes=100,
                hbm_memory_usage_bytes=50,
                hbm_memory_bandwidth_utilization=0.0,
            )
        ]
        mock_monitor_client = MockDeviceMonitorClient.default_config().set(
            fake_usage=fake_usage,
        )
        device_monitor_cfg = DeviceMonitor.default_config().set(
            monitor_client=mock_monitor_client,
            check_interval_in_sec=0.1,
            log_every_n=1,
        )
        device_monitor = device_monitor_cfg.instantiate()
        with device_monitor.start_monitoring():
            time.sleep(0.2)
            self.assertTrue(device_monitor.is_host_idle())
