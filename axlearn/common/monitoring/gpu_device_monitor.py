# Copyright © 2025 Apple Inc.

"""GPU Device Monitor for fetching GPU metrics."""

import jax
from absl import logging

from axlearn.common.monitoring.device_monitor import DeviceMonitor, DeviceMonitorClient
from axlearn.common.monitoring.gpu_client import NVMLMetrics
from axlearn.common.utils import DeviceUsage as Usage


class GPUMonitorClient(DeviceMonitorClient):
    """Client for fetching GPU metrics"""

    def collect_metrics(self) -> list[Usage]:
        num_devices = NVMLMetrics.get_gpu_device_count()

        usages = []
        for device_id in range(num_devices):
            device_usage = Usage(device_id=device_id)
            device_usage.device_utilization = NVMLMetrics.get_gpu_device_utilization(
                device_id=device_id
            )
            (
                device_usage.hbm_memory_usage_bytes,
                device_usage.hbm_memory_total_bytes,
            ) = NVMLMetrics.get_gpu_device_memory(device_id=device_id)
            device_usage.hbm_memory_bandwidth_utilization = (
                NVMLMetrics.get_gpu_device_memory_utilization(device_id=device_id)
            )

            usages.append(device_usage)

        return usages

    def is_host_idle(self, usages: list[Usage]) -> bool:
        for usage in usages:
            if usage.device_utilization <= 0.1 and usage.hbm_memory_bandwidth_utilization <= 0.1:
                logging.info("GPU device %d is idle.", usage.device_id)
                return True
        return False


def create_gpu_monitor() -> DeviceMonitorClient.Config:
    device_platform: str = jax.local_devices()[0].platform
    assert (
        device_platform == "gpu"
    ), f"device_platform {device_platform} not matching device_monitor gpu."

    monitor_client = GPUMonitorClient.default_config().set(
        platform=device_platform,
    )
    return DeviceMonitor.default_config().set(
        monitor_client=monitor_client,
    )
