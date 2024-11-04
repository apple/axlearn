# Copyright © 2024 Apple Inc.
"""TPU Device Monitor for fetching TPU metrics from libtpu and tpu-device-plugin."""

import os
from typing import Optional, Sequence

import jax
from absl import logging
from tpu_info import device

from axlearn.cloud.gcp.monitoring.tpu_client import TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR
from axlearn.cloud.gcp.monitoring.tpu_client import MetricV2Name as MetricName
from axlearn.cloud.gcp.monitoring.tpu_client import get_chip_metrics_v2 as get_chip_metrics
from axlearn.cloud.gcp.monitoring.tpu_client import (
    validate_available_metrics_v2 as validate_available_metrics,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.monitoring.device_monitor import DeviceMonitor, DeviceMonitorClient
from axlearn.common.utils import DeviceUsage as Usage

DEVICE_KIND_TO_CHIP_TYPE = {
    "TPU v4": device.TpuChip.V4,
    "TPU v5": device.TpuChip.V5P,
    "TPU v5 lite": device.TpuChip.V5E,
    "TPU v6e": device.TpuChip.V6E,
}


class TPUMonitorClient(DeviceMonitorClient):
    """Client for fetching TPU metrics from libtpu."""

    @config_class
    class Config(DeviceMonitorClient.Config):
        """Configures TPUMonitorClient.

        Fields:
            chip_type: The type of TPU chip.
            metrics: The list of metrics to be fetched.
                Currently, we use HBM_MEMORY_BANDWIDTH_UTILIZATION and TENSORCORE_UTILIZATION,
                to detect the idle status of the host.
            addr: The address of the TPU metrics.
                Currently, we use the tpu-device-plugin metrics server.

        """

        chip_type: Required[device.TpuChip] = REQUIRED
        metric_list: Sequence[MetricName] = [
            MetricName.HBM_MEMORY_BANDWIDTH_UTILIZATION,
            MetricName.TENSORCORE_UTILIZATION,
        ]
        addr: Optional[str] = TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._enabled = validate_available_metrics(cfg.metric_list, addr=cfg.addr)
        logging.log_if(logging.ERROR, not self._enabled, "TPU metrics are not supported.")

    def collect_metrics(self) -> list[Usage]:
        """Collect TPU metrics."""
        if not self._enabled:  # Return empty list if we see any unsupported metrics.
            return []
        cfg: TPUMonitorClient.Config = self.config
        usages = get_chip_metrics(cfg.metric_list, chip_type=cfg.chip_type, addr=cfg.addr)
        # TODO(kelvin-zou): get DCN metrics from container.
        return usages

    def is_host_idle(self, usages: list[Usage]) -> bool:
        """Check if the TPU device on the host are idle."""
        for usage in usages:
            if (
                usage.hbm_memory_bandwidth_utilization <= 0.1
                and usage.tensorcore_utilization <= 0.1
            ):
                logging.info("TPU device %d is idle.", usage.device_id)
                return True
        return False


def create_tpu_monitor() -> DeviceMonitor.Config:
    """Registers a tpu monitor class."""
    if os.environ.get("NODE_IP") is None:
        logging.error("NODE_IP is not set, skip registering TPU monitor.")
        return None
    device_platform: str = jax.local_devices()[0].platform
    device_kind = jax.local_devices()[0].device_kind
    assert (
        device_platform == "tpu"
    ), f"device_platform {device_platform} not matching device_monitor tpu."
    node_ip: str = os.environ["NODE_IP"]
    monitor_client = TPUMonitorClient.default_config().set(
        platform=device_platform,
        chip_type=DEVICE_KIND_TO_CHIP_TYPE[device_kind],
        addr=f"{node_ip}:2112",  # Default port:2112 for tpu-device-plugin.
    )
    return DeviceMonitor.default_config().set(
        monitor_client=monitor_client,
    )
