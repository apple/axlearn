# Copyright © 2024 Apple Inc.

# Some of the code in this file is adapted from:
#
# AI-Hypercomputer/cloud-accelerator-diagnostics:
# Copyright 2023 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Reference:
# https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/metrics.py

"""Client for fetching TPU metrics from libtpu."""
import enum
import urllib.request
from typing import Sequence

import grpc
from absl import logging
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families
from tpu_info import device
from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc

from axlearn.common.utils import DeviceUsage as Usage

# Default address for libtpu metrics server.
# Reference:
# https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/7d2b2921fc9393a3dec7be5440e25132c217549b/tpu_info/tpu_info/metrics.py#L47
LIB_TPU_METRICS_SERVER_ADDR = "localhost:8431"


# Default address for tpu-device-plugin metrics server.
# Reference:
# https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/autoscaling-tpu#metrics
TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR = "localhost:2112"


# Interface names for libtpu metrics.
# Reference:
# https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/7d2b2921fc9393a3dec7be5440e25132c217549b/tpu_info/tpu_info/metrics.py#L29
class MetricName(enum.Enum):
    """Metric names defined in libtpu."""

    TENSORCORE_DUTY_CYCLE_PERCENT = "tpu.runtime.tensorcore.dutycycle.percent"
    HBM_MEMORY_TOTAL_BYTES = "tpu.runtime.hbm.memory.total.bytes"
    HBM_MEMORY_USAGE_BYTES = "tpu.runtime.hbm.memory.usage.bytes"


# Interface names for tpu-device-plugin, which are plumbed from sources like libtpu.
class MetricV2Name(enum.Enum):
    """Metric names defined in tpu-device-plugin."""

    TENSORCORE_DUTY_CYCLE_PERCENT = "duty_cycle_node"
    TENSORCORE_UTILIZATION = "tensorcore_utilization"
    HBM_MEMORY_TOTAL_BYTES = "memory_total"
    HBM_MEMORY_USAGE_BYTES = "memory_used"
    HBM_MEMORY_BANDWIDTH_UTILIZATION = "memory_bandwidth_utilization"


# At the moment all architecures have 4 chips per node.
# We may need to revisit this assumption for future gens.
CHIPS_PER_NODE = {
    device.TpuChip.V4: 4,
    device.TpuChip.V5E: 4,
    device.TpuChip.V5P: 4,
    device.TpuChip.V6E: 4,
}


# TODO(Kelvin-zou): will revisit this function to make it more libtpu native.
def validate_available_metrics(
    metric_list: Sequence[MetricName], *, addr: str = LIB_TPU_METRICS_SERVER_ADDR
) -> bool:
    """Validate the available metrics against the supported metrics from libtpu.

    Args:
        metric_list: The metrics to be fetched from the libtpu.
        addr: GRPC server from libtpu. Defaults to LIB_TPU_METRICS_SERVER_ADDR.

    Returns:
        bool: True if all metrics are supported, False otherwise.
    """
    # Considering the low cost of opening a new grpc client for each call,
    # we don't cache the client.
    channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
    client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

    # Manually annotate type until GRPC supports annotations
    # See https://github.com/grpc/grpc/issues/29041
    resp: tpu_metrics.MetricResponse = client.ListSupportedMetrics(
        tpu_metrics.ListSupportedMetricsRequest()
    )
    # Log all supported metrics, this should be only called once.
    logging.info("Validating metrics: %s", resp.supported_metric)
    supported_metric_names = {
        supported_metric.metric_name for supported_metric in resp.supported_metric
    }
    # Validate metric_list against the supported metrics.
    is_valid = True
    for metric_name in metric_list:
        if metric_name.value not in supported_metric_names:
            logging.error("Metric %s is not supported.", metric_name.value)
            is_valid = False
    if is_valid:
        logging.info("Metrics validation passed.")
    return is_valid


# TODO(Kelvin-zou): will revisit this function to make it more libtpu native.
def get_chip_metrics(
    metric_list: Sequence[MetricName],
    *,
    chip_type: device.TpuChip,
    addr: str = LIB_TPU_METRICS_SERVER_ADDR,
) -> list[Usage]:
    """Gets usage statistics for all attached TPU devices from libtpu.

    Args:
        metric_list: List of metrics to fetch from libtpu.
        chip_type: TPU chip version. Determines how metrics are interpreted.
        addr: GRPC server address of libtpu metrics server.
            Defaults to LIB_TPU_METRICS_SERVER_ADDR.

    Returns:
        List of usage statistics for each TPU device.
    """
    # Considering the low cost of opening a new grpc client for each call, we do live query.
    # The tcp connection may be cached by a lower level
    channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
    client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

    def sorted_metric_response(
        metric_name: str,
    ) -> list[tpu_metrics.Metric]:
        # Manually annotate type until GRPC supports annotations.
        # See https://github.com/grpc/grpc/issues/29041
        resp: tpu_metrics.MetricResponse = client.GetRuntimeMetric(
            tpu_metrics.MetricRequest(metric_name=metric_name)
        )
        return sorted(resp.metric.metrics, key=lambda m: m.attribute.value.int_attr)

    metric_results = [Usage(device_id=i) for i in range(CHIPS_PER_NODE[chip_type])]
    for metric_name in metric_list:
        metric_result = sorted_metric_response(metric_name.value)

        if CHIPS_PER_NODE[chip_type] != len(metric_result):
            raise SystemError("Metrics not found for all chips, this indicates a serious issue.")

        if metric_name == MetricName.HBM_MEMORY_TOTAL_BYTES:
            for i, metric in enumerate(metric_result):
                metric_results[i].hbm_memory_total_bytes = metric.gauge.as_int
        elif metric_name == MetricName.HBM_MEMORY_USAGE_BYTES:
            for i, metric in enumerate(metric_result):
                metric_results[i].hbm_memory_usage_bytes = metric.gauge.as_int
        elif metric_name == MetricName.TENSORCORE_DUTY_CYCLE_PERCENT:
            for i, metric in enumerate(metric_result):
                metric_results[i].device_duty_cycle_percent = metric.gauge.as_double

    return metric_results


def validate_available_metrics_v2(
    metric_list: Sequence[MetricName], *, addr: str = TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR
) -> bool:
    """Validate the available metrics against the supported metrics from tpu device plugin.

    Args:
        metric_list: The metrics to be fetched from the tpu device plugin.
        addr: Address of tpu-device-plugin metrics server.
            Defaults to TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR.

    Returns:
        True if all metrics are supported, False otherwise.
    """
    # Due to no official way to list all metrics,
    # we do a live query and check if the metrics are supported.
    try:
        with urllib.request.urlopen(f"http://{addr}/metrics") as response:
            contents = response.read().decode("utf-8")
            families = list(text_string_to_metric_families(contents))
            supported_metrics = set()
            for family in families:
                if isinstance(family, Metric):
                    supported_metrics.add(family.name)
            is_valid = True
            for metric in metric_list:
                if metric.value not in supported_metrics:
                    logging.error("Metric %s is not supported.", metric.value)
                    is_valid = False
            if is_valid:
                logging.info("Supported metrics: %s", supported_metrics)
            return is_valid
    except urllib.error.URLError as e:
        logging.log_first_n(logging.ERROR, "Failed to fetch metrics from %s: %s", 5, addr, e)
        return False


def get_chip_metrics_v2(
    metric_list: Sequence[MetricV2Name],
    *,
    chip_type: device.TpuChip,
    addr: str = TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR,
) -> list[Usage]:
    """Gets usage statistics for tpu devices on the node, from tpu-device-plugin.

    Args:
        metric_list: List of metrics to fetch from tpu-device-plugin.
        chip_type: TPU chip version. Determines how metrics are interpreted.
        addr: Address of tpu-device-plugin metrics server.
            Defaults to TPU_DEVICE_PLUGIN_METRICS_SERVER_ADDR.

    Returns:
        List of usage statistics for each TPU device
    """
    devices_per_node = CHIPS_PER_NODE[chip_type]
    # Consider the low cost of opening a new connection for each call, we do live query.
    try:
        with urllib.request.urlopen(f"http://{addr}/metrics") as response:
            contents = response.read().decode("utf-8")
            families = list(text_string_to_metric_families(contents))
            metric_results = [Usage(device_id=i) for i in range(CHIPS_PER_NODE[chip_type])]
            for family in families:
                if isinstance(family, Metric) and family.name in [i.value for i in metric_list]:
                    assert len(family.samples) == devices_per_node
                    for i, metric in enumerate(family.samples):
                        if family.name == MetricV2Name.HBM_MEMORY_TOTAL_BYTES.value:
                            metric_results[i].hbm_memory_total_bytes = metric[2]
                        elif family.name == MetricV2Name.HBM_MEMORY_USAGE_BYTES.value:
                            metric_results[i].hbm_memory_usage_bytes = metric[2]
                        elif family.name == MetricV2Name.TENSORCORE_DUTY_CYCLE_PERCENT.value:
                            metric_results[i].device_duty_cycle_percent = metric[2]
                        elif family.name == MetricV2Name.TENSORCORE_UTILIZATION.value:
                            metric_results[i].device_utilization = metric[2]
                        elif family.name == MetricV2Name.HBM_MEMORY_BANDWIDTH_UTILIZATION.value:
                            metric_results[i].hbm_memory_bandwidth_utilization = metric[2]

            return metric_results

    except urllib.error.URLError as e:
        logging.log_first_n(logging.ERROR, "Failed to fetch metrics from %s: %s", 5, addr, e)
        return []
