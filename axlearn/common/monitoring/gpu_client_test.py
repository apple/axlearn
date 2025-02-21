# Copyright © 2025 Apple Inc.

"""Tests for gpu_client.py"""

import atexit
import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import jax
from absl.testing import parameterized

from axlearn.common.monitoring import gpu_client


@dataclasses.dataclass
class MockUtilizationSampleValue:
    uiVal: int  # pylint: disable=invalid-name


@dataclasses.dataclass
class MockUtilizationSample:
    sampleValue: MockUtilizationSampleValue  # pylint: disable=invalid-name


@unittest.skipIf(jax.default_backend() != "gpu", "Skip due to not a gpu backend")
class TestGPUMetrics(parameterized.TestCase):
    """
    Test class for GPU Client methods
    """

    def setUp(self):
        self.nvml_mock_patch = mock.patch("axlearn.common.monitoring.gpu_client.nvml")
        self.nvml_mock = self.nvml_mock_patch.start()

        gpu_client.NVMLMetrics.nvml_initialized = False

    def tearDown(self):
        self.nvml_mock_patch.stop()

    def test_init_nvml(self):
        gpu_client.NVMLMetrics.init_nvml()
        self.nvml_mock.nvmlInit.assert_called_once()

        # call again, and ensure only one call was made
        gpu_client.NVMLMetrics.init_nvml()
        self.nvml_mock.nvmlInit.assert_called_once()

        # simulate exit
        atexit._run_exitfuncs()  # pylint: disable=protected-access
        self.nvml_mock.nvmlShutdown.assert_called_once()

    def test_init_nvml_error(self):
        self.nvml_mock.nvmlInit.side_effect = [Exception]

        with self.assertRaises(Exception):
            gpu_client.NVMLMetrics.init_nvml()

        self.nvml_mock.nvmlInit.assert_called_once()
        # simulate exit and assert that shutdown was not called
        atexit._run_exitfuncs()  # pylint: disable=protected-access
        self.nvml_mock.nvmlShutdown.assert_not_called()

    def test_get_gpu_device_count(self):
        self.nvml_mock.nvmlDeviceGetCount.return_value = 8

        res = gpu_client.NVMLMetrics.get_gpu_device_count()

        self.nvml_mock.nvmlInit.assert_called_once()
        self.nvml_mock.nvmlDeviceGetCount.assert_called_once()

        self.assertEqual(res, 8)

    def test_get_gpu_device_utilization(self):
        self.nvml_mock.nvmlDeviceGetSamples.return_value = (
            0,
            [
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=10)),
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=20)),
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=30)),
            ],
        )
        self.nvml_mock.NVML_GPU_UTILIZATION_SAMPLES = "util_samples"
        device_handle_mock = mock.MagicMock()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.return_value = device_handle_mock

        res = gpu_client.NVMLMetrics.get_gpu_device_utilization(device_id=0)

        self.nvml_mock.nvmlInit.assert_called_once()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
        self.nvml_mock.nvmlDeviceGetSamples.assert_called_once_with(
            device_handle_mock, "util_samples", 0
        )

        self.assertEqual(res, 20.0)

    def test_get_gpu_device_memory(self):
        self.nvml_mock.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(
            used=100.0, total=200.0
        )
        device_handle_mock = mock.MagicMock()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.return_value = device_handle_mock

        used, total = gpu_client.NVMLMetrics.get_gpu_device_memory(device_id=0)

        self.nvml_mock.nvmlInit.assert_called_once()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
        self.nvml_mock.nvmlDeviceGetMemoryInfo.assert_called_once_with(device_handle_mock)

        self.assertEqual(used, 100.0)
        self.assertEqual(total, 200.0)

    def test_get_gpu_device_memory_utilization(self):
        self.nvml_mock.nvmlDeviceGetSamples.return_value = (
            0,
            [
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=10)),
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=20)),
                MockUtilizationSample(sampleValue=MockUtilizationSampleValue(uiVal=30)),
            ],
        )
        self.nvml_mock.NVML_MEMORY_UTILIZATION_SAMPLES = "mem_samples"
        device_handle_mock = mock.MagicMock()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.return_value = device_handle_mock

        res = gpu_client.NVMLMetrics.get_gpu_device_memory_utilization(device_id=0)

        self.nvml_mock.nvmlInit.assert_called_once()
        self.nvml_mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
        self.nvml_mock.nvmlDeviceGetSamples.assert_called_once_with(
            device_handle_mock, "mem_samples", 0
        )

        self.assertEqual(res, 20.0)
