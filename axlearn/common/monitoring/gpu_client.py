# Copyright © 2025 Apple Inc.

"""Client for fetching GPU metrics via NVML."""
import atexit

from absl import logging


class NVMLMetrics:
    """NVMLMetrics provides interfaces to fetch GPU utilization/memory metrics via NVML.

    Calling `pynvml.nvmlInit` multiple times will lead to potential issues and it should only
    be called once.

    And when the operations are completed, `pynvml.nvmlShutdown` should be called. Currently it is
    called using `atexit`.
    """

    nvml_initialized = False
    nvml = None

    @classmethod
    def init_nvml(cls):
        """It is not thread-safe. Please see the docstring of the class for more details.

        Users should not call `init_nvml` multiple times.
        """
        # pylint: disable-next=import-error,import-outside-toplevel
        import pynvml as nvml  # pytype: disable=import-error

        cls.nvml = nvml
        if not cls.nvml_initialized:
            try:
                nvml.nvmlInit()
            except:
                logging.exception("Failed to initialize NVML Library for GPU metrics monitoring.")
                raise
            else:
                cls.nvml_initialized = True
                atexit.register(nvml.nvmlShutdown)

    @classmethod
    def get_gpu_device_count(cls):
        cls.init_nvml()

        try:
            return cls.nvml.nvmlDeviceGetCount()
        except:
            logging.exception("Failed to get GPU device count.")
            raise

    @classmethod
    def get_gpu_device_utilization(cls, device_id: int) -> float:
        cls.init_nvml()

        # pylint: disable-next=import-error,import-outside-toplevel
        from pynvml import NVMLError  # pytype: disable=import-error

        try:
            device_handle = cls.nvml.nvmlDeviceGetHandleByIndex(device_id)

            # Get all the utilization samples in the device buffer.
            # Typically this covers about 10-13 seconds of data.
            # Reference: https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
            # Search for nvmlDeviceGetSamples.
            samples = cls.nvml.nvmlDeviceGetSamples(
                device_handle, cls.nvml.NVML_GPU_UTILIZATION_SAMPLES, 0
            )
            util_samples = [sample.sampleValue.uiVal for sample in samples[1]]
            if not util_samples:
                logging.warning("No samples returned from pynvml.")
                return 0
            average_utilization = sum(util_samples) / len(util_samples)
            return average_utilization
        except NVMLError as e:
            logging.exception("Failed to get GPU utilization metrics for device %d.", device_id)
            logging.exception(e)
            raise

    @classmethod
    def get_gpu_device_memory(cls, device_id: int) -> tuple[float, float]:
        cls.init_nvml()

        # pylint: disable-next=import-error,import-outside-toplevel
        from pynvml import NVMLError  # pytype: disable=import-error

        try:
            device_handle = cls.nvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = cls.nvml.nvmlDeviceGetMemoryInfo(device_handle)

            # Return tuple for memory usage, and total (in Bytes).
            return mem_info.used, mem_info.total
        except NVMLError as e:
            logging.exception("Failed to get GPU memory info for device %d.", device_id)
            logging.exception(e)
            raise

    @classmethod
    def get_gpu_device_memory_utilization(cls, device_id: int) -> float:
        cls.init_nvml()

        # pylint: disable-next=import-error,import-outside-toplevel
        from pynvml import NVMLError  # pytype: disable=import-error

        try:
            device_handle = cls.nvml.nvmlDeviceGetHandleByIndex(device_id)

            # Get all the utilization samples in the device buffer.
            # Typically this covers about 10-13 seconds of data.
            samples = cls.nvml.nvmlDeviceGetSamples(
                device_handle, cls.nvml.NVML_MEMORY_UTILIZATION_SAMPLES, 0
            )
            util_samples = [sample.sampleValue.uiVal for sample in samples[1]]
            average_utilization = sum(util_samples) / len(util_samples)
            return average_utilization
        except NVMLError as e:
            logging.exception("Failed to get GPU utilization metrics for device %d.", device_id)
            logging.exception(e)
            raise
