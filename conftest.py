# Copyright © 2024 Apple Inc.

"""Configures pytest to distribute tests to multiple GPUs.

This is not enabled by default and requires explicit opt-in by setting the environment variable
`AXLEARN_CI_GPU_TESTS`. This is because not all GPU tests are single-GPU tests.
An additional environment variable `AXLEARN_CI_NUM_DEVICES_PER_WORKER` can be set to control the
number of GPUs visible to each worker.

Example usage on 8-GPU machines:
- 1 GPU per worker:
    AXLEARN_CI_GPU_TESTS=1 pytest -n 8 axlearn/common/flash_attention/gpu_attention_test.py
- 4 GPUs per worker:
    AXLEARN_CI_GPU_TESTS=1 pytest -n 32 axlearn/common/flash_attention/gpu_attention_test.py
"""
import os


# pylint: disable-next=unused-argument
def pytest_configure(config):
    if "AXLEARN_CI_GPU_TESTS" not in os.environ:
        return
    worker_idx = int(os.getenv("PYTEST_XDIST_WORKER", "gw0").lstrip("gw"))
    # Evenly distribute work to all GPUs.
    num_devices_per_worker = int(os.environ.get("AXLEARN_CI_NUM_DEVICES_PER_WORKER", "1"))
    num_devices = int(os.environ.get("AXLEARN_CI_NUM_DEVICES", "8"))
    starting_device_idx = (
        worker_idx % (num_devices // num_devices_per_worker)
    ) * num_devices_per_worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(device_idx)
        for device_idx in range(starting_device_idx, starting_device_idx + num_devices_per_worker)
    )
