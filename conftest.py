# Copyright © 2024 Apple Inc.

"""Configures pytest to distribute tests to multiple GPUs.

This is not enabled by default and requires explicit opt-in by setting the environment variable
PARALLEL_GPU_TEST. This is because not all GPU tests are single-GPU tests.

Example usage on 8 GPU machines:
PARALLEL_GPU_TEST=1 pytest -n 8 axlearn/common/flash_attention/gpu_attention_test.py
"""
import os


# pylint: disable-next=unused-argument
def pytest_configure(config):
    if "PARALLEL_GPU_TEST" not in os.environ:
        return
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    num_gpus = int(os.getenv("NUM_GPUS", "8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(worker_id.lstrip("gw")) % num_gpus)
