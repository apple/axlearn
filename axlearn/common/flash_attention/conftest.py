# Copyright © 2024 Apple Inc.

"""Configures pytest to distribute tests to multiple GPUs.

Usage on 8 GPU machines: PARALLEL_GPU_TEST=1 pytest -n 8 gpu_attention_test.py
"""
import os


def pytest_configure(_):
    if "PARALLEL_GPU_TEST" not in os.environ:
        return
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    os.environ["CUDA_VISIBLE_DEVICES"] = worker_id.lstrip("gw")
