# Copyright © 2025 Apple Inc.

"""Utilities for loading golden test data from .npz files."""

import os

import numpy as np

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "../experiments/testdata")


def load_golden(module_name: str, test_name: str) -> dict:
    """Load golden data for a specific test from the module-level .npz file.

    Args:
        module_name: Dotted module path, e.g. "axlearn.common.causal_lm_test".
        test_name: Test method name, e.g. "test_against_hf_gpt2_lm".

    Returns:
        Nested dict with keys like "params", "inputs", "outputs", each containing
        sub-dicts of numpy arrays.
    """
    path = os.path.join(_TESTDATA_DIR, f"{module_name}.npz")
    data = np.load(path)
    prefix = f"{test_name}/"
    tree = {}
    for key in data.files:
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = data[key]
    return tree
