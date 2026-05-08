# Copyright © 2025 Apple Inc.

"""Shared utilities for golden file generation."""

import os
from typing import Any

import numpy as np
import torch

_output_dir: str | None = None


def set_output_dir(path: str):
    """Set the output directory for golden files."""
    global _output_dir
    _output_dir = path


def _get_testdata_dir() -> str:
    if _output_dir is not None:
        return _output_dir
    # Fallback: relative to this file (works outside Bazel).
    return os.path.join(os.path.dirname(__file__), "..")


def setup_determinism(torch_seed: int = 0):
    """Set seeds for reproducible golden file generation."""
    np.random.seed(0)
    torch.manual_seed(torch_seed)
    torch.use_deterministic_algorithms(True)


def to_numpy_tree(tree: Any) -> Any:
    """Recursively convert jax/torch tensors to np.ndarray for pickling."""
    if isinstance(tree, dict):
        return {k: to_numpy_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(to_numpy_tree(v) for v in tree)
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    if hasattr(tree, "__jax_array__") or type(tree).__name__ == "ArrayImpl":
        return np.asarray(tree)
    if isinstance(tree, np.ndarray):
        return tree
    return tree


def save_golden(module_name: str, test_name: str, data: dict):
    """Save golden data as .npy file.

    Args:
        module_name: Dotted module path, e.g. "axlearn.common.bert_test".
        test_name: Test method name, e.g. "test_for_mlm".
        data: Dict with "params", "inputs", "outputs" keys.
    """
    testdata_dir = _get_testdata_dir()
    out_dir = os.path.join(testdata_dir, module_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{test_name}.npy")
    np.save(path, to_numpy_tree(data), allow_pickle=True)
    print(f"  Saved: {path}")


def golden_path(module_name: str, test_name: str) -> str:
    """Return the path where a golden file would be saved."""
    return os.path.join(_get_testdata_dir(), module_name, f"{test_name}.npy")
