#!/usr/bin/env python3
"""Standalone script to preload a model from GCS using Colocated Python.

This script reads the checkpoint index to determine the model structure and creates
appropriate TensorSpec objects for preloading.

Usage:
    # Run colocated benchmark (default, no profiling)
    python colocated_python_benchmark.py \
        --ckpt_path gs://your-bucket/path/to/checkpoint --method colocated

    # Run colocated benchmark with profiling
    python colocated_python_benchmark.py \
        --ckpt_path gs://your-bucket/path/to/checkpoint --method colocated --profile

    # Run default (direct to TPU) benchmark
    python colocated_python_benchmark.py \
        --ckpt_path gs://your-bucket/path/to/checkpoint --method default
"""

import argparse
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import pathwaysutils  # pytype: disable=import-error
from jax._src.mesh import thread_resources
from jax.experimental import mesh_utils
from jax.experimental.array_serialization import serialization as array_serialization

from axlearn.common import utils
from axlearn.common.array_serialization import GlobalAsyncCheckpointManager
from axlearn.common.checkpointer import parse_step_from_dir, read_index_file
from axlearn.common.utils import TensorSpec, infer_mesh_shape


def maybe_profile(enabled: bool, profile_dir: Optional[str]):
    """Return JAX profiler context if enabled, otherwise a no-op context manager.

    Args:
        enabled: Whether profiling is enabled.
        profile_dir: Directory to save profiling results.

    Returns:
        Context manager for profiling or no-op.
    """
    if enabled:
        assert profile_dir is not None, "profile_dir must be set when profiling is enabled"
        return jax.profiler.trace(profile_dir)
    else:
        # Return a no-op context manager
        return nullcontext()


def create_mesh(mesh_shape=(1, 1, 1, 1, 1, 16, -1)):
    """Create a JAX mesh for distributed computation."""
    inferred_mesh_shape = infer_mesh_shape(mesh_shape)
    print(f"Using mesh shape {inferred_mesh_shape} for {len(jax.devices())} devices")
    devices = mesh_utils.create_device_mesh(inferred_mesh_shape)
    return jax.sharding.Mesh(
        devices, ("pipeline", "data", "expert", "fsdp", "seq", "track", "model")
    )


def create_state_spec_from_checkpoint(ckpt_path: str):
    """Create a NestedTensorSpec from checkpoint index information."""
    index = read_index_file(ckpt_path)
    print(f"Read checkpoint index with {len(index)} entries")

    state_spec = {}

    for path, value in index:
        if path == "step":
            continue

        # Filter out learner state
        if is_learner_path(path):
            continue

        if isinstance(value, dict) and "shape" in value and "dtype" in value:
            # pylint: disable=eval-used
            shape = eval(value["shape"]) if isinstance(value["shape"], str) else value["shape"]
            dtype_str = value["dtype"]

            # Convert dtype string to jax dtype
            dtype = getattr(jnp, dtype_str, jnp.float32)
            if dtype == jnp.float32:
                dtype = jnp.bfloat16

            # Create nested dict structure from path
            keys = path.split("/")
            current = state_spec
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = TensorSpec(shape=shape, dtype=dtype)

    return state_spec


def is_learner_path(path: str) -> bool:
    """Check if a path is part of the learner state."""
    # Exclude all learner paths (optimizer state, ema, etc.)
    return path.startswith("learner/")


# pylint: disable-next=unused-argument
def get_inference_partition_spec(path: str, shape: tuple) -> jax.sharding.PartitionSpec:
    """Get inference-friendly partition spec based on tensor path and shape."""
    if "track" in path:
        return jax.sharding.PartitionSpec(None, "track", "model")

    return jax.sharding.PartitionSpec()


def create_checkpoint_spec_from_state(ckpt_dir: str, state_spec: dict):
    """Create checkpoint spec following the pattern from TensorStoreStateStorage._get_spec."""

    tensorstore_specs = []
    global_shapes = []
    dtypes = []
    shardings = []

    # Get current mesh for creating shardings
    mesh = thread_resources.env.physical_mesh
    if not mesh.shape:
        raise RuntimeError("Checkpoint restoration must take place within the context of a Mesh")

    # Process each tensor in the state spec
    for path, value in utils.flatten_items(state_spec, separator="/"):
        if isinstance(value, TensorSpec):
            # Get dtype
            dtype = getattr(value.dtype, "dtype", value.dtype)

            # Create storage path and tensorstore spec
            gda_path = os.path.join(ckpt_dir, "gda", path)
            tensorstore_spec = array_serialization.get_tensorstore_spec(gda_path)

            # Get inference-friendly partition spec based on tensor path and shape
            partition_spec = get_inference_partition_spec(path, value.shape)

            # Create sharding with the appropriate partition spec
            sharding = jax.sharding.NamedSharding(mesh, partition_spec)

            tensorstore_specs.append(tensorstore_spec)
            global_shapes.append(value.shape)
            dtypes.append(dtype)
            shardings.append(sharding)

    return tensorstore_specs, shardings, global_shapes, dtypes


def cleanup_loaded_arrays(loaded_arrays: list) -> None:
    """Clean up loaded arrays and free device memory.

    This function ensures a fair comparison between benchmarks by:
    1. Blocking until all async operations complete
    2. Deleting Python references to free device memory
    3. Clearing JAX caches
    4. Forcing device synchronization to ensure HBM cleanup completes

    Args:
        loaded_arrays: List of JAX arrays to clean up.
    """
    print("\nCleaning up arrays...")

    # Block until all arrays are ready (ensures async ops complete)
    for arr in loaded_arrays:
        arr.block_until_ready()

    loaded_arrays.clear()
    del loaded_arrays

    # Clear JAX in-memory compilation cache
    # (Persistent cache is disabled via JAX_ENABLE_COMPILATION_CACHE=0)
    jax.clear_caches()

    # Force device synchronization to ensure all HBM deallocations complete
    # This is critical - without it, deallocation may be async and incomplete
    jax.block_until_ready(jax.numpy.array(0))

    print("Cleanup complete.")


def load_model(
    tensorstore_specs: Sequence[Dict[str, Any]],
    shardings: Sequence[jax.sharding.NamedSharding],
    global_shapes: Sequence[tuple],
    dtypes: Sequence[jnp.dtype],
    use_colocated_python: bool = False,
):
    """Load model from checkpoint.

    Args:
        tensorstore_specs: TensorStore specifications for each array.
        shardings: Target shardings for the restored arrays.
        global_shapes: Global shapes for each array.
        dtypes: Data types for each array.
        use_colocated_python: If True, load to CPU first then transfer to TPU.
            If False, load directly to TPU.

    Returns:
        List of restored JAX arrays.
    """
    manager = GlobalAsyncCheckpointManager()
    restored_values = manager.deserialize(
        shardings=shardings,
        tensorstore_specs=tensorstore_specs,
        global_shapes=global_shapes,
        dtypes=dtypes,
        concurrent_gb=192,
        use_colocated_python=use_colocated_python,
    )
    print(f"Loaded {len(restored_values)} arrays")

    return restored_values


def main():
    parser = argparse.ArgumentParser(description="Preload model from GCS checkpoint")
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="GCS path to checkpoint directory (e.g., gs://bucket/path/to/checkpoint)",
    )
    parser.add_argument(
        "--method",
        choices=["colocated", "default"],
        default="colocated",
        help="Loading method to benchmark: 'colocated' (CPU preload) or 'default' (direct to TPU)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX profiler (adds overhead, disable for accurate benchmarking)",
    )
    args = parser.parse_args()

    # Disable persistent compilation cache for fair benchmarking
    # This ensures benchmarks compile fresh and don't benefit from cached kernels
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"

    if os.getenv("JAX_PLATFORMS") == "proxy":
        pathwaysutils.initialize()
    else:
        jax.distributed.initialize()

    print(f"JAX devices: {jax.devices()}")

    # Validate checkpoint path
    if not args.ckpt_path.startswith("gs://"):
        raise ValueError(f"Only GCS paths (gs://) are supported, got: {args.ckpt_path}")
    profile_dir = None
    if args.profile:
        # Create timestamped profile directory (minute-level granularity)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        base_path = args.ckpt_path.split("/checkpoints")[0]
        profile_dir = f"{base_path}/profiles/{args.method}_{timestamp}/"
        print(f"Profiling enabled - results will be saved to {profile_dir}")

    step = parse_step_from_dir(args.ckpt_path)
    print(f"Starting model preload from: {args.ckpt_path} (step {step})")

    # Read checkpoint structure (doesn't need mesh)
    print("Reading checkpoint structure...")
    state_spec = create_state_spec_from_checkpoint(args.ckpt_path)
    print(f"Found {len(jax.tree_util.tree_leaves(state_spec))} tensors in checkpoint")

    print(f"--- Running {args.method} benchmark ---")
    loaded_values = None
    try:
        with create_mesh():
            # Create checkpoint specs (needs mesh)
            tensorstore_specs, shardings, global_shapes, dtypes = create_checkpoint_spec_from_state(
                args.ckpt_path, state_spec
            )

            with maybe_profile(args.profile, profile_dir):
                start_time = time.perf_counter()
                loaded_values = load_model(
                    tensorstore_specs=tensorstore_specs,
                    shardings=shardings,
                    global_shapes=global_shapes,
                    dtypes=dtypes,
                    use_colocated_python=(args.method == "colocated"),
                )
                print(f"✅ Successfully loaded model from {args.ckpt_path}")
                print(f"Total time took {time.perf_counter() - start_time:.2f} seconds")
                print(f"   Total parameters: {sum(x.size for x in loaded_values):,}")
    finally:
        # Always clean up, even if benchmark fails
        if loaded_values is not None:
            cleanup_loaded_arrays(loaded_values)


if __name__ == "__main__":
    main()
