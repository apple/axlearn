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
from contextlib import contextmanager
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


@contextmanager
def maybe_profile(enabled: bool, profile_dir: Optional[str]):
    """JAX profiler context if enabled, otherwise a no-op.

    Args:
        enabled: Whether profiling is enabled.
        profile_dir: Directory to save profiling results.
    """
    if enabled:
        assert profile_dir is not None, "profile_dir must be set when profiling is enabled"
        jax.profiler.start_trace(profile_dir)
    try:
        yield
    finally:
        if enabled:
            jax.profiler.stop_trace()


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

    # Track sharding statistics
    sharded_bytes = 0
    replicated_bytes = 0
    per_shard_bytes = 0
    num_sharded = 0
    num_replicated = 0

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

            # Compute tensor size in bytes downloaded from GCS.
            # Checkpoints are stored as fp32 (4 bytes per element).
            element_size = 4
            tensor_bytes = 1
            for d in value.shape:
                tensor_bytes *= d
            tensor_bytes *= element_size

            if partition_spec == jax.sharding.PartitionSpec():
                replicated_bytes += tensor_bytes
                num_replicated += 1
            else:
                sharded_bytes += tensor_bytes
                num_sharded += 1
                # Compute per-shard size by dividing by the number of shards
                num_shards = 1
                for axis in partition_spec:
                    if axis is not None:
                        if isinstance(axis, tuple):
                            for a in axis:
                                num_shards *= mesh.shape[a]
                        else:
                            num_shards *= mesh.shape[axis]
                per_shard_bytes += tensor_bytes // num_shards

    num_devices = len(mesh.devices.flat)
    print(f"Sharding stats ({num_devices} devices):")
    print(
        f"  Sharded:    {num_sharded} tensors, {sharded_bytes / 10**9:.2f} GB "
        f"(per-shard total: {per_shard_bytes / 10**9:.2f} GB)"
    )
    print(f"  Replicated: {num_replicated} tensors, {replicated_bytes / 10**9:.2f} GB")
    print(f"  Per-device total: {(per_shard_bytes + replicated_bytes) / 10**9:.2f} GB")

    return tensorstore_specs, shardings, global_shapes, dtypes


def load_model(
    tensorstore_specs: Sequence[Dict[str, Any]],
    shardings: Sequence[jax.sharding.NamedSharding],
    global_shapes: Sequence[tuple],
    dtypes: Sequence[jnp.dtype],
):
    """Load model from checkpoint.

    Args:
        tensorstore_specs: TensorStore specifications for each array.
        shardings: Target shardings for the restored arrays.
        global_shapes: Global shapes for each array.
        dtypes: Data types for each array.

    Returns:
        List of restored JAX arrays.
    """
    manager = GlobalAsyncCheckpointManager()
    restored_values = manager.deserialize(
        shardings=shardings,
        tensorstore_specs=tensorstore_specs,
        global_shapes=global_shapes,
        dtypes=dtypes,
        concurrent_gb=400,
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
    parser.add_argument(
        "--num_iters",
        type=int,
        default=1,
        help="Number of times to repeat the load benchmark (default: 1)",
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

    num_iterations = args.num_iters
    print(f"--- Running {args.method} benchmark ({num_iterations} iterations) ---")
    loaded_values = None
    try:
        with create_mesh():
            # Create checkpoint specs (needs mesh)
            tensorstore_specs, shardings, global_shapes, dtypes = create_checkpoint_spec_from_state(
                args.ckpt_path, state_spec
            )

            if args.method == "colocated":
                os.environ["COLOCATED_PYTHON_DESERIALIZE"] = "1"

            loaded_values = None
            with maybe_profile(args.profile, profile_dir):
                for i in range(num_iterations):
                    print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
                    start_time = time.perf_counter()
                    loaded_values = load_model(
                        tensorstore_specs=tensorstore_specs,
                        shardings=shardings,
                        global_shapes=global_shapes,
                        dtypes=dtypes,
                    )
                    elapsed = time.perf_counter() - start_time
                    print(f"✅ Successfully loaded model from {args.ckpt_path}")
                    print(f"Total time took {elapsed:.2f} seconds")
                    print(f"   Total parameters: {sum(x.size for x in loaded_values):,}")

                    # Drop reference to TPU arrays and sleep for memory observation.
                    if i < num_iterations - 1:
                        del loaded_values
                        loaded_values = None
                        time.sleep(60)
    finally:
        if loaded_values is not None:
            del loaded_values


if __name__ == "__main__":
    main()
