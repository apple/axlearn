#!/usr/bin/env python3
"""
Standalone script to preload a model from GCS using Colocated Python.

This script reads the checkpoint index to determine the model structure and creates
appropriate TensorSpec objects for preloading.

Usage:
    python preload_model_script.py --ckpt_path gs://your-bucket/path/to/checkpoint
"""

import argparse
import asyncio
import functools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from axlearn.common import utils
from axlearn.common.array_serialization import _async_deserialize
from axlearn.common.checkpointer import parse_step_from_dir, read_index_file
from axlearn.common.utils import infer_mesh_shape, TensorSpec
from jax._src import array, typing
from jax._src.mesh import thread_resources
from jax.experimental import colocated_python
from jax.experimental.array_serialization import serialization as array_serialization, tensorstore_impl
from jax.experimental import mesh_utils
import pathwaysutils

# Patch for jax 0.6.2
array_serialization.create_async_array_from_callback = tensorstore_impl._create_async_array_from_callback
array_serialization.estimate_read_memory_footprint = tensorstore_impl.estimate_read_memory_footprint

# Add ajax to Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@colocated_python.colocated_python
def run_deserializer():
    # Object should be created once per process.
    # pylint: disable=protected-access
    concurrent_bytes = 1099511627776
    sys.stderr.write("colocated python run_deserializer")
    start_time = time.time()
    byte_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
    h2d_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
    thread_pool = ThreadPoolExecutor(1)

    future_arrays = jax.tree.map(
        functools.partial(
            _async_deserialize,
            byte_limiter=byte_limiter,
            h2d_limiter=h2d_limiter,
            single_thread_pool=thread_pool,
        ),
        cpu_shardings,
        tensorstore_specs,
        global_shapes,
        dtypes,
    ).block_until_ready()
    async def gather_func():
        sys.stderr.write("gather_func")
        return await asyncio.gather(*future_arrays)
    result = asyncio.run(gather_func())
    deserialize_time = time.time() - start_time
    sys.stderr.write(f"Deserialize completed in {deserialize_time:.2f} seconds")
    return result

def _colocated_deserialize(
    shardings: Sequence[jax.sharding.NamedSharding],
    tensorstore_specs: Sequence[dict[str, Any]],
    global_shapes: Sequence[array.Shape],
    dtypes: Sequence[typing.DTypeLike],
):
    concurrent_bytes = 1099511627776
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    
    if len(cpu_devices) > 1:
        cpu_mesh = colocated_python.colocated_cpu_devices(thread_resources.env.physical_mesh)
        cpu_shardings = [
            jax.sharding.NamedSharding(cpu_mesh, sharding.spec) for sharding in shardings
        ]
    else:
        cpu_shardings = [
            jax.sharding.SingleDeviceSharding(cpu_devices[0]) for sharding in shardings
        ]
    print("In  _colocated_deserialize")
    print(cpu_shardings)

    def output_spec_fn():
        print("output_spec_fn")
        return [
            jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)
            for shape, dtype, sharding in zip(global_shapes, dtypes, cpu_shardings)
        ]
    
    run_deserializer1 = run_deserializer.specialize(
        devices=cpu_devices,
        out_specs_fn=output_spec_fn,
    )
    
    # Try running in the current event loop if one exists, otherwise create new one
    result = run_deserializer1()
    return result


def create_mesh(mesh_shape=(1, 1, 1, 1, 1, -1)):
    """Create a JAX mesh for distributed computation."""
    inferred_mesh_shape = infer_mesh_shape(mesh_shape)
    print(f"Using mesh shape {inferred_mesh_shape} for {len(jax.local_devices())} devices")
    devices = mesh_utils.create_device_mesh(inferred_mesh_shape)
    return jax.sharding.Mesh(devices, ("pipeline", "data", "expert", "fsdp", "seq", "model"))


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
        
        if isinstance(value, dict) and 'shape' in value and 'dtype' in value:
            shape = eval(value['shape']) if isinstance(value['shape'], str) else value['shape']
            dtype_str = value['dtype']
            
            # Convert dtype string to jax dtype
            dtype = getattr(jnp, dtype_str, jnp.float32)
            if dtype == jnp.float32:
                dtype = jnp.bfloat16
            
            # Create nested dict structure from path
            keys = path.split('/')
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
    return path.startswith('learner/')


def get_inference_partition_spec(path: str, shape: tuple) -> jax.sharding.PartitionSpec:
    """Get inference-friendly partition spec based on tensor path and shape.
    
    Based on set_inference_partition_spec function logic:
    - Attention weights: shard on fsdp and model axes
    - Feed-forward linear1 weights: shard on (fsdp, model)
    - Feed-forward linear2 weights: shard on (model, fsdp)
    - Other parameters: shard on fsdp axis only
    """
    fsdp_axis = "fsdp"
    tp_axis = "model"
    
    # Check if this is an attention weight
    if any(attn_key in path for attn_key in ["self_attention", "cross_attention"]):
        if any(weight_key in path for weight_key in ["i_proj", "o_proj", "q_proj", "k_proj", "v_proj"]):
            # Attention projection weights: shard on (fsdp, model)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(fsdp_axis, tp_axis)
    
    # Check if this is a feed-forward layer weight
    elif "feed_forward" in path:
        if "linear1" in path and "weight" in path:
            # Feed-forward linear1 weights: shard on (fsdp, model)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(fsdp_axis, tp_axis)
        elif "linear2" in path and "weight" in path:
            # Feed-forward linear2 weights: shard on (model, fsdp)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(tp_axis, fsdp_axis)
    
    # For other parameters (embeddings, layer norms, etc.), shard on fsdp axis
    if len(shape) >= 1:
        return jax.sharding.PartitionSpec(fsdp_axis)
    
    # For scalars or unknown cases, no sharding
    return jax.sharding.PartitionSpec()


def create_checkpoint_spec_from_state(ckpt_dir: str, state_spec: dict):
    """Create checkpoint spec following the pattern from TensorStoreStateStorage._get_spec."""
    
    tensorstore_specs = []
    shapes = []
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
            shapes.append(value.shape)
            dtypes.append(dtype)
            shardings.append(sharding)
    
    return tensorstore_specs, shardings, shapes, dtypes


def preload_model(ckpt_path: str):
    """Main function to preload a model from GCS checkpoint."""
    step = parse_step_from_dir(ckpt_path)
    print(f"Starting model preload from: {ckpt_path} (step {step})")
    
    if not ckpt_path.startswith("gs://"):
        raise ValueError(f"Only GCS paths (gs://) are supported, got: {ckpt_path}")
    
    with create_mesh():
        print("Reading checkpoint structure...")
        state_spec = create_state_spec_from_checkpoint(ckpt_path)
        
        print(f"Found {len(jax.tree_util.tree_leaves(state_spec))} tensors in checkpoint")
        
        tensorstore_specs, shardings, shapes, dtypes = create_checkpoint_spec_from_state(ckpt_path, state_spec)
        
        print("Preloading checkpoint to CPU memory...")
        start_time = time.perf_counter()
        
        preloaded_values = _colocated_deserialize(
            shardings=shardings,
            tensorstore_specs=tensorstore_specs,
            global_shapes=shapes,
            dtypes=dtypes,
        )
        
        preload_time = time.perf_counter() - start_time
        print(f"Preload completed in {preload_time:.2f} seconds")
        print(f"Preloaded {len(preloaded_values)} arrays")

        return preloaded_values


def main():

    parser = argparse.ArgumentParser(description="Preload model from GCS checkpoint")
    parser.add_argument("--ckpt_path", required=True,
                       help="GCS path to checkpoint directory (e.g., gs://bucket/path/to/checkpoint)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    #print(f"JAX devices: {jax.devices()}")
    devices=jax.devices()
    preloaded_values = preload_model(ckpt_path=args.ckpt_path)
    
    print(f"✅ Successfully preloaded model from {args.ckpt_path}")
    print(f"   Total parameters: {sum(x.size for x in preloaded_values):,}")
    z=jax.device_put(preloaded_values[0],devices[0])
    z.block_until_ready()
    #print("device put time:", time.time()-start_time)
    
    


if __name__ == "__main__":
    print("initializing pathwaysutils")
    pathwaysutils.initialize()
    print("pathwaysutils initialized")
    main()