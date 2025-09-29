#!/usr/bin/env python3
"""
Standalone script to preload a model from GCS using Colocated Python.

This script reads the checkpoint index to determine the model structure and creates
appropriate TensorSpec objects for preloading.

Usage:
    python load_model_colocated.py --ckpt_path gs://your-bucket/path/to/checkpoint
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

def create_mesh(mesh_shape=(1, 1, 1, 1, 1, -1)):
    """Create a JAX mesh for distributed computation."""
    inferred_mesh_shape = infer_mesh_shape(mesh_shape)
    print(f"Using mesh shape {inferred_mesh_shape} for {len(jax.local_devices())} devices")
    devices = mesh_utils.create_device_mesh(inferred_mesh_shape)
    return jax.sharding.Mesh(devices, ("pipeline", "data", "expert", "fsdp", "seq", "model"))

def is_learner_path(path: str) -> bool:
    """Check if a path is part of the learner state."""
    # Exclude all learner paths (optimizer state, ema, etc.)
    return path.startswith('learner/')

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
    print("thread resources mesh: ",mesh)

    devices = jax.experimental.mesh_utils.create_device_mesh((4,1),devices=jax.local_devices())
    mesh_created = jax.sharding.Mesh(devices, axis_names=("data","model"))
    print("mesh_created: ",mesh_created)


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
            sharding = jax.sharding.NamedSharding(mesh_created, jax.sharding.PartitionSpec('data','model'))#partition_spec)
            print("Is addressable?", sharding.is_fully_addressable)
            print("Sharding devices:", sharding.device_set)
            print("Addressable subset:", sharding.addressable_devices)
            
            tensorstore_specs.append(tensorstore_spec)
            shapes.append(value.shape)
            dtypes.append(dtype)
            shardings.append(sharding)
    
    return tensorstore_specs, shardings, shapes, dtypes


def main():
    ### Parse Arguments ######
    parser = argparse.ArgumentParser(description="Preload model from GCS checkpoint")
    parser.add_argument("--ckpt_path", required=True,
                       help="GCS path to checkpoint directory (e.g., gs://bucket/path/to/checkpoint)")
    parser.add_argument("--profile_dir", required=True,
                       help="GCS path to profile directory (e.g., gs://bucket/path/to/checkpoint)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    ##### Getting the step from checkpointer #####
    step = parse_step_from_dir(args.ckpt_path)
    print(f"Starting model preload from: {args.ckpt_path} (step {step})")
    
    if not args.ckpt_path.startswith("gs://"):
        raise ValueError(f"Only GCS paths (gs://) are supported, got: {args.ckpt_path}")
    
    #### colocated code  #######
    devices = jax.devices()
    print(len(devices))
    cpu_devices = colocated_python.colocated_cpu_devices(devices)
    print(cpu_devices)
    local_devices=jax.local_devices()
    num_devices=len(local_devices)
    print("local devices: ",len(local_devices))

    with create_mesh():
        jax.profiler.start_trace(args.profile_dir)
        print("Reading checkpoint structure...")
        state_spec = create_state_spec_from_checkpoint(args.ckpt_path)
        
        print(f"Found {len(jax.tree_util.tree_leaves(state_spec))} tensors in checkpoint")
        
        tensorstore_specs, shardings, global_shapes, dtypes = create_checkpoint_spec_from_state(args.ckpt_path, state_spec)
        
        print("Preloading checkpoint to CPU memory...")
        if len(cpu_devices) > 1:
            cpu_mesh = colocated_python.colocated_cpu_devices(thread_resources.env.physical_mesh)
            cpu_shardings = [
                jax.sharding.NamedSharding(cpu_mesh, sharding.spec) for sharding in shardings
            ]
        else:
            cpu_shardings = [
                jax.sharding.SingleDeviceSharding(cpu_devices[0]) for sharding in shardings
            ]
        
        def output_spec_fn():
            return [
                jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)
                for shape, dtype, sharding in zip(global_shapes, dtypes, cpu_shardings)
            ]

        @colocated_python.colocated_python
        def run_deserializer():
        # Object should be created once per process.
        # pylint: disable=protected-access
            concurrent_bytes = 1099511627776
            byte_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
            h2d_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
            thread_pool = ThreadPoolExecutor(10)

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
            )
            async def gather_func():
                return await asyncio.gather(*future_arrays)
            result = asyncio.run(gather_func())
            return result


        
        run_deserializer = run_deserializer.specialize(
                devices=cpu_devices,
                out_specs_fn=output_spec_fn,)
                
        start_time = time.perf_counter()

        preloaded_values = run_deserializer()
        preload_time = time.perf_counter() - start_time
        print(f"Preload completed to CPU in {preload_time:.2f} seconds")
        print(f"Preloaded {len(preloaded_values)} arrays")

        total_size = sum(a.size for a in preloaded_values)
        print("total_size of preloaded: ",total_size)

        #### calculate size of the checkpoint ####
        total_gb = (total_size * preloaded_values[0].dtype.itemsize)/ (1024**3)
        print("total_gb: ",total_gb)

        print("shardings length",len(shardings))

        print("Transferring arrays to TPU...")
        restored_values = []
        
        start_time = time.perf_counter()
        # zip(preloaded_values, shardings)

        for i in range(4):
            arr=jax.device_put(preloaded_values[i], shardings[i])
            restored_values.append(arr)
        
        for device_array in restored_values:
            device_array.block_until_ready()
        
        transfer_time = time.perf_counter() - start_time
        print(f"Transfer completed to TPU in {transfer_time:.2f} seconds")

        jax.profiler.stop_trace()

        # total_data_moved_gb = data_gb_per_device * 4
        # throughput_gb_s = total_data_moved_gb / transfer_time

        # print(f"Data per device: {data_gb_per_device:.2f} GiB")
        # print(
        #     "Total data transferred from host per operation:"
        #     f" {total_data_moved_gb:.2f} GiB"
        # )
        # print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s:.2f} GiB/s")

                
if __name__ == "__main__":
    main()