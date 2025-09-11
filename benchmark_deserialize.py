"""
A script to benchmark the GlobalAsyncCheckpointManager.deserialize function.

This script contains a local patch for the deserialization logic to work around
a bug in the installed axlearn library, avoiding any modification to the library itself.
"""

import asyncio
import functools
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from absl import app, flags
from jax._src import array, typing
from jax._src.layout import Layout
from jax.experimental.array_serialization import serialization
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from axlearn.common.array_serialization import (
    GlobalAsyncCheckpointManager,
    _get_premapped_buffer_size,
)
from axlearn.common.checkpointer import read_state_spec
from axlearn.common.utils import flatten_items

# JAX platforms might be initialized by another process.
# We follow the logic in axlearn.common.launch to initialize JAX.
if os.environ.get("JAX_PLATFORMS", "") == "proxy":
    import pathwaysutils  # type: ignore

    pathwaysutils.initialize()
else:
    jax.distributed.initialize()


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    "gs://cloud-tpu-multipod-dev-axlearn/stoelinga-v7-70b-17/checkpoints/step_00000100/",
    "The GCS path to the checkpoint step directory.",
)
flags.DEFINE_integer("num_iterations", 5, "The number of benchmark iterations.")
flags.DEFINE_integer("warmup_iterations", 1, "The number of warmup iterations.")


# --- Local Patch for Deserialization ---
# The following functions are copied from axlearn.common.array_serialization
# and patched locally to fix a TypeError without modifying the library.


def _blocking_device_put(out: np.ndarray, layout: Layout) -> jax.Array:
    return jax.block_until_ready(jax.device_put(out, layout))


async def _patched_async_deserialize(
    user_in_sharding: jax.sharding.Sharding | Layout,
    tensorstore_spec: dict[str, Any],
    global_shape: Optional[Sequence[int]],
    dtype: Optional[typing.DTypeLike],
    *,
    h2d_limiter: serialization._LimitInFlightBytes,
    byte_limiter: serialization._LimitInFlightBytes,
    single_thread_pool: ThreadPoolExecutor,
):
    """Patched version of _async_deserialize."""
    in_sharding = (
        user_in_sharding.sharding if isinstance(user_in_sharding, Layout) else user_in_sharding
    )
    if not isinstance(in_sharding, jax.sharding.Sharding):
        raise ValueError(
            "sharding passed to deserialization should be specified, concrete and"
            f" an instance of `jax.sharding.Sharding`. Got {in_sharding}"
        )
    dll = user_in_sharding.device_local_layout if isinstance(user_in_sharding, Layout) else None
    t = await ts.open(
        tensorstore_spec,
        open=True,
        assume_metadata=False,
        context=serialization.TS_CONTEXT,
    )
    shape = tuple(t.shape if global_shape is None else global_shape)
    new_shard_shape = in_sharding.shard_shape(shape)
    loop = asyncio.get_running_loop()

    async def cb(index: array.Index, device: jax.Device):
        requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
        restricted_domain = t.domain.intersect(requested_domain)
        requested_bytes = serialization.estimate_read_memory_footprint(t, restricted_domain)
        await byte_limiter.wait_for_bytes(requested_bytes)
        read_ts = t[restricted_domain]
        if dtype is not None:
            read_ts = ts.cast(read_ts, dtype)
        if tuple(t.shape) == shape:
            out = np.empty(new_shard_shape, read_ts.dtype.numpy_dtype)
        else:
            out = np.zeros(new_shard_shape, read_ts.dtype.numpy_dtype)

        await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(
            read_ts
        )

        if out.dtype == jnp.int4:
            out = jnp.asarray(out)

        out_size = out.size * out.dtype.itemsize
        mb_256 = 256 * 1024 * 1024
        out_size = math.ceil(out_size / mb_256) * mb_256

        sharding_for_put = jax.sharding.SingleDeviceSharding(
            device, memory_kind=in_sharding.memory_kind
        )
        if dll is not None:
            sharding_for_put = Layout(dll, sharding_for_put)

        try:
            await h2d_limiter.wait_for_bytes(out_size)
            result = await loop.run_in_executor(None, _blocking_device_put, out, sharding_for_put)
            await h2d_limiter.release_bytes(out_size)
        except ValueError as e:
            if "Requested more bytes than we reserved" not in str(e):
                raise e
            result = await loop.run_in_executor(
                single_thread_pool, _blocking_device_put, out, sharding_for_put
            )

        await byte_limiter.release_bytes(requested_bytes)
        return result

    # This is the patched line.
    # pylint: disable-next=protected-access
    return await serialization.create_async_array_from_callback(shape, in_sharding, cb)


class PatchedGlobalAsyncCheckpointManager(GlobalAsyncCheckpointManager):
    """An override of the manager to use our patched deserialize logic."""

    def deserialize(
        self,
        shardings: Sequence[Union[jax.sharding.Sharding, Layout]],
        tensorstore_specs: Sequence[dict[str, Any]],
        global_shapes: Optional[Sequence[array.Shape]] = None,
        dtypes: Optional[Sequence[typing.DTypeLike]] = None,
        concurrent_gb: int = 32,
    ):
        self.wait_until_finished()
        concurrent_bytes = concurrent_gb * 10**9

        async def _run_deserializer():
            # pylint: disable=protected-access
            byte_limiter = serialization._LimitInFlightBytes(concurrent_bytes)
            h2d_limiter = serialization._LimitInFlightBytes(_get_premapped_buffer_size())
            future_arrays = jax.tree.map(
                functools.partial(
                    _patched_async_deserialize,  # Use our patched function.
                    byte_limiter=byte_limiter,
                    h2d_limiter=h2d_limiter,
                    single_thread_pool=self._single_thread_pool,
                ),
                shardings,
                tensorstore_specs,
                [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
                [None] * len(tensorstore_specs) if dtypes is None else dtypes,
            )
            return await asyncio.gather(*future_arrays)

        fut = asyncio.run_coroutine_threadsafe(_run_deserializer(), self._loop)
        return fut.result()


def main(argv: Sequence[str]) -> None:
    """Benchmarks the deserialize function."""
    del argv

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("data",))

    state_spec = read_state_spec(FLAGS.checkpoint_dir)
    flat_state_spec = flatten_items(state_spec, separator="/")

    ts_specs, shardings_list, global_shapes, dtypes = [], [], [], []

    for path, spec in flat_state_spec:
        gda_path = os.path.join(FLAGS.checkpoint_dir, "gda", path)
        ts_specs.append(get_tensorstore_spec(gda_path))

        partition_spec = PartitionSpec()
        if len(spec.shape) > 0 and spec.shape[0] % len(devices) == 0:
            partition_spec = PartitionSpec("data", *(None,) * (len(spec.shape) - 1))

        shardings_list.append(NamedSharding(mesh, partition_spec))
        global_shapes.append(spec.shape)
        dtypes.append(spec.dtype)

    manager = PatchedGlobalAsyncCheckpointManager()

    def run_deserialize():
        """Runs deserialization across all tensors."""
        start_time = time.time()
        restored_arrays = manager.deserialize(
            shardings=shardings_list,
            tensorstore_specs=ts_specs,
            global_shapes=global_shapes,
            dtypes=dtypes,
        )
        for arr in restored_arrays:
            arr.block_until_ready()
        return time.time() - start_time

    print(f"Running {FLAGS.warmup_iterations} warmup iterations...")
    for _ in range(FLAGS.warmup_iterations):
        run_deserialize()

    print(f"Running {FLAGS.num_iterations} benchmark iterations...")
    durations = []
    for i in range(FLAGS.num_iterations):
        duration = run_deserialize()
        print(f"Iteration {i+1} took {duration:.4f} seconds.")
        durations.append(duration)

    print("\n--- Benchmark Results ---")
    print(f"Number of devices: {len(devices)}")
    print(f"Iterations: {FLAGS.num_iterations}")
    print(f"Average time: {sum(durations) / len(durations):.4f} seconds")
    print(f"Min time: {min(durations):.4f} seconds")
    print(f"Max time: {max(durations):.4f} seconds")
    print("-------------------------\n")

    manager.stop()


if __name__ == "__main__":
    app.run(main)
