# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/jax:
# Copyright 2021 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/orbax:
# Copyright [2024] The Orbax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Array serialization utilities.

Reference:
https://github.com/google/orbax/blob/3cc343c63c769e4b2df44f3e57f6b5b43569df32/checkpoint/orbax/checkpoint/serialization.py
https://github.com/google/jax/blob/595a620804e810335a870e93975a78504b2e95e5/jax/experimental/array_serialization/serialization.py
"""
import asyncio
import functools
import math
import os
import threading
import time
from collections import defaultdict
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from absl import logging
from jax._src import array, typing
from jax._src.layout import Layout
from jax.experimental.array_serialization import serialization

from axlearn.common.utils import Tensor


@dataclass
class _ShardInfo:
    """Stores information for a maybe sliced jax.Shard.

    Attributes:
        data: The actual data of the shard.
        index: The index of the shard.
        slice_arg: Arguments for `lax.slice_in_dim` in the form of (start_idx, limit_idx, axis).
            If `None`, it indicates that this shard doesn't need to be sliced.
        replica_count: The replication count for this shard.
    """

    data: Tensor
    index: tuple[slice, ...]
    slice_arg: Optional[tuple[int, int, int]]
    replica_count: int

    def shard_coordinate(self):
        """Gets the shard coordinate according to the zarr format used by tensorstore."""
        coords = []
        for s in self.index:
            if s.start is None:
                coords.append(0)
                continue
            size = s.stop - s.start
            assert s.start % size == 0
            coords.append(s.start // size)
        # Special case for scalar.
        if len(coords) == 0:
            return "0"
        return ".".join(str(x) for x in coords)


# Tuple (and thus hashable) representation of a slice object (start, end, step).
_SliceTuple = tuple[Optional[int], Optional[int], Optional[int]]


def _slices_to_tuple(slices: list[slice]) -> tuple[_SliceTuple, ...]:
    """Converts a list of slices to a hashable representation."""
    return tuple(((s.start, s.stop, s.step) for s in slices))


def _num_replicas_per_shard(arr: Tensor) -> dict[tuple[_SliceTuple, ...], int]:
    """Gets the global replication count for each unique shard."""
    replica_count = defaultdict(int)
    for slices in arr.sharding.devices_indices_map(arr.shape).values():
        # Slice() object is not hashable before python 3.12.
        # Manually convert it to hashable types.
        replica_count[_slices_to_tuple(slices)] += 1
    return dict(replica_count)


def _get_shard_infos(
    arr_inp: Tensor, *, max_data_shard_degree: int, shard_threshold_bytes: int
) -> list[_ShardInfo]:
    """Returns a list of _ShardInfo for addressable shards that need to be saved.

    If replica count for the shards are greater than 0, all replicas will save slices of the
    shard provided that any dim of the shard is divisible by the replica count. If no such
    dim exists, we fallback to only replica 0 saving the shard.
    """
    shard_infos: list[_ShardInfo] = []
    replica_count_map = _num_replicas_per_shard(arr_inp)
    for shard in arr_inp.addressable_shards:
        replica_count = replica_count_map[_slices_to_tuple(shard.index)]
        assert replica_count > 0
        shard_degree = (
            min(replica_count, max_data_shard_degree)
            if max_data_shard_degree > 0
            else replica_count
        )
        should_skip = (
            shard_degree == 1
            or shard.data.nbytes < shard_threshold_bytes
            or shard.replica_id >= shard_degree
        )
        for axis, size in enumerate(shard.data.shape):
            # Find the first dim divisible by partial replication size.
            if should_skip or size % shard_degree != 0:
                continue
            part_size = size // shard_degree
            slice_obj = shard.index[axis]
            assert slice_obj.step is None
            start_offset = shard.replica_id * part_size
            end_offset = start_offset + part_size
            # When an axis of a tensor is not sharded, the slice object corresponding to
            # that axis will be (None, None, None).
            slice_start = slice_obj.start or 0
            shard_infos.append(
                _ShardInfo(
                    shard.data,
                    shard.index[:axis]
                    + (slice(slice_start + start_offset, slice_start + end_offset),)
                    + shard.index[axis + 1 :],
                    (start_offset, end_offset, axis),
                    shard_degree,
                )
            )
            break
        else:
            # We only have 1 replica or shard is not evenly divisible across replicas.
            # Assign replica=0 only.
            if shard.replica_id == 0:
                shard_infos.append(_ShardInfo(shard.data, shard.index, None, 1))
    return shard_infos


def _transfer_to_host(data: Tensor) -> Tensor:
    """Asynchronously transfers a shard to host memory. Does not block.

    modified from
    https://github.com/google/orbax/blob/ebb3e6d75f9ccb52bf862f1740943a45b18f4dac/checkpoint/orbax/checkpoint/serialization.py#L268
    """
    device = list(data.devices())[0]
    has_pinned_host = any(m.kind == "pinned_host" for m in device.addressable_memories())
    if has_pinned_host:
        # If available, transfer to pinned host memory.
        data = jax.device_put(
            data, jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
        )
    else:
        data.copy_to_host_async()
    return data


async def _slice_shard_and_copy_to_host(shard_infos: list[_ShardInfo]):
    """Slices each shard according to shard info and then copy the sliced result to host.

    The .data field of each shard_info is modified in-place.
    """
    # Note: jax.lax.slice_in_dim in _slice_fn will be cached in jit cache after first call.
    shard_data = jax.tree.map(_slice_fn, shard_infos)
    shard_data = jax.tree.map(_transfer_to_host, shard_data)

    await asyncio.sleep(0)  # Allow other D2Hs to launch.

    # No need to call jax.block_until_ready since np array conversion is blocking.
    for info, data in zip(shard_infos, shard_data):
        # Ensure that jax.Array's internal numpy array can be zero-copied. This guards
        # against consumers like tensorstore that would otherwise copy silently.
        info.data = np.array(data, copy=False)


def _slice_fn(info: _ShardInfo) -> Tensor:
    """Performs slicing according to a shard_info and returns the sliced array."""
    s = info.slice_arg
    if s is not None:
        return jax.lax.slice_in_dim(info.data, start_index=s[0], limit_index=s[1], axis=s[2])
    return info.data


def _local_size(arr_inp: Tensor) -> int:
    """Calculates the size of a Tensor in bytes in the local process."""
    return sum(shard.data.nbytes for shard in arr_inp.addressable_shards)


def _fix_metadata(tspec: dict[str, Any], shard_infos: list[_ShardInfo]):
    """Revises the medadata of a tensorspec based on `shard_infos`."""
    if len(shard_infos) != 0:
        # All shards have the same shape after data-sharding, so using [0] is sufficient.
        tspec["chunks"] = tuple(int(x) for x in np.maximum(1, shard_infos[0].data.shape))
    return tspec


class TensorstoreSpecModifier:
    def __call__(self, spec: dict[str, Any], *, shard_infos: list[_ShardInfo]):
        ...


async def _async_serialize(
    arr_inp: Tensor,
    tensorstore_spec: dict[str, Any],
    d2h_future: futures.Future,
    *,
    limiter: Optional[serialization._LimitInFlightBytes],
    tensorstore_spec_modifier: Optional[TensorstoreSpecModifier] = None,
    max_data_shard_degree: int,
    shard_threshold_bytes: int,
):
    """Similar to `serialization.async_serialize`, but limiting peak host memory usage and sharding
    along data-parallel axis.

    Specifically, TensorStores are opened only for shards which correspond to the current host, and
    only if
    1. Replica id of the shards is less than max_data_shard_degree or max_data_shard_degree is -1
    2. Current in-flight writes are below a user-supplied limit.

    We also simplify the API slightly by assuming replica_id=0 and primary_host=0.
    Reference:
    https://github.com/google/jax/blob/595a620804e810335a870e93975a78504b2e95e5/jax/experimental/array_serialization/serialization.py#L188
    """
    shard_infos = _get_shard_infos(
        arr_inp,
        max_data_shard_degree=max_data_shard_degree,
        shard_threshold_bytes=shard_threshold_bytes,
    )
    if not shard_infos:
        d2h_future.set_result(shard_infos)
        return

    nbytes = sum(info.data.nbytes // info.replica_count for info in shard_infos)
    # Await for limiter before D2H.
    if limiter is not None:
        await limiter.wait_for_bytes(nbytes)

    # Fully addressable arrays lead to races between multiple writing hosts.
    assert not (
        isinstance(arr_inp, array.ArrayImpl)
        and jax.process_count() > 1
        and arr_inp.is_fully_addressable
    )
    # pylint: disable-next=protected-access
    if not serialization._spec_has_metadata(tensorstore_spec):
        # pylint: disable-next=protected-access
        tensorstore_spec["metadata"] = serialization._get_metadata(arr_inp)
    if "dtype" not in tensorstore_spec:
        tensorstore_spec["dtype"] = jax.numpy.dtype(arr_inp.dtype).name

    # Original `arr_inp` might be deleted after this point.
    await _slice_shard_and_copy_to_host(shard_infos)
    # Fix metadata after slicing to get the right shape.
    _fix_metadata(tensorstore_spec["metadata"], shard_infos)
    if tensorstore_spec_modifier is not None:
        tensorstore_spec_modifier(tensorstore_spec, shard_infos=shard_infos)

    # Set future after we updated tensorstore spec.
    d2h_future.set_result(shard_infos)
    await asyncio.sleep(0)  # Allow other D2Hs to set result.

    # `ts.open` runs twice for process 0 because for the first time, we just get the future to be
    # awaited upon in the background thread. The second one runs with `assume_metadata=True` which
    # does no I/O operation and returns the tensorstore object. For every process other than `0`,
    # we open with `assume_metadata=True`.
    if jax.process_index() == 0:
        await serialization.ts.open(
            serialization.ts.Spec(tensorstore_spec),
            create=True,
            open=True,
            context=serialization.TS_CONTEXT,
        )
    t = await serialization.ts.open(
        serialization.ts.Spec(tensorstore_spec),
        open=True,
        assume_metadata=True,
        context=serialization.TS_CONTEXT,
    )

    # Avoid additional copy of input array into the TensorStore chunk cache. If `arr_inp` is a
    # jax.Array, the result of converting it to a NumPy array, as is done internally by TensorStore,
    # is guaranteed to be immutable and therefore it is safe to retain reference indefinitely.
    is_jax_array = isinstance(arr_inp, jax.Array)
    await asyncio.gather(
        *(
            t[info.index].write(info.data, can_reference_source_data_indefinitely=is_jax_array)
            for info in shard_infos
        )
    )
    if limiter is not None:
        await limiter.release_bytes(nbytes)


async def _run_serializer(
    arrays: list[Tensor],
    tensorstore_specs: list[dict[str, Any]],
    d2h_futures: list[futures.Future],
    *,
    max_concurrent_bytes: Optional[int] = None,
    tensorstore_spec_modifier: Optional[TensorstoreSpecModifier] = None,
    max_data_shard_degree: int,
    shard_threshold_bytes: int,
):
    """Asynchronously serializes a list of tensors with _async_serialize."""
    # We add 1 because LimitInFlightBytes expects a limit strictly greater than any request.
    # pylint: disable=protected-access
    limiter = (
        serialization._LimitInFlightBytes(max_concurrent_bytes + 1)
        if max_concurrent_bytes
        else None
    )
    # pylint: enable=protected-access
    future_writer = jax.tree.map(
        functools.partial(
            _async_serialize,
            limiter=limiter,
            max_data_shard_degree=max_data_shard_degree,
            shard_threshold_bytes=shard_threshold_bytes,
            tensorstore_spec_modifier=tensorstore_spec_modifier,
        ),
        arrays,
        tensorstore_specs,
        d2h_futures,
    )
    try:
        await asyncio.gather(*future_writer)
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        # If any _async_serialize call gives an exception, set future so the caller won't be
        # blocked by future.result(). This handling is sufficient because other tasks cannot
        # call set_result after set_exception in _run_serializer. The reason is that exception
        # in asyncio.gather will yield to the awaiter immediately, and the exception handling
        # code is fully synchronous. Therefore, no other synchronous code, including set_result,
        # will be called during exception handling. Event loop will cancel immediately after
        # _run_serializer exits, canceling all pending tasks.
        for fut in d2h_futures:
            if not fut.done():
                fut.set_exception(e)
        raise e


def _blocking_device_put(out: Tensor, layout: Layout) -> Tensor:
    return jax.block_until_ready(jax.device_put(out, layout))


async def _async_deserialize(
    user_in_sharding: jax.sharding.Sharding | Layout,
    tensorstore_spec: dict[str, Any],
    global_shape: Optional[Sequence[int]],
    dtype: Optional[typing.DTypeLike],
    *,
    h2d_limiter: serialization._LimitInFlightBytes,
    byte_limiter: serialization._LimitInFlightBytes,
    single_thread_pool: ThreadPoolExecutor,
):
    """Modified from
    https://github.com/jax-ml/jax/blob/e7ec418eba9ada336f755613948cbdf4a9e97d59/jax/experimental/array_serialization/serialization.py#L345

    Changes:
    1. ts.cast is used rather than np.astype to allow casting on-the-fly.
    2. Avoid allocating a zero array if the global shape is the same as the shape of the tensor
       stored in the checkpoint, which should be true for majority of the cases.
    3. Limit in flight padded H2D size to be smaller than premapped buffer size on TPU, so all H2Ds
       can fit in the pre-mapped buffer. This is to avoid the significant runtime cost of
       allocating large DMA buffers on-demand and to avoid having extra memory footprint for extra
       DMA buffers. For tensors whose size exceed the entirety of the premapped buffer, their H2D
       will be serialized using a single threaded threadpool. For non TPU backend, no limit on
       in flight H2D is imposed.

    Combination of these optimizations speed up the loading of checkpoints as much as 5x if it's
    not network-bound.

    ## Background on TPU H2D

    Each H2D consists of the following steps:

    Host buffer -> linearize -> (map DMA buffers) -> PCIe Copy, where linearization is the
    conversion from host native layout to TPU native tiled layout.

    If there is sufficient capacity in the premapped DMA buffers, the map DMA step can be skipped,
    and we linearize to a section of the pre-mapped DMA buffer directly. If there is sufficient
    capacity in the pre-mapped buffer, we can perform several linearization concurrently for
    improved performance. However, if there isn't sufficient capacity in the premapped buffer,
    on-demand DMA buffer mapping is needed, and this is often very slow. Additionally, concurrently
    mapping DMA buffers are neither faster (due to OS overhead) nor memory-efficient. Transparent
    huge pages (THP) can help, but it's only for jax 0.5.1+.
    """
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
        # Limit the bytes read for every shard.
        await byte_limiter.wait_for_bytes(requested_bytes)
        read_ts = t[restricted_domain]
        # Use ts.cast rather than np.astype since ts can perform casting on-the-fly.
        if dtype is not None:
            read_ts = ts.cast(read_ts, dtype)
        if tuple(t.shape) == shape:
            # If the restore shape is the same as shape in ckpt, we can avoid the cost of
            # allocating a zero array first.
            out = np.empty(new_shard_shape, read_ts.dtype.numpy_dtype)
        else:
            # This maybe needed because the shape the array was saved with is smaller
            # than the requested shape of the array in which it will be reloaded. So
            # the extra values will be filled with 0s.
            out = np.zeros(new_shard_shape, read_ts.dtype.numpy_dtype)

        await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(
            read_ts
        )

        # Convert to jnp array so that layouts are initialized properly for
        # sub-byte dtypes.
        # TODO(yashkatariya): This is a band-aid fix. Figure out a better way to
        # make this work.
        if out.dtype == jnp.int4:
            out = jnp.asarray(out)  # type: ignore

        out_size = out.size * out.dtype.itemsize
        # Pad to next 256mb. This is a very conservative padding.
        mb_256 = 256 * 1024 * 1024
        out_size = math.ceil(out_size / mb_256) * mb_256

        layout = Layout(dll, jax.sharding.SingleDeviceSharding(device))
        try:
            await h2d_limiter.wait_for_bytes(out_size)
            result = await loop.run_in_executor(None, _blocking_device_put, out, layout)
            await h2d_limiter.release_bytes(out_size)
        except ValueError as e:
            if "Requested more bytes than we reserved" not in str(e):
                raise e  # Raise if it's not the type of error we expect.
            logging.log_first_n(
                logging.WARNING,
                "Tensor shard for tensor %s (padded size %d bytes) exceeded "
                "premapped buffer size %d. Consider allocating larger premapped buffer using "
                "TPU_PREMAPPED_BUFFER_SIZE for improved H2D performance.",
                32,
                str(out.shape),
                out_size,
                # pylint: disable-next=protected-access
                h2d_limiter._max_bytes,
            )
            result = await loop.run_in_executor(
                single_thread_pool, _blocking_device_put, out, layout
            )

        await byte_limiter.release_bytes(requested_bytes)
        return result

    return await serialization.create_async_array_from_callback(shape, in_sharding, cb)


# Reference:
# https://github.com/google/orbax/blob/ebb3e6d75f9ccb52bf862f1740943a45b18f4dac/checkpoint/orbax/checkpoint/future.py#L49
class _ThreadRaisingException(threading.Thread):
    """Thread that raises an exception if it encounters an error."""

    _exception: Optional[Exception] = None

    def run(self):
        try:
            super().run()
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            self._exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exception is not None:
            raise self._exception


# Reference:
# https://github.com/google/orbax/blob/ebb3e6d75f9ccb52bf862f1740943a45b18f4dac/checkpoint/orbax/checkpoint/type_handlers.py#L670
class _CommitFuture:
    """Represents the result of a background commit."""

    def __init__(self, coro):
        self._t = _ThreadRaisingException(target=lambda: asyncio.run(coro))
        self._t.start()

    def result(self, timeout: Optional[int] = None) -> Any:
        return self._t.join(timeout=timeout)


def _get_premapped_buffer_size():
    if jax.default_backend() == "tpu":
        # If TPU_PREMAPPED_BUFFER_SIZE is not set, default is 4GB.
        return int(os.getenv("TPU_PREMAPPED_BUFFER_SIZE", "4294967296"))
    # On all other backends, use 1TB (effectively unlimited).
    return 1099511627776


class GlobalAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """Similar to GlobalAsyncCheckpointManager but allows passing additional futures to be awaited
    while asynchronously serializing tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        self._single_thread_pool = ThreadPoolExecutor(1)

    def stop(self):
        """Cleans up any internal threads."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._single_thread_pool.shutdown()

    def __del__(self):
        self.stop()
        return super().__del__()

    def serialize(
        self,
        arrays: list[Tensor],
        tensorstore_specs: list[dict],
        *,
        on_commit_callback: Callable[[], None],
        additional_futures: Optional[list[futures.Future]] = None,
    ):
        logging.info("Waiting for previous serialization to finish.")
        self.wait_until_finished()

        commit_futures = [[] for _ in range(len(tensorstore_specs))]

        # pylint: disable-next=redefined-outer-name
        async def _run_serializer():
            future_writer = jax.tree.map(
                serialization.async_serialize, arrays, tensorstore_specs, commit_futures
            )
            return await asyncio.gather(*future_writer)

        # Note: We need to run the coroutine in another event loop driven by a separate thread.
        # The current event loop might be already running an async function when `serialize` is
        # invoked from a coroutine, in which case asyncio.get_running_loop().run_until_complete()
        # would not be able to execute another coroutine to completion.
        asyncio.run_coroutine_threadsafe(_run_serializer(), self._loop).result()

        self._add_futures(
            jax.tree_util.tree_flatten(commit_futures)[0] + (additional_futures or [])
        )

        # Used in wait_until_finished to check on process != 0, if the checkpoint
        # has finished writing.
        self._start_async_commit(on_commit_callback)

    # Copied from (with modifications)
    # https://github.com/jax-ml/jax/blob/66037d10e7742c4fcadd07f0459a00813ec7ed5f/jax/experimental/array_serialization/serialization.py#L413-L429
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
            # Object should be created once per process.
            # pylint: disable=protected-access
            byte_limiter = serialization._LimitInFlightBytes(concurrent_bytes)
            h2d_limiter = serialization._LimitInFlightBytes(_get_premapped_buffer_size())

            future_arrays = jax.tree.map(
                functools.partial(
                    _async_deserialize,
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


class BoundedDataShardedAsyncCheckpointManager(GlobalAsyncCheckpointManager):
    """Similar to GlobalAsyncCheckpointManager but with few improvements:

    1. Tensorstore calls now run in a background event loop, hiding the cost of `ts.open` and
    `ts.copy`. Now, only D2H blocks training while serialization is fully asynchronous.
    2. Added additional sharding along data-parallel axis during save to further reduce host memory
    overhead and improves D2H time. It's achieved by sharding the first dim that's divisible by the
    data-parallel dim. We manipulate shard.index to match the sliced shard, so to tensorstore it
    behaves as if we're sharding along the data-parallel axis. If no such dim is found, we use the
    old way to save-restore, i.e. using the first (0th) replica to do the save only.
    3. Optionally one can specify max_concurrent_gb to limit in-flight host memory during
    device-to-host transfers and tensorstore writes.

    Args:
        max_concurrent_gb: Max concurrent shards (in GB) to write.
        max_data_shard_degree: Max sharding degree of model weights along data-parallel axis.
            `None` and `1` means no sharding. `-1` means fully shard along data-parallel
            replicas. `>1` means custom sharding degree and should almost always be a power of 2.
        shard_threshold_bytes: Threshold for a array shard to be data-sharded. A value of None
            or <= 0 means always data-shard according to max_data_shard_degree.
        timeout_secs: Barrier timeout in seconds.
    """

    def __init__(
        self,
        *,
        max_concurrent_gb: Optional[int] = None,
        timeout_secs: int = 300,
        max_data_shard_degree: Optional[int] = None,
        shard_threshold_bytes: Optional[int] = None,
    ):
        super().__init__(timeout_secs)
        self._logged_spec = False

        if max_concurrent_gb is None:
            self._max_concurrent_bytes = None
        else:
            if max_concurrent_gb <= 0:
                raise ValueError("max_concurrent_gb must be strictly positive.")
            self._max_concurrent_bytes = int(max_concurrent_gb * 10**9)

        self._max_data_shard_degree = 1 if max_data_shard_degree is None else max_data_shard_degree
        if self._max_data_shard_degree == 0:
            raise NotImplementedError("max_data_shard_degree cannot be 0.")
        self._shard_threshold_bytes = shard_threshold_bytes or 0

    def _tensorstore_spec_modifier(self, spec: dict[str, Any], *, shard_infos: list[_ShardInfo]):
        """A function that modifies the tensorstore spec for an array in-place.

        This function will be called after tensorstore metadata is populated and the shard infos
        for the array are computed.
        """
        del spec, shard_infos

    def _tensorstore_spec_log_fn(self, specs: list[dict[str, Any]]):
        """A function that will be called **once** after the tensorstore specs are populated.

        Specifically, this function will be called **once** during the first checkpoint after
        `self._tensorstore_spec_modifier` is invoked for each array. `specs` is a list of specs
        corresponding the `arrays` argument in `self.serialize`.
        """
        del specs

    def serialize(
        self,
        arrays: list[Tensor],
        tensorstore_specs: list[dict],
        *,
        on_commit_callback: Callable[[], None],
        additional_futures: Optional[list[futures.Future]] = None,
    ):
        """See JAX `GlobalAsyncCheckpointManager` docstring."""

        start_t = time.time()
        self.wait_until_finished()
        elapsed = time.time() - start_t
        if elapsed > 1:
            logging.warning(
                "Waiting time on previous serialization to finish is %f > 1s. "
                "Decreasing checkpointing frequency may improve performance.",
                elapsed,
            )

        max_shard_bytes = max((0, *(_local_size(array) for array in arrays)))
        max_concurrent_bytes = self._max_concurrent_bytes
        if max_concurrent_bytes is not None and max_shard_bytes > max_concurrent_bytes:
            logging.warning(
                "Max shard size %s exceeds max_concurrent_bytes %s. "
                "Will adjust max_concurrent_bytes to fit.",
                max_shard_bytes,
                max_concurrent_bytes,
            )
            max_concurrent_bytes = max_shard_bytes

        start_t = time.time()
        d2h_futures = [futures.Future() for _ in arrays]
        # Opens tensorstore and write data. This whole process happens in a
        # different thread and is fully async, except D2H because we wait for
        # then here.
        self._add_futures(
            [
                _CommitFuture(
                    _run_serializer(
                        arrays,
                        tensorstore_specs,
                        d2h_futures,
                        tensorstore_spec_modifier=self._tensorstore_spec_modifier,  # type: ignore
                        max_concurrent_bytes=max_concurrent_bytes,
                        max_data_shard_degree=self._max_data_shard_degree,
                        shard_threshold_bytes=self._shard_threshold_bytes,
                    )
                )
            ]
            + (additional_futures or [])
        )

        # Block until D2H is complete and get shard_info for logging.
        # Training can only proceed after D2H finishes.
        shard_infos = [f.result() for f in d2h_futures]

        if not self._logged_spec:
            # Log this only once.
            for arr, infos, spec in zip(arrays, shard_infos, tensorstore_specs):
                logging.info(
                    "Addressable shard shape: %s, saving shard shape: %s, spec: %s",
                    str(arr.addressable_data(0).shape),
                    "None" if len(infos) == 0 else str(infos[0].data.shape),
                    str(spec),
                )
            self._logged_spec = True
            self._tensorstore_spec_log_fn(tensorstore_specs)

        logging.info("D2H during save took %fs. Starting async commit.", time.time() - start_t)
        self._start_async_commit(on_commit_callback)
