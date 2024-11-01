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
import threading
import time
from collections import defaultdict
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import numpy as np
from absl import logging
from jax._src import array, config
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


def _get_shard_infos(arr_inp: Tensor, *, max_data_shard_degree: int) -> list[_ShardInfo]:
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
        for axis, size in enumerate(shard.data.shape):
            # Find the first dim divisible by partial replication size.
            if max_data_shard_degree == 1 or replica_count == 1 or size % replica_count != 0:
                continue
            part_size = size // replica_count
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
                    replica_count,
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
    if config.enable_memories.value and has_pinned_host:
        # If available, transfer to pinned host memory.
        data = jax.device_put(
            data, jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
        )
    else:
        data.copy_to_host_async()
    return data


async def _slice_shard_and_copy_to_host(shard_infos: list[_ShardInfo], d2h_future: futures.Future):
    """Slices each shard according to shard info and then copy the sliced result to host.

    The .data field of each shard_info is modified in-place.
    """
    # Note: jax.lax.slice_in_dim in _slice_fn will be cached in jit cache after first call.
    shard_data = jax.tree_map(_slice_fn, shard_infos)
    shard_data = jax.tree_map(_transfer_to_host, shard_data)

    await asyncio.sleep(0)  # Allow other D2Hs to launch.

    # No need to call jax.block_until_ready since np array conversion is blocking.
    for info, data in zip(shard_infos, shard_data):
        # Ensure that jax.Array's internal numpy array can be zero-copied. This guards
        # against consumers like tensorstore that would otherwise copy silently.
        info.data = np.array(data, copy=False)

    d2h_future.set_result(shard_infos)
    await asyncio.sleep(0)  # Allow other D2Hs to set result.


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
        tspec["chunks"] = np.array(np.maximum(1, shard_infos[0].data.shape))
    return tspec


async def _async_serialize(
    arr_inp: Tensor,
    tensorstore_spec: dict[str, Any],
    d2h_future: futures.Future,
    *,
    limiter: Optional[serialization._LimitInFlightBytes] = None,
    max_data_shard_degree: Optional[int] = None,
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
    shard_infos = _get_shard_infos(arr_inp, max_data_shard_degree=max_data_shard_degree)
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
    await _slice_shard_and_copy_to_host(shard_infos, d2h_future)
    # Fix metadata after slicing to get the right shape.
    _fix_metadata(tensorstore_spec["metadata"], shard_infos)

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
    max_data_shard_degree: Optional[int] = None,
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
            _async_serialize, limiter=limiter, max_data_shard_degree=max_data_shard_degree
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


class GlobalAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """Similar to GlobalAsyncCheckpointManager but allows passing additional futures to be awaited
    while asynchronously serializing tensors.
    """

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

        asyncio.run(_run_serializer())

        self._add_futures(
            jax.tree_util.tree_flatten(commit_futures)[0] + (additional_futures or [])
        )

        # Used in wait_until_finished to check on process != 0, if the checkpoint
        # has finished writing.
        self._start_async_commit(on_commit_callback)


class BoundedDataShardedAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """Similar to GlobalAsyncCheckpointManager but with few improvements:

    1. Writing to tensorstore requires no host-to-host copy most of the time. This reduces host
    memory usage while also reduces blocking time of the checkpointing process.
    2. Tensorstore calls now run in a background event loop, hiding the cost of `ts.open` and
    `ts.copy`. Now, only D2H blocks training while serialization is fully asynchronous.
    3. Added additional sharding along data-parallel axis during save to further reduce host memory
    overhead and improves D2H time. It's achieved by sharding the first dim that's divisible by the
    data-parallel dim. We manipulate shard.index to match the sliced shard, so to tensorstore it
    behaves as if we're sharding along the data-parallel axis. If no such dim is found, we use the
    old way to save-restore, i.e. using the first (0th) replica to do the save only.
    4. Optionally one can specify max_concurrent_gb to limit in-flight host memory during
    device-to-host transfers and tensorstore writes.

    Args:
        max_concurrent_gb: Max concurrent shards (in GB) to write.
        max_data_shard_degree: Max sharding degree of model weights along data-parallel axis.
            `None` and `1` means no sharding. `-1` means fully shard along data-parallel
            replicas. `>1` means custom sharding degree (currently not implemented).
        timeout_secs: Barrier timeout in seconds.
    """

    def __init__(
        self,
        *,
        max_concurrent_gb: Optional[int] = None,
        timeout_secs: int = 300,
        max_data_shard_degree: Optional[int] = None,
    ):
        super().__init__(timeout_secs)
        self._logged_spec = False

        if max_concurrent_gb is None:
            self._max_concurrent_bytes = None
        else:
            if max_concurrent_gb <= 0:
                raise ValueError("max_concurrent_gb must be strictly positive.")
            self._max_concurrent_bytes = int(max_concurrent_gb * 10**9)

        self._max_data_shard_degree = max_data_shard_degree or 1
        if self._max_data_shard_degree not in (1, -1):
            raise NotImplementedError(
                "max_data_shard_degree is not implemented for values other than 1 and -1"
            )

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
                        max_concurrent_bytes=max_concurrent_bytes,
                        max_data_shard_degree=self._max_data_shard_degree,
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

        logging.info("D2H during save took %fs. Starting async commit.", time.time() - start_t)
        self._start_async_commit(on_commit_callback)
