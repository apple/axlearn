# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/jax:
# Copyright 2021 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Array serialization utilities.

Reference:
https://github.com/google/jax/blob/595a620804e810335a870e93975a78504b2e95e5/jax/experimental/array_serialization/serialization.py
"""

import asyncio
import functools
from typing import Callable, Dict, List, Set

import jax
from absl import logging
from jax._src.array import Shard
from jax.experimental.array_serialization import serialization

from axlearn.common.utils import Tensor


def _proxy(fut: asyncio.Future) -> asyncio.Future:
    """Returns a proxy that can be used to await (but does not cancel) `fut`."""
    loop = asyncio.get_event_loop()
    proxy = loop.create_future()

    def callback(_):
        # Callback may be invoked from a separate event loop.
        if not loop.is_closed():
            loop.call_soon_threadsafe(proxy.set_result, None)

    fut.add_done_callback(callback)
    return proxy


async def _release(
    limiter: serialization._LimitInFlightBytes,
    commit: asyncio.Future,
    nbytes: int,
):
    """Releases resources when `commit` completes."""
    await _proxy(commit)
    await limiter.release_bytes(nbytes)


async def _acquire_and_write(
    t,
    *,
    limiter: serialization._LimitInFlightBytes,
    shard: Shard,
    nbytes: int,
    release_tasks: Set,
):
    """Initiates a write for the given shard.

    This waits until `limiter` admits the shard, and blocks until the device-host copy is complete
    before returning. The shard may be committed in an async fashion; the commit future is returned
    so that the caller can await it at a later point in time.
    """
    await limiter.wait_for_bytes(nbytes)
    # TODO(markblee): Investigate can_reference_source_data_indefinitely after updating tensorstore.
    write_future = t[shard.index].write(shard.data)
    await write_future.copy
    # Release without blocking the event loop (commits can complete asynchronously).
    release_task = asyncio.create_task(_release(limiter, write_future.commit, nbytes))
    release_tasks.add(release_task)
    release_task.add_done_callback(release_tasks.discard)
    return write_future.commit


def _local_shards(array: Tensor) -> List[Shard]:
    """Returns addressable shards with replica_id=0."""
    return [shard for shard in array.addressable_shards if shard.replica_id == 0]


async def async_serialize(
    array: Tensor,
    tensorstore_spec: Dict,
    *,
    limiter: serialization._LimitInFlightBytes,
) -> List[asyncio.Future]:
    """Similar to `serialization.async_serialize`, but limiting peak host memory usage.

    Specifically, TensorStores are opened only for shards which correspond to the current host, and
    only if the current in-flight writes are below a user-supplied limit.

    We also simplify the API slightly by assuming replica_id=0 and primary_host=0, and by returning
    commit futures rather than mutating an input array.

    Reference:
    https://github.com/google/jax/blob/595a620804e810335a870e93975a78504b2e95e5/jax/experimental/array_serialization/serialization.py#L188
    """
    # pylint: disable=protected-access

    # Fully addressable arrays lead to races between multiple writing hosts.
    assert not (
        isinstance(array, serialization.array.ArrayImpl)
        and jax.process_count() > 1
        and array.is_fully_addressable
    )
    if not serialization._spec_has_metadata(tensorstore_spec):
        tensorstore_spec["metadata"] = serialization._get_metadata(array)
    if "dtype" not in tensorstore_spec:
        tensorstore_spec["dtype"] = jax.numpy.dtype(array.dtype).name

    if jax.process_index() == 0:
        await serialization.ts.open(
            serialization.ts.Spec(tensorstore_spec),
            create=True,
            open=True,
        )

    local_shards = _local_shards(array)
    if not local_shards:
        return []

    # Opening with assume_metadata=True should incur no IO ops.
    t = await serialization.ts.open(
        serialization.ts.Spec(tensorstore_spec),
        open=True,
        assume_metadata=True,
        context=serialization.TS_CONTEXT,
    )
    # Keep references to release() calls, since asyncio only keeps weak references to tasks:
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    release_tasks = set()
    return await asyncio.gather(
        *(
            # Memory usage seems to be proportional to the array size rather than shard sizes.
            # TODO(markblee): Investigate why this is the case.
            _acquire_and_write(
                t, limiter=limiter, shard=shard, nbytes=array.nbytes, release_tasks=release_tasks
            )
            for shard in local_shards
        )
    )


class BoundedAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """A concurrency-bounded implementation of JAX array serialization.

    The main difference is that we attempt to keep at most `max_concurrent_gb` bytes in memory.
    """

    def __init__(self, *, max_concurrent_gb: int, timeout_secs: int = 300):
        super().__init__(timeout_secs)
        if max_concurrent_gb <= 0:
            raise ValueError("max_concurrent_gb must be strictly positive.")
        self._max_concurrent_bytes = int(max_concurrent_gb * 10**9)

    def serialize(
        self,
        arrays: List[Tensor],
        tensorstore_specs: List[Dict],
        *,
        on_commit_callback: Callable[[], None],
    ):
        """See JAX `GlobalAsyncCheckpointManager` docstring."""

        logging.info("Waiting for previous serialization to finish.")
        self.wait_until_finished()

        max_shard_bytes = max((0, *(array.nbytes for array in arrays)))
        max_concurrent_bytes = self._max_concurrent_bytes
        if max_shard_bytes > max_concurrent_bytes:
            logging.warning(
                "Max shard size %s exceeds max_concurrent_bytes %s. "
                "Will adjust max_concurrent_bytes to fit.",
                max_shard_bytes,
                max_concurrent_bytes,
            )
            max_concurrent_bytes = max_shard_bytes

        async def _run_serializer(*, arrays, tensorstore_specs, max_concurrent_bytes):
            # We add 1 because LimitInFlightBytes expects a limit strictly greater than any request.
            # pylint: disable-next=protected-access
            limiter = serialization._LimitInFlightBytes(max_concurrent_bytes + 1)
            future_writer = jax.tree_util.tree_map(
                functools.partial(async_serialize, limiter=limiter),
                arrays,
                tensorstore_specs,
            )
            return await asyncio.gather(*future_writer)

        commit_futures = asyncio.run(
            _run_serializer(
                arrays=arrays,
                tensorstore_specs=tensorstore_specs,
                max_concurrent_bytes=max_concurrent_bytes,
            ),
        )
        commit_futures = jax.tree_util.tree_flatten(commit_futures)[0]
        self._add_futures(commit_futures)
        logging.info("Starting async commit.")
        self._start_async_commit(on_commit_callback)
