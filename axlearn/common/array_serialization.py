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
from typing import Callable, Dict, List

import jax
from absl import logging
from jax._src.array import Shard
from jax.experimental.array_serialization import serialization

from axlearn.common.utils import Tensor


def _proxy(fut: asyncio.Future) -> asyncio.Future:
    """Returns a proxy that can be used to await (but does not cancel) `fut`."""
    proxy = asyncio.Future()

    def callback(_):
        proxy.set_result(None)

    fut.add_done_callback(callback)
    return proxy


async def _release(limiter: asyncio.BoundedSemaphore, commit: asyncio.Future):
    """Releases resources when `commit` completes."""
    await _proxy(commit)
    limiter.release()


async def _open_and_write(
    *,
    limiter: asyncio.BoundedSemaphore,
    tensorstore_spec: Dict,
    shard: Shard,
):
    """Initiates a write for the given shard.

    This waits until `limiter` admits the shard, and blocks until the device-host copy is complete
    before returning. The shard may be committed in an async fashion, with the commit future
    appended to `commit_futures`.
    """
    await limiter.acquire()

    # Opening with assume_metadata=True should incur no IO ops.
    t = await serialization.ts.open(
        serialization.ts.Spec(tensorstore_spec),
        open=True,
        assume_metadata=True,
        context=serialization.TS_CONTEXT,
    )

    # TODO(markblee): investigate can_reference_source_data_indefinitely after updating tensorstore.
    write_future = t[shard.index].write(shard.data)
    await write_future.copy

    # Release without blocking the event loop (commits can complete asynchronously).
    asyncio.create_task(_release(limiter, write_future.commit))
    return write_future.commit


async def async_serialize(
    array: Tensor,
    tensorstore_spec: Dict,
    *,
    limiter: asyncio.BoundedSemaphore,
) -> List[asyncio.Future]:
    """Similar to `serialization.async_serialize`, but limiting peak host memory usage.

    Specifically, TensorStores are opened on a per-shard basis, only for shards which correspond
    to the current host, and only if the current in-flight writes are below a user-supplied limit.

    We also simplify the API slightly by assuming replica_id=0 and primary_host=0.

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

    commit_futures = []

    if jax.process_index() == 0:
        await serialization.ts.open(
            serialization.ts.Spec(tensorstore_spec),
            create=True,
            open=True,
        )

    local_shards = [shard for shard in array.addressable_shards if shard.replica_id == 0]
    if not local_shards:
        return commit_futures

    commit_futures.extend(
        await asyncio.gather(
            *(
                _open_and_write(
                    limiter=limiter,
                    tensorstore_spec=tensorstore_spec,
                    shard=shard,
                )
                for shard in local_shards
            )
        )
    )
    return commit_futures


class BoundedAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """A concurrency-bounded implementation of JAX array serialization.

    The main difference is that we write at most `max_concurrency` shards concurrently.
    """

    def __init__(self, *, max_concurrency: int, timeout_secs: int = 300):
        super().__init__(timeout_secs)
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be strictly positive.")
        self._max_concurrency = max_concurrency

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

        async def _run_serializer(*, arrays, tensorstore_specs, max_concurrency):
            limiter = asyncio.BoundedSemaphore(value=max_concurrency)
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
                max_concurrency=self._max_concurrency,
            )
        )
        commit_futures = jax.tree_util.tree_flatten(commit_futures)[0]
        self._add_futures(commit_futures)
        logging.info("Starting async commit.")
        self._start_async_commit(on_commit_callback)
