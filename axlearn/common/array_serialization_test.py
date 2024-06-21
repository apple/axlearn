# Copyright © 2024 Apple Inc.

"""Tests array serialization utils."""
# pylint: disable=protected-access

import asyncio
import contextlib
import functools
from typing import List, Optional
from unittest import mock

import jax.numpy as jnp
from absl.testing import parameterized
from jax import Shard

from axlearn.common import array_serialization
from axlearn.common.array_serialization import (
    BoundedAsyncCheckpointManager,
    _acquire_and_write,
    _proxy,
    async_serialize,
    serialization,
)


class _LimitAndRecordBytes(serialization._LimitInFlightBytes):
    """A thin wrapper that records peak concurrent bytes."""

    def __init__(self, num_bytes):
        super().__init__(num_bytes)
        self._max_concurrent_bytes = 0
        self._wait_calls = []

    async def wait_for_bytes(self, requested_bytes):
        self._wait_calls.append(requested_bytes)
        await super().wait_for_bytes(requested_bytes)
        self._max_concurrent_bytes += requested_bytes

    async def release_bytes(self, requested_bytes):
        await super().release_bytes(requested_bytes)
        self._max_concurrent_bytes -= requested_bytes

    def get_stats(self):
        return self._max_concurrent_bytes, self._wait_calls


class SerializerTest(parameterized.TestCase):
    """Tests serialization utils."""

    def test_fully_addressable(self):
        """Tests that we don't attempt to serialize fully addressable arrays across multiple hosts,
        which can lead to races.
        """
        with mock.patch("jax.process_count", return_value=2), self.assertRaises(Exception):
            asyncio.run(
                async_serialize(jnp.array(1), {}, limiter=serialization._LimitInFlightBytes(1)),
                debug=True,
            )

    @parameterized.parameters(
        # Test that both replica_id=0 shards are written.
        dict(
            process_index=0,
            shards=[
                dict(id=0, replica_id=0, nbytes=2),
                dict(id=1, replica_id=0, nbytes=1),
                dict(id=2, replica_id=1, nbytes=2),
            ],
            expect_opened=[0, 1],
        ),
        # Test that only replica_id=1 shard is written.
        dict(
            process_index=1,
            shards=[
                dict(id=0, replica_id=1, nbytes=2),
                dict(id=1, replica_id=1, nbytes=1),
                dict(id=2, replica_id=0, nbytes=2),
            ],
            expect_opened=[2],
        ),
        # Test a case with no shards.
        dict(process_index=0, shards=[], expect_opened=[]),
        # Test a case with no shards.
        dict(
            process_index=0,
            shards=[dict(id=0, replica_id=1, nbytes=2)],
            expect_opened=[],
        ),
    )
    def test_async_serialize(
        self,
        process_index: int,
        shards: List,
        expect_opened: List[int],
    ):
        """Tests async_serialize.

        We should only write shards which belong to the current process.
        """
        mock_open_fut = mock.AsyncMock()
        mock_open = mock.Mock(side_effect=mock_open_fut)
        ts_spec = {"dummy_key": "dummy_value"}
        shards = [
            mock.MagicMock(
                id=shard["id"],
                replica_id=shard["replica_id"],
                data=mock.Mock(nbytes=shard["nbytes"]),
            )
            for shard in shards
        ]
        array = mock.MagicMock(
            dtype=jnp.bfloat16,
            addressable_shards=shards,
        )

        patch_ts = mock.patch.multiple(
            f"{serialization.__name__}.ts",
            open=mock_open,
            Spec=mock.DEFAULT,
        )
        patch_process_id = mock.patch("jax.process_index", return_value=process_index)
        patch_write = mock.patch(f"{array_serialization.__name__}._acquire_and_write")

        with patch_process_id, patch_write as mock_write, patch_ts as mock_ts:
            asyncio.run(async_serialize(array, ts_spec, limiter=mock.Mock()), debug=True)

            # Check that open with assume_metadata=False is only invoked for process 0.
            self.assertEqual(
                not mock_open.call_args_list[0][1].get("assume_metadata", False),
                process_index == 0,
            )

            # Check that open with assume_metadata=True is called with the right spec.
            self.assertTrue(
                all(call_args[0] == (ts_spec,) for call_args in mock_ts["Spec"].call_args_list)
            )
            self.assertGreater(mock_open_fut.await_count, 0)

            # Check that open_and_write is only invoked for local shards.
            if expect_opened:
                # Make sure we opened all owned shards.
                opened_shards = [call_args[1]["shard"] for call_args in mock_write.call_args_list]
                self.assertCountEqual(expect_opened, [shard.id for shard in opened_shards])
            else:
                # If no local shards, open_and_write should not be invoked.
                self.assertFalse(mock_write.called)

    def test_proxy_cancel(self):
        """Tests that cancelling a proxy does not cancel the original future."""

        async def call():
            fut = asyncio.Future()
            proxy_coro = _proxy(fut)
            proxy_coro.cancel()
            self.assertFalse(fut.cancelled())

        asyncio.run(call(), debug=True)

    def test_proxy_await(self):
        """Tests that awaiting a proxy awaits the original future."""

        async def set_result(f):
            await asyncio.sleep(0.1)
            f.set_result(None)

        async def call():
            fut = asyncio.Future()
            asyncio.create_task(set_result(fut))
            await _proxy(fut)
            self.assertTrue(fut.done())

        asyncio.run(call(), debug=True)

    @parameterized.parameters(
        # Test a case where we process shards one-by-one.
        dict(
            shards=[1, 1, 1],
            max_concurrent_bytes=1,
            expect_max_concurrent_bytes=1,
        ),
        # Test a case where we process multiple shards at once.
        dict(
            shards=[1, 2, 1, 1, 1],
            max_concurrent_bytes=3,
            expect_max_concurrent_bytes=3,
        ),
        # Test a case where copy fails, which should surface immediately.
        dict(
            shards=[1, 1],
            max_concurrent_bytes=1,
            expect_max_concurrent_bytes=1,
            copies=[None, RuntimeError("copy_fail")],
            expect_error=RuntimeError("copy_fail"),
        ),
        # Test a case where commit fails, but we don't wait for it.
        dict(
            shards=[1],
            max_concurrent_bytes=3,
            expect_max_concurrent_bytes=1,
            commits=[RuntimeError("commit_fail")],
        ),
    )
    def test_acquire_and_write(
        self,
        shards: List[int],
        max_concurrent_bytes: int,
        expect_max_concurrent_bytes: int,
        commits: Optional[List[RuntimeError]] = None,
        copies: Optional[List[RuntimeError]] = None,
        expect_error: Optional[RuntimeError] = None,
    ):
        """Tests TensorStore write.

        Specifically, it should respect max_concurrent_bytes, and block on copies but not commits.
        """
        written = []
        concurrent_bytes = 0
        shards = [
            mock.Mock(index=i, data=mock.Mock(nbytes=nbytes)) for i, nbytes in enumerate(shards)
        ]
        num_shards = len(shards)
        commits = commits or [None] * num_shards
        copies = copies or [None] * num_shards

        async def mock_commit(i):
            nonlocal concurrent_bytes
            if commits[i] is not None:
                raise commits[i]
            await asyncio.sleep(0.1)
            concurrent_bytes -= shards[i].data.nbytes

        async def mock_copy(i):
            nonlocal concurrent_bytes
            concurrent_bytes += shards[i].data.nbytes
            await asyncio.sleep(0)  # Yield to event loop.
            if copies[i] is not None:
                raise copies[i]

        expected_commit_futures = [
            mock.AsyncMock(wraps=functools.partial(mock_commit, i=i))() for i in range(num_shards)
        ]

        def mock_write(data, i):
            written.append(data)
            return mock.AsyncMock(
                copy=mock.AsyncMock(wraps=functools.partial(mock_copy, i=i))(),
                commit=expected_commit_futures[i],
            )

        mock_write_futs = [
            mock.Mock(**{"write.side_effect": functools.partial(mock_write, i=i)})
            for i in range(num_shards)
        ]

        def mock_ts_index(self, idx):
            del self
            return mock_write_futs[idx]

        class MockTs(mock.AsyncMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__getitem__ = mock_ts_index

        mock_ts = MockTs()

        async def acquire_and_write(*, shards: List[Shard]):
            limiter = _LimitAndRecordBytes(max_concurrent_bytes + 1)
            release_tasks = set()
            commit_futures = await asyncio.gather(
                *(
                    _acquire_and_write(
                        mock_ts,
                        limiter=limiter,
                        shard=shard,
                        nbytes=shard.data.nbytes,
                        release_tasks=release_tasks,
                    )
                    for i, shard in enumerate(shards)
                )
            )
            return commit_futures, limiter.get_stats()

        if expect_error:
            ctx = self.assertRaisesRegex(type(expect_error), str(expect_error))
        else:
            ctx = contextlib.nullcontext()

        with (
            ctx,
            # Mock out _proxy since the commit futures aren't actually futures.
            mock.patch(f"{array_serialization.__name__}._proxy", side_effect=lambda fut: fut),
        ):
            commit_futures, (limiter_concurrent_bytes, _) = asyncio.run(
                acquire_and_write(shards=shards), debug=True
            )
            # Check that max_concurrent_bytes is respected (as recorded by limiter).
            self.assertBetween(limiter_concurrent_bytes, 1, expect_max_concurrent_bytes)
            # Check that max_concurrent_bytes is respected (as recorded by actual copy/commit).
            self.assertBetween(concurrent_bytes, 1, expect_max_concurrent_bytes)
            # Check that all shards written once.
            self.assertCountEqual([shard.data for shard in shards], written)
            # Check that commit futures are tracked.
            self.assertCountEqual(expected_commit_futures, commit_futures)

    @parameterized.parameters(
        # Test that we can always fit the largest shard.
        dict(
            arrays=[[1, 2], [1]],
            max_concurrent_gb=1,
            expect_max_concurrent_gb=3,
        ),
        # Test empty shards.
        dict(
            arrays=[],
            max_concurrent_gb=1,
            expect_max_concurrent_gb=1,
        ),
    )
    def test_serialize(
        self, arrays: List[List[int]], max_concurrent_gb: int, expect_max_concurrent_gb: int
    ):
        arrays = [
            mock.Mock(
                addressable_shards=[
                    mock.Mock(replica_id=0, **{"data.nbytes": int(shard * 10**9)})
                    for shard in array
                ],
                nbytes=int(sum(array) * 10**9),
            )
            for array in arrays
        ]
        tensorstore_specs = [{} for _ in range(len(arrays))]
        expect_max_concurrent_bytes = int(expect_max_concurrent_gb * 10**9)

        manager = BoundedAsyncCheckpointManager(max_concurrent_gb=max_concurrent_gb)
        with (
            mock.patch(f"{array_serialization.__name__}.async_serialize") as mock_serialize,
            mock.patch(f"{serialization.__name__}._LimitInFlightBytes") as mock_limiter,
            mock.patch.multiple(
                manager,
                wait_until_finished=mock.DEFAULT,
                _add_futures=mock.DEFAULT,
                _start_async_commit=mock.DEFAULT,
            ) as mocks,
        ):
            manager.serialize(arrays, tensorstore_specs, on_commit_callback=lambda: None)

            self.assertTrue(mocks["wait_until_finished"].called)

            # Make sure async_serialize called for all arrays.
            self.assertCountEqual(
                list(zip(arrays, tensorstore_specs)),
                [call_args[0] for call_args in mock_serialize.call_args_list],
            )
            # Make sure limiter constructed with the right limit.
            self.assertIn(expect_max_concurrent_bytes + 1, mock_limiter.call_args[0])

    def test_max_concurrency(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            BoundedAsyncCheckpointManager(max_concurrent_gb=0)

    def test_serialize_consecutive(self):
        """Tests that serialize waits for prior serialization to finish."""
        manager = BoundedAsyncCheckpointManager(max_concurrent_gb=1)
        mock_serialize = mock.patch(
            f"{array_serialization.__name__}.async_serialize", return_value=mock.AsyncMock()
        )
        with mock_serialize, mock.patch.object(manager, "wait_until_finished") as mock_wait:
            manager.serialize([], [], on_commit_callback=lambda *args: None)
            self.assertTrue(mock_wait.called)
