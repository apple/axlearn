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
    _open_and_write,
    _proxy,
    async_serialize,
    serialization,
)


class SerializerTest(parameterized.TestCase):
    """Tests serialization utils."""

    def test_fully_addressable(self):
        """Tests that we don't attempt to serialize fully addressable arrays across multiple hosts,
        which can lead to races.
        """
        with mock.patch("jax.process_count", return_value=2), self.assertRaises(Exception):
            asyncio.run(async_serialize(jnp.array(1), {}, limiter=asyncio.BoundedSemaphore()))

    @parameterized.parameters(
        dict(process_index=0),
        dict(process_index=1),
        dict(process_index=0, addressable_shards=[]),
    )
    def test_async_serialize(self, process_index: int, addressable_shards: Optional[List] = None):
        """Tests async_serialize.

        We should only open_and_write shards which belong to the current process.
        """
        mock_open_fut = mock.AsyncMock()()
        mock_open = mock.Mock(return_value=mock_open_fut)
        ts_spec = {"dummy_key": "dummy_value"}
        if addressable_shards is None:
            addressable_shards = [
                mock.MagicMock(id=0, replica_id=0),
                mock.MagicMock(id=1, replica_id=0),
                mock.MagicMock(id=2, replica_id=1),
            ]
        array = mock.MagicMock(
            dtype=jnp.bfloat16,
            addressable_shards=addressable_shards,
        )

        patch_ts = mock.patch.multiple(
            f"{serialization.__name__}.ts",
            open=mock_open,
            Spec=mock.DEFAULT,
        )
        patch_process_id = mock.patch("jax.process_index", return_value=process_index)
        patch_write = mock.patch(f"{array_serialization.__name__}._open_and_write")

        async def serialize(array, spec):
            limiter = asyncio.BoundedSemaphore()
            return await async_serialize(array, spec, limiter=limiter)

        with patch_process_id, patch_write as mock_write, patch_ts as mock_ts:
            asyncio.run(serialize(array, ts_spec))
            # Check that open is only invoked for process 0.
            self.assertEqual(mock_open.called, process_index == 0)
            # Check that open is invoked with the right spec.
            if process_index == 0:
                self.assertIn(ts_spec, mock_ts["Spec"].call_args[0])
            else:
                self.assertIsNone(mock_ts["Spec"].call_args)

            # Check that open_and_write is only invoked for local shards.
            # If no local shards, open_and_write should not be invoked.
            if addressable_shards:
                expected_shard_ids = [
                    shard.id for shard in addressable_shards if shard.replica_id == 0
                ]
                opened_shards = [call_args[1]["shard"] for call_args in mock_write.call_args_list]
                self.assertCountEqual(expected_shard_ids, [shard.id for shard in opened_shards])
            else:
                self.assertFalse(mock_write.called)

    def test_proxy_cancel(self):
        """Tests that cancelling a proxy does not cancel the original future."""

        async def call():
            fut = asyncio.Future()
            proxy_fut = _proxy(fut)
            proxy_fut.cancel()
            self.assertFalse(fut.cancelled())

        asyncio.run(call())

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

        asyncio.run(call())

    @parameterized.parameters(
        # Test a case where we process shards one-by-one.
        dict(num_shards=3, max_concurrency=1),
        # Test a case where we process shards all at once.
        dict(num_shards=3, max_concurrency=3),
        # Test a case where copy fails, which should surface immediately.
        dict(
            num_shards=2,
            max_concurrency=1,
            copies=[None, RuntimeError("copy_fail")],
            expect_error=RuntimeError("copy_fail"),
        ),
        # Test a case where commit fails, but we don't wait for it.
        dict(num_shards=1, max_concurrency=3, commits=[RuntimeError("commit_fail")]),
    )
    def test_open_and_write(
        self,
        num_shards: int,
        max_concurrency: int,
        commits: Optional[List[RuntimeError]] = None,
        copies: Optional[List[RuntimeError]] = None,
        expect_error: Optional[RuntimeError] = None,
    ):
        """Tests open_and_write.

        Specifically, it should respect max_concurrency, and block on copies but not commits.
        """
        written = []
        concurrent_copy = 0
        max_concurrency_actual = 0
        commits = commits or [None] * num_shards
        copies = copies or [None] * num_shards

        async def mock_commit(i):
            if commits[i] is not None:
                raise commits[i]
            await asyncio.sleep(1)
            return i

        async def mock_copy(i):
            nonlocal concurrent_copy
            await asyncio.sleep(0)  # Yield to event loop.
            concurrent_copy -= 1
            if copies[i] is not None:
                raise copies[i]

        expected_commit_futures = [
            mock.AsyncMock(wraps=functools.partial(mock_commit, i=i))() for i in range(num_shards)
        ]

        def mock_write(i):
            nonlocal concurrent_copy, max_concurrency_actual
            concurrent_copy += 1
            max_concurrency_actual = max(max_concurrency_actual, concurrent_copy)
            written.append(i)
            return mock.AsyncMock(
                copy=mock.AsyncMock(wraps=functools.partial(mock_copy, i=i))(),
                commit=expected_commit_futures[i],
            )

        mock_write_futs = [
            mock.Mock(**{"write.side_effect": mock_write}) for _ in range(num_shards)
        ]

        def mock_ts_index(self, idx):
            del self
            return mock_write_futs[idx]

        class MockTs(mock.AsyncMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__getitem__ = mock_ts_index

        mock_ts = mock.AsyncMock(return_value=MockTs())

        async def open_and_write(*, shards: List[Shard]):
            limiter = asyncio.BoundedSemaphore(max_concurrency)
            return await asyncio.gather(
                *(
                    _open_and_write(
                        limiter=limiter,
                        tensorstore_spec={"idx": i},
                        shard=shard,
                    )
                    for i, shard in enumerate(shards)
                )
            )

        if expect_error:
            ctx = self.assertRaisesRegex(type(expect_error), str(expect_error))
        else:
            ctx = contextlib.nullcontext()

        with (
            ctx,
            # Mock out _proxy since the commit futures aren't actually futures.
            mock.patch(f"{array_serialization.__name__}._proxy", side_effect=lambda fut: fut),
            mock.patch.multiple(
                f"{serialization.__name__}.ts",
                open=mock_ts,
                Spec=mock.DEFAULT,
            ) as mocks,
        ):
            shards = [mock.Mock(index=i, data=i) for i in range(num_shards)]
            commit_futures = asyncio.run(open_and_write(shards=shards))

            # Check that open is called with the right spec.
            self.assertCountEqual(
                [({"idx": i},) for i in range(num_shards)],
                [call_args[0] for call_args in mocks["Spec"].call_args_list],
            )

            # Check that open is awaited once per shard.
            self.assertEqual(num_shards, mock_ts.await_count)
            # Check that max_concurrency is respected.
            self.assertLessEqual(max_concurrency_actual, max_concurrency)
            # Check that all shards written once.
            self.assertCountEqual([shard.data for shard in shards], written)
            # Check that commit futures are tracked.
            self.assertCountEqual(expected_commit_futures, commit_futures)

    def test_max_concurrency(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            BoundedAsyncCheckpointManager(max_concurrency=0)

    def test_serialize_consecutive(self):
        """Tests that serialize waits for prior serialization to finish."""
        manager = BoundedAsyncCheckpointManager(max_concurrency=1)
        mock_serialize = mock.patch(
            f"{array_serialization.__name__}.async_serialize", return_value=mock.AsyncMock()
        )
        with mock_serialize, mock.patch.object(manager, "wait_until_finished") as mock_wait:
            manager.serialize([], [], on_commit_callback=lambda *args: None)
            self.assertTrue(mock_wait.called)
