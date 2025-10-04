# Copyright © 2024 Apple Inc.

"""Tests array serialization utils."""
# pylint: disable=protected-access

import asyncio
import functools
import re
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorstore as ts
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils

from axlearn.common import array_serialization
from axlearn.common.array_serialization import (
    BoundedDataShardedAsyncCheckpointManager,
    _async_serialize,
    _CommitFuture,
    _get_shard_infos,
    _num_replicas_per_shard,
    _run_serializer,
    _ShardInfo,
    _slice_shard_and_copy_to_host,
    _transfer_to_host,
    futures,
    serialization,
)


@contextmanager
def get_tensorstore_spec(arr: jax.Array):
    """Create tensorstore for an array and saves to a temporary location."""
    tempdir = tempfile.TemporaryDirectory()
    tensorstore_spec = {
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": tempdir.name,
        },
        "metadata": {
            "dataType": str(arr.dtype),
            "dimensions": arr.shape,
        },
        "create": True,
        "delete_existing": True,
    }
    try:
        yield tensorstore_spec
    finally:
        tempdir.cleanup()


class SerializerTest(parameterized.TestCase):
    """Tests serialization utils."""

    def test_fully_addressable(self):
        """Tests that we don't attempt to serialize fully addressable arrays across multiple hosts,
        which can lead to races.
        """
        with mock.patch("jax.process_count", return_value=2), self.assertRaises(Exception):
            asyncio.run(
                _async_serialize(
                    jnp.array(1),
                    {},
                    futures.Future(),
                    limiter=serialization._LimitInFlightBytes(1),
                    max_data_shard_degree=-1,
                    shard_threshold_bytes=0,
                ),
                debug=True,
            )

    def _create_partially_replicated_array(self, sharded: bool):
        single_device_arr = jnp.arange(0, 1023 * 1024).reshape(1023, 1024)
        if sharded:
            if jax.device_count() != 8 or jax.process_count() != 1:
                self.skipTest("Incorrect device count for mesh.")
            devices = mesh_utils.create_device_mesh((8,))
            mesh = jax.sharding.Mesh(devices.reshape((4, 2)), ("x", "y"))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "y"))
            arr = jax.device_put(single_device_arr, sharding)
            return arr
        return single_device_arr

    @parameterized.parameters(False, True)
    def test_async_serialize_d2h_sync(self, sharded):
        arr = self._create_partially_replicated_array(sharded)

        ts_open_handle: Any = None
        old_open = array_serialization.ts.open

        async def ts_open_patch(*args, **kwargs):
            nonlocal ts_open_handle
            ts_open_handle = await old_open(*args, **kwargs)
            return ts_open_handle

        def jit_fn(arr):
            return arr + 1

        # Use AOT to reduce jit start time.
        jit_fn = jax.jit(jit_fn, donate_argnums=0).lower(arr).compile()

        old_transfer = _transfer_to_host

        # Delay execution by 1 second.
        def transfer_to_host_patch(*args, **kwargs):
            time.sleep(1)
            return old_transfer(*args, **kwargs)

        d2h_future = array_serialization.futures.Future()
        with mock.patch(
            f"{array_serialization.__name__}.ts.open",
            ts_open_patch,
        ), get_tensorstore_spec(arr) as spec, mock.patch(
            f"{array_serialization.__name__}._transfer_to_host", transfer_to_host_patch
        ):
            # Either RuntimeError(Array has been deleted with shape) or
            # ValueError(...Buffer has been deleted or donated...) may occur.
            with pytest.raises((RuntimeError, ValueError), match=re.escape("delete")):
                f = _CommitFuture(
                    _run_serializer(
                        [arr],
                        [spec],
                        [d2h_future],
                        max_concurrent_bytes=arr.nbytes,
                        max_data_shard_degree=-1,
                        shard_threshold_bytes=-1,
                    )
                )
                # Throws Array deleted exception if not waiting for d2h_future.
                jit_fn(arr)
                f.result()

        arr = self._create_partially_replicated_array(sharded)
        arr_host = jax.device_get(arr)
        d2h_future = array_serialization.futures.Future()
        with mock.patch(
            f"{array_serialization.__name__}.ts.open",
            ts_open_patch,
        ), get_tensorstore_spec(arr) as spec, mock.patch(
            f"{array_serialization.__name__}._transfer_to_host", transfer_to_host_patch
        ):
            f = _CommitFuture(
                _run_serializer(
                    [arr],
                    [spec],
                    [d2h_future],
                    max_concurrent_bytes=arr.nbytes,
                    max_data_shard_degree=-1,
                    shard_threshold_bytes=-1,
                )
            )
            d2h_future.result()
            # If D2H is finished, arr can be safely donated.
            jit_fn(arr)
            f.result()

            # Verify serialization result is correct.
            # pylint: disable-next=unsubscriptable-object
            arr_data = ts_open_handle[:].read().result()
            self.assertTrue(np.all(arr_data == arr_host))

    @parameterized.parameters(False, True)
    def test_async_serialize_exception_safety(self, sharded):
        arr = self._create_partially_replicated_array(sharded)

        async def ts_open_patch(*_, **__):
            raise RuntimeError("Test")

        d2h_future = array_serialization.futures.Future()
        with mock.patch(
            f"{array_serialization.__name__}.ts.open",
            ts_open_patch,
        ), get_tensorstore_spec(arr) as spec:
            f = _CommitFuture(
                _run_serializer(
                    [arr], [spec], [d2h_future], max_data_shard_degree=-1, shard_threshold_bytes=-1
                )
            )
            d2h_future.result()
            with pytest.raises(RuntimeError, match=re.escape("Test")):
                f.result()

        def transfer_to_host_patch(*_):
            raise RuntimeError("Test")

        d2h_future = array_serialization.futures.Future()
        with mock.patch(
            f"{array_serialization.__name__}._transfer_to_host",
            transfer_to_host_patch,
        ), get_tensorstore_spec(arr) as spec:
            f = _CommitFuture(
                _run_serializer(
                    [arr], [spec], [d2h_future], max_data_shard_degree=-1, shard_threshold_bytes=-1
                )
            )
            # Exceptions will be raised in both the d2h future and the commit future.
            with pytest.raises(RuntimeError, match=re.escape("Test")):
                d2h_future.result()
            with pytest.raises(RuntimeError, match=re.escape("Test")):
                f.result()

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
        self, arrays: list[list[int]], max_concurrent_gb: int, expect_max_concurrent_gb: int
    ):
        arrays = [
            mock.Mock(
                addressable_shards=[
                    mock.Mock(
                        replica_id=0, **{"data.nbytes": int(shard * 10**9), "data.shape": ()}
                    )
                    for shard in array
                ],
                nbytes=int(sum(array) * 10**9),
                dtype=jax.numpy.bfloat16,
            )
            for array in arrays
        ]
        tensorstore_specs = [{} for _ in range(len(arrays))]
        expect_max_concurrent_bytes = int(expect_max_concurrent_gb * 10**9)

        concurrent_bytes = 0

        class FakeTs:
            def __getitem__(self, *_):
                return self

            async def write(self, data: jax.Array, **_):
                await asyncio.sleep(0.1)
                nonlocal concurrent_bytes
                concurrent_bytes -= data.nbytes

        async def open_patch(*_, **__):
            return FakeTs()

        async def _copy_to_host_patch(shard_infos: list[_ShardInfo]):
            nonlocal concurrent_bytes
            for info in shard_infos:
                concurrent_bytes += info.data.nbytes
            # In-flight bytes should be lower than the expected max bytes
            self.assertLessEqual(concurrent_bytes, expect_max_concurrent_bytes)

        manager = BoundedDataShardedAsyncCheckpointManager(max_concurrent_gb=max_concurrent_gb)
        with (
            mock.patch(
                f"{array_serialization.__name__}._num_replicas_per_shard",
                lambda *args: defaultdict(lambda: 1),
            ),
            mock.patch(f"{array_serialization.__name__}._slices_to_tuple", lambda *_: 1),
            mock.patch(
                f"{array_serialization.__name__}._slice_shard_and_copy_to_host", _copy_to_host_patch
            ),
            mock.patch(
                f"{array_serialization.__name__}.serialization._get_metadata", lambda *_: {}
            ),
            mock.patch(f"{array_serialization.__name__}.ts.open", open_patch),
            mock.patch(f"{array_serialization.__name__}.ts.Spec", mock.MagicMock()),
        ):
            manager.serialize(arrays, tensorstore_specs, on_commit_callback=lambda: None)
            manager.wait_until_finished()

    @parameterized.product(
        arrays=[[jnp.arange(0, 1024 * 1024)]],
        max_concurrent_gb=[1],
        load_to_pinned_host=[True, False],
        jax_platforms=["proxy", "gpu", "tpu", "cpu"],
        enable_gcs_grpc=["true", "false"],
    )
    def test_deserialize(
        self,
        arrays: list[list[int]],
        max_concurrent_gb: int,
        load_to_pinned_host: bool,
        jax_platforms: str,
        enable_gcs_grpc: str,
    ):
        if jax.device_count() != 8 or jax.process_count() != 1 or jax.default_backend() == "cpu":
            self.skipTest(
                "Skip preloading check if device is CPU since CPU does not support "
                + "pinned host memory kind."
            )

        devices = mesh_utils.create_device_mesh((8,))
        mesh = jax.sharding.Mesh(devices, "x")
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("x",)))
        data = [jax.device_put(arr, sharding) for arr in arrays]

        # create a temporary tensorstore spec for the arrays
        @contextmanager
        def get_tensorstore_spec_for_deserialization(arrays: list[jax.Array]):
            tempdir = tempfile.TemporaryDirectory()
            kvstore_spec = {
                "driver": "gcs",
                "bucket": "fake-bucket",
                "path": f"fake-path/{time.time()}",
            }

            create_spec = lambda arr: {
                "driver": "zarr",
                "kvstore": kvstore_spec,
                "dtype": str(arr.dtype),
                "create": True,
                "delete_existing": True,
            }

            try:
                # Yield the temp path so our mock can redirect GCS writes to a local file.
                yield [create_spec(arr) for arr in arrays], tempdir.name
            finally:
                tempdir.cleanup()

        manager = BoundedDataShardedAsyncCheckpointManager(max_concurrent_gb=max_concurrent_gb)

        captured_specs = []
        original_ts_open = array_serialization.ts.open

        async def mock_ts_open(spec_arg, *args, **kwargs):
            # Convert spec to dict for inspection, as it can be a ts.Spec object.
            spec_dict = spec_arg.to_json() if isinstance(spec_arg, ts.Spec) else spec_arg
            captured_specs.append(spec_dict)

            # Create a copy to modify for the underlying call.
            spec_for_call = spec_dict.copy()
            # If it's a gcs spec (from serialize) or gcs_grpc (from deserialize),
            # redirect it to a functional file backend to allow the test to complete.
            if spec_for_call.get("kvstore", {}).get("driver") in ("gcs", "gcs_grpc"):
                spec_for_call["kvstore"] = {"driver": "file", "path": temp_path}

            # Re-wrap if the original was a ts.Spec object.
            call_arg = ts.Spec(spec_for_call) if isinstance(spec_arg, ts.Spec) else spec_for_call
            return await original_ts_open(call_arg, *args, **kwargs)

        # Write the data to local files
        with get_tensorstore_spec_for_deserialization(data) as (
            tensorstore_spec,
            temp_path,
        ), mock.patch(
            f"{array_serialization.__name__}.{_ts_open}", new=mock_ts_open
        ), mock.patch.dict(
            "os.environ", {"JAX_PLATFORMS": jax_platforms, "ENABLE_GCS_GRPC": enable_gcs_grpc}
        ):
            manager.serialize(
                data,
                tensorstore_spec,
                on_commit_callback=lambda: None,
            )
            manager.wait_until_finished()

            # Deserialize the data
            load_sharding = [
                sharding.with_memory_kind("pinned_host") if load_to_pinned_host else sharding
            ]
            deserialized_data = manager.deserialize(
                load_sharding,
                tensorstore_spec,
            )

        # Assert that the spec and context passed to ts.open during deserialize was modified.
        # The calls for deserialize are the last ones captured.
        deserialize_specs = captured_specs[-len(data) :]
        self.assertGreater(len(deserialize_specs), 0)
        for spec in deserialize_specs:
            # gcs_grpc driver should be used when either:
            # - JAX_PLATFORMS is "proxy" (running on GCP/Pathways), OR
            # - ENABLE_GCS_GRPC is "true" (explicit opt-in)
            # Otherwise, gcs driver should be used (no change)
            should_use_grpc = jax_platforms == "proxy" or enable_gcs_grpc == "true"
            expected_driver = "gcs_grpc" if should_use_grpc else "gcs"
            self.assertEqual(spec["kvstore"]["driver"], expected_driver)
        # Verify the deserialized data matches the original data
        self.assertEqual(len(data), len(deserialized_data))
        for arr, d_arr in zip(data, deserialized_data):
            self.assertEqual(arr.shape, d_arr.shape)
            self.assertEqual(arr.dtype, d_arr.dtype)
            # Check that the data matches
            np.testing.assert_array_equal(arr, d_arr)

            if load_to_pinned_host:
                # Verify that the data is loaded to pinned host memory
                self.assertTrue(d_arr.sharding.memory_kind == "pinned_host")

    def test_max_concurrency(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            BoundedDataShardedAsyncCheckpointManager(max_concurrent_gb=0)

    def test_serialize_consecutive(self):
        """Tests that serialize waits for prior serialization to finish."""
        manager = BoundedDataShardedAsyncCheckpointManager(max_concurrent_gb=1)
        mock_serialize = mock.patch(
            f"{array_serialization.__name__}._async_serialize", return_value=mock.AsyncMock()
        )
        with mock_serialize, mock.patch.object(manager, "wait_until_finished") as mock_wait:
            manager.serialize([], [], on_commit_callback=lambda *args: None)
            self.assertTrue(mock_wait.called)

    def test_zero_copy_d2h(self):
        @functools.partial(jax.jit, donate_argnums=0)
        def _donate_argnum_fn(x):
            return x + 1

        x = jnp.arange(0, 10)
        x_np = np.arange(0, 10)
        x.copy_to_host_async()
        x_zero_copy = np.array(x, copy=False)
        y = _donate_argnum_fn(x)
        y.copy_to_host_async()
        # Test if x_zero_copy still lives after original x is donated.
        self.assertTrue(np.all(x_zero_copy == x_np))

    def _verify_shard_info(
        self,
        single_device_arr: jax.Array,
        arr: jax.Array,
        max_data_shard_degree: int,
        shard_threshold_bytes: int,
    ):
        shard_infos = _get_shard_infos(
            arr,
            max_data_shard_degree=max_data_shard_degree,
            shard_threshold_bytes=shard_threshold_bytes,
        )

        # Write each shard to output and check if it's the same as the original
        # single device array. If same, that means all shards should cover all
        # indices of the original array.
        out_array = np.empty_like(single_device_arr)
        asyncio.run(_slice_shard_and_copy_to_host(shard_infos))
        for info in shard_infos:
            out_array[info.index] = info.data
        self.assertTrue(np.all(out_array == np.array(single_device_arr)))

    @parameterized.product(
        max_data_shard_degree=[1, -1, 2, 4, 8], shard_threshold_bytes=[1000 * 1000 * 1000, 1]
    )
    def test_shard_info_partially_replicated(
        self, max_data_shard_degree: int, shard_threshold_bytes: int
    ):
        if jax.device_count() != 8 or jax.process_count() != 1:
            self.skipTest("Incorrect device count for mesh.")

        single_device_arr = jnp.arange(0, 1024 * 1024).reshape(1024, 1024)
        devices = mesh_utils.create_device_mesh((8,))
        mesh = jax.sharding.Mesh(devices.reshape((4, 2)), ("x", "y"))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "y"))
        arr = jax.device_put(single_device_arr, sharding)

        replica_count = _num_replicas_per_shard(arr)
        self.assertEqual(replica_count[((None, None, None), (0, 512, None))], 4)
        self.assertEqual(replica_count[((None, None, None), (512, 1024, None))], 4)

        self._verify_shard_info(
            single_device_arr, arr, max_data_shard_degree, shard_threshold_bytes
        )

    @parameterized.product(
        max_data_shard_degree=[1, -1, 2, 4, 8], shard_threshold_bytes=[1000 * 1000 * 1000, 1]
    )
    def test_shard_info_fully_sharded(self, max_data_shard_degree: int, shard_threshold_bytes: int):
        if jax.device_count() != 8 or jax.process_count() != 1:
            self.skipTest("Incorrect device count for mesh.")
        single_device_arr = jnp.arange(0, 1024 * 1024).reshape(1024, 1024)
        devices = mesh_utils.create_device_mesh((8,))
        mesh = jax.sharding.Mesh(devices.reshape((4, 2)), ("x", "y"))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y"))
        arr = jax.device_put(single_device_arr, sharding)

        replica_count = _num_replicas_per_shard(arr)
        self.assertEqual(replica_count[((0, 256, None), (0, 512, None))], 1)

        self._verify_shard_info(
            single_device_arr, arr, max_data_shard_degree, shard_threshold_bytes
        )

    @parameterized.product(
        sz=[1, 11, 16, 21],
        max_data_shard_degree=[1, -1, 2, 4, 8],
        shard_threshold_bytes=[1000 * 1000 * 1000, 1],
    )
    def test_shard_info_fully_replicated(
        self, sz: int, max_data_shard_degree: int, shard_threshold_bytes: int
    ):
        if jax.device_count() != 8 or jax.process_count() != 1:
            self.skipTest("Incorrect device count for mesh.")
        single_device_arr = jnp.arange(0, sz)
        devices = mesh_utils.create_device_mesh((8,))
        mesh = jax.sharding.Mesh(devices, "x")
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
        arr = jax.device_put(single_device_arr, sharding)

        replica_count = _num_replicas_per_shard(arr)
        # Fully replicated on 8 devices.
        self.assertEqual(replica_count[((None, None, None),)], 8)

        self._verify_shard_info(
            single_device_arr, arr, max_data_shard_degree, shard_threshold_bytes
        )

    @parameterized.parameters(
        dict(
            index=(slice(2, 4, None), slice(None, None, None)),
            expected="1.0",
        ),
        dict(
            index=(slice(None, None, None), slice(2, 3, None), slice(None, None, None)),
            expected="0.2.0",
        ),
        dict(
            index=(),  # Scalar.
            expected="0",
        ),
        dict(
            index=(slice(None, None, None),),  # Replicated.
            expected="0",
        ),
    )
    def test_shard_coordinate(self, index, expected):
        data = jnp.zeros(())
        self.assertEqual(
            _ShardInfo(data=data, index=index, slice_arg=None, replica_count=1).shard_coordinate(),
            expected,
        )


if __name__ == "__main__":
    absltest.main()
