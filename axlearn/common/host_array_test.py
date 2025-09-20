# Copyright © 2023 Apple Inc.

"""Tests global-host array conversions.

Some tests are intended to be run on TPU.
"""

import functools
import math
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl import logging
from absl.testing import absltest, parameterized
from jax._src.sharding_impls import get_process_index_and_count, local_to_global_shape
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

from axlearn.common.test_utils import TestCase, is_supported_mesh_shape, is_supported_platform
from axlearn.common.utils import (
    DataPartitionType,
    MeshShape,
    data_partition_type_to_spec,
    flatten_items,
    global_to_host_array,
    host_to_global_device_array,
    infer_mesh_shape,
    replicate_to_local_data,
)


def _infer_num_partitions(
    shape: Sequence[int],
    mesh: Mesh,
    partition: Union[PartitionSpec, DataPartitionType],
) -> Sequence[int]:
    """Returns the number of partitions along each dim."""

    num_partitions = []
    if isinstance(partition, PartitionSpec):
        for dim_partition in partition:
            axes = jax.tree.leaves(dim_partition)
            num_partitions.append(math.prod(mesh.shape[axis] for axis in axes))
    elif partition == DataPartitionType.FULL:
        num_partitions.append(jax.device_count())
    elif partition != DataPartitionType.REPLICATED:
        raise NotImplementedError(partition)

    # Complete the partitions.
    num_partitions += [1] * (len(shape) - len(num_partitions))
    assert len(num_partitions) == len(shape)
    return num_partitions


def _is_supported(*, platform: str, mesh_shape: MeshShape) -> bool:
    return is_supported_platform(platform) and is_supported_mesh_shape(mesh_shape)


def _ordered_devices(mesh_shape: MeshShape, process_shape: MeshShape) -> np.ndarray:
    """Returns devices of shape `mesh_shape` with consistent host ordering.

    `process_shape` indicates how the hosts should be laid out. For example, if `mesh_shape` is
    (4,4) and `process_shape` is (2,2), the top-left quadrant will be assigned device IDs from
    process 0, the top-right quadrant from process 1, etc.
    """
    assert len(mesh_shape) == len(process_shape), "ndim should match"

    process_count = math.prod(jax.tree.leaves(process_shape))
    process_index_to_device = [[] for _ in range(process_count)]
    for x in jax.devices():
        process_index_to_device[x.process_index].append(x)
    devices = np.full(mesh_shape, None)
    per_process_shape = tuple(mesh_shape[i] // process_shape[i] for i in range(len(mesh_shape)))
    for i, j in np.ndindex(mesh_shape):
        r, c = i // per_process_shape[0], j // per_process_shape[1]
        process_index = r * 2 + c
        devices[i, j] = process_index_to_device[process_index].pop()
    return devices.reshape(mesh_shape)


# TODO(markblee): Consolidate with utils_test.
class HostArrayTest(TestCase):
    @parameterized.product(
        platform=("cpu", "tpu"),
        mesh_shape=[
            (-1, 1),  # Fully partitioned along one dim.
            (2, -1),  # Partitioned along multiple dims.
            (-1, 2),  # Test the other way.
            (1, -1),
        ],
        process_shape=[
            # Each process produces single dim.
            [1],  # Not divisible by number of devices (replicated).
            [8],  # Divisible by number of devices.
            [16],  # Multiple elements per device.
            # Each process produces multiple dims.
            [1, 1],  # Not divisible by number of devices (replicated).
            [2, 1],  # Can be partitioned over dim=0, replicated on dim=1.
            [16, 1],  # Multiple elements per device.
            [2, 4],  # Can be fully partitioned.
            [8, 8],  # Can be fully partitioned.
        ],
        partition=(
            DataPartitionType.FULL,
            DataPartitionType.REPLICATED,
            PartitionSpec("data"),
            PartitionSpec("data", "model"),
        ),
    )
    # NOTE: while annotated with `for_8_devices`, this runs on other configurations.
    @pytest.mark.for_8_devices
    def test_fixed_process_shape(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        process_shape: Sequence[int],
        partition: Union[DataPartitionType, PartitionSpec],
    ):
        """Tests roundtrip host-to-global and global-to-host with fixed process shape."""

        mesh_shape = infer_mesh_shape(mesh_shape)
        if not _is_supported(platform=platform, mesh_shape=mesh_shape):
            self.skipTest("Unsupported platform/mesh.")

        devices = mesh_utils.create_device_mesh(mesh_shape, allow_split_physical_axes=True)
        mesh = jax.sharding.Mesh(devices, ("data", "model"))

        partition = data_partition_type_to_spec(partition)
        sharding = jax.NamedSharding(mesh, partition)

        # Number of dims should match number of partitioned axes.
        if len(process_shape) < len(partition):
            self.skipTest("Incompatible process_shape/partition.")

        # Infer global shape from local_shape and number of processes.
        global_shape = local_to_global_shape(sharding, process_shape)
        # Partition should divide global_shape evenly.
        partitions = _infer_num_partitions(global_shape, mesh=mesh, partition=partition)
        if any(dim % num_parts != 0 for dim, num_parts in zip(global_shape, partitions)):
            self.skipTest("Incompatible global_shape/partitioning.")

        with mesh:
            host_arrays = dict(
                x=jax.random.uniform(jax.random.PRNGKey(jax.process_count()), shape=process_shape)
            )

            global_arrays = host_to_global_device_array(host_arrays, partition=partition)
            for path, value in flatten_items(global_arrays):
                self.assertEqual(tuple(global_shape), value.shape, msg=path)
            global_arrays["y"] = 2 * global_arrays["x"]
            restored_host_arrays = global_to_host_array(global_arrays)
            for path, restored_value in flatten_items(restored_host_arrays):
                self.assertEqual(tuple(process_shape), restored_value.shape, msg=path)

            # "x" and "y" are partitioned consistently.
            np.testing.assert_array_equal(restored_host_arrays["y"], 2 * restored_host_arrays["x"])

            # Check round-trip equality of host_to_global_device_array and global_to_host_array.
            np.testing.assert_array_equal(host_arrays["x"], restored_host_arrays["x"])

    @parameterized.product(
        platform=["cpu", "tpu"],
        mesh_shape=[(1, 1), (-1, 1), (1, -1), (-1, 2), (2, -1)],
        global_shape=[[1], [16], [8, 8], [16, 2]],
        partition=[
            DataPartitionType.FULL,
            DataPartitionType.REPLICATED,
            PartitionSpec("data"),
            PartitionSpec("data", "model"),
        ],
    )
    # NOTE: while annotated with `for_8_devices`, this runs on other configurations.
    @pytest.mark.for_8_devices
    def test_fixed_global_shape(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        global_shape: Sequence[int],
        partition: Union[PartitionSpec, DataPartitionType],
    ):
        """Tests roundtrip host-to-global and global-to-host with fixed global shape."""

        mesh_shape = infer_mesh_shape(mesh_shape)
        if not _is_supported(platform=platform, mesh_shape=mesh_shape):
            self.skipTest("Unsupported platform/mesh.")
        logging.info(
            "platform=%s mesh_shape=%s global_shape=%s data_partition=%s",
            platform,
            mesh_shape,
            global_shape,
            partition,
        )
        devices = mesh_utils.create_device_mesh(mesh_shape, allow_split_physical_axes=True)
        mesh = jax.sharding.Mesh(devices, ("data", "model"))
        logging.info("Global mesh: %s", mesh)

        partition = data_partition_type_to_spec(partition)
        # Number of dims should match number of partitioned axes.
        if len(global_shape) < len(partition):
            self.skipTest("Incompatible process_shape/partition.")

        partitions = _infer_num_partitions(global_shape, mesh=mesh, partition=partition)
        if any(dim % num_parts != 0 for dim, num_parts in zip(global_shape, partitions)):
            self.skipTest("Incompatible global_shape/partitioning.")

        with mesh:
            sharding = jax.sharding.NamedSharding(mesh, partition)

            ndim = len(global_shape)
            process_shape = []
            for dim in range(ndim):
                _, num_shards = get_process_index_and_count(sharding, dim=dim, ndims=ndim)
                process_shape.append(global_shape[dim] // num_shards)

            host_arrays = dict(
                x=jax.random.uniform(jax.random.PRNGKey(jax.process_index()), shape=process_shape)
            )
            global_arrays = host_to_global_device_array(host_arrays, partition=partition)
            for path, value in flatten_items(global_arrays):
                self.assertEqual(tuple(global_shape), value.shape, msg=path)
            global_arrays["y"] = 2 * global_arrays["x"]
            restored_host_arrays = global_to_host_array(global_arrays, partition=partition)
            for path, restored_value in flatten_items(restored_host_arrays):
                restored_shape = restored_value.shape
                self.assertEqual(tuple(process_shape), restored_shape, msg=path)

            # "x" and "y" are partitioned consistently.
            np.testing.assert_array_equal(restored_host_arrays["y"], 2 * restored_host_arrays["x"])

            # Check round-trip equality of host_to_global_device_array and global_to_host_array.
            np.testing.assert_array_equal(host_arrays["x"], restored_host_arrays["x"])

    @parameterized.product(
        platform=["cpu", "tpu"],
        mesh_shape=[(-1, 2), (2, -1)],
        process_shape=[(2, 1), (4, 4), (1, 8)],
    )
    @pytest.mark.tpu
    def test_host_to_global_multiple_dims(
        self, platform: str, mesh_shape: tuple, process_shape: tuple
    ):
        """Test a case where we form the global batch with uniform sharding over multiple dims."""

        mesh_shape = infer_mesh_shape(mesh_shape)
        if not _is_supported(platform=platform, mesh_shape=mesh_shape):
            self.skipTest("Unsupported platform/mesh.")

        device_count = jax.device_count()
        process_count = jax.process_count()
        print(f"{device_count=}, {process_count=}")
        assert device_count > 1

        if process_count % 2 != 0:
            self.skipTest("Requires even number of processes.")

        # Tile the global array across processes (N/2, 2).
        # Process i gets (i//2, i%2).
        process_arrays = [
            [
                jax.random.uniform(jax.random.PRNGKey(i), shape=process_shape),
                jax.random.uniform(jax.random.PRNGKey(i + 1), shape=process_shape),
            ]
            for i in range(0, process_count, 2)
        ]
        global_array = np.concatenate(
            [np.concatenate(row, axis=1) for row in process_arrays], axis=0
        )

        # Group devices so that hosts are tiled in the same way as process_arrays.
        # This allows us to later replicate the global array with the same host-ordering for
        # comparison purposes.
        devices = _ordered_devices(mesh_shape, (process_count // 2, 2))
        mesh = jax.sharding.Mesh(devices.reshape(mesh_shape), ("x", "y"))

        # Shard both dims uniformly.
        partition = PartitionSpec("x", "y")
        process_index = jax.process_index()

        # Partition should divide global_shape evenly.
        partitions = _infer_num_partitions(global_array.shape, mesh=mesh, partition=partition)
        if any(dim % num_parts != 0 for dim, num_parts in zip(global_array.shape, partitions)):
            self.skipTest("Incompatible global_shape/partitioning.")

        with mesh:
            # Each process has a slice.
            local_x = process_arrays[process_index // 2][process_index % 2]
            batch = host_to_global_device_array(local_x, partition=partition)

            # Check that sharding is as expected.
            self.assertEqual(partition, batch.sharding.spec)
            # Check the shape is expected.
            self.assertEqual(batch.shape, global_array.shape)
            # Check that contents are as expected.
            restored_array = replicate_to_local_data(batch)
            self.assertNestedEqual(global_array, restored_array)

    @parameterized.product(
        platform=["cpu", "tpu"],
        mesh_shape=[(-1, 2), (2, -1)],
        process_shapes=[
            {"x": (2, 1)},
            {"x": (4, 4)},
            {"x": (1, 8)},
            {"x": (8, 4), "y": (4, 8)},  # Test a case with mixed shapes.
        ],
    )
    @pytest.mark.tpu
    def test_global_to_host_multiple_dims(
        self, platform: str, mesh_shape: tuple, process_shapes: dict[str, tuple]
    ):
        """Test a case where we form the global batch with uniform sharding over multiple dims."""

        mesh_shape = infer_mesh_shape(mesh_shape)
        if not _is_supported(platform=platform, mesh_shape=mesh_shape):
            self.skipTest("Unsupported platform/mesh.")

        device_count = jax.device_count()
        process_count = jax.process_count()
        print(f"{device_count=}, {process_count=}")
        assert device_count > 1

        if process_count % 2 != 0:
            self.skipTest("Requires even number of processes.")

        # Build an array that can be sharded over multiple dims.
        process_arrays = {}
        for k, shape in process_shapes.items():
            process_arrays[k] = [
                [
                    jax.random.uniform(jax.random.PRNGKey(i), shape=shape),
                    jax.random.uniform(jax.random.PRNGKey(i + 1), shape=shape),
                ]
                for i in range(0, process_count, 2)
            ]
        global_arrays = {}
        for k, arrays in process_arrays.items():
            global_arrays[k] = jnp.concatenate(
                [jnp.concatenate(row, axis=1) for row in arrays], axis=0
            )

        # Build a mesh with consistent host tiling.
        devices = _ordered_devices(mesh_shape, (process_count // 2, 2))
        mesh = jax.sharding.Mesh(devices.reshape(mesh_shape), ("x", "y"))

        # Shard both dims uniformly.
        partition = PartitionSpec("x", "y")

        # Partition should divide global_shape evenly.
        for k, global_array in global_arrays.items():
            partitions = _infer_num_partitions(global_array.shape, mesh=mesh, partition=partition)
            if any(dim % num_parts != 0 for dim, num_parts in zip(global_array.shape, partitions)):
                self.skipTest("Incompatible global_shape/partitioning.")

        process_index = jax.process_index()

        with mesh:
            # Shard both dims uniformly.
            out_sharding = jax.NamedSharding(mesh, partition)

            @functools.partial(jax.jit, in_shardings=None, out_shardings=out_sharding)
            def shard(x):
                return x

            sharded_global_arrays = shard(global_arrays)
            local_arrays = global_to_host_array(sharded_global_arrays)
            for k, local_array in local_arrays.items():
                self.assertEqual(local_array.shape, process_shapes[k])
                self.assertNestedAllClose(
                    local_array, process_arrays[k][process_index // 2][process_index % 2]
                )
            restored_arrays = host_to_global_device_array(local_arrays, partition=out_sharding.spec)
            self.assertNestedAllClose(replicate_to_local_data(restored_arrays), global_arrays)


if __name__ == "__main__":
    absltest.main()
