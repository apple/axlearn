# Copyright © 2023 Apple Inc.

"""Tests GDA-related functions in utils.py.

This is a separate test to avoid dependency on test_utils/torch.
Some tests are intended to be run on TPU.
"""

import itertools

import jax
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental import mesh_utils, pjit

from axlearn.common.test_utils import TestCase, is_supported_mesh_shape
from axlearn.common.utils import (
    DataPartitionType,
    Tensor,
    data_partition_type_to_spec,
    host_to_global_device_array,
)


class GDATest(TestCase):
    @parameterized.parameters(
        itertools.product(
            ((1, 1), (8, 1), (4, 2)),  # mesh_shape
            (1, 16),  # per_host_batch_size
            (DataPartitionType.FULL, DataPartitionType.REPLICATED),  # data_partition
        )
    )
    def test_host_array_to_gda(self, mesh_shape, per_host_batch_size, data_partition):
        logging.info(
            "mesh_shape=%s per_host_batch_size=%s data_partition=%s",
            mesh_shape,
            per_host_batch_size,
            data_partition,
        )
        if not is_supported_mesh_shape(mesh_shape):
            return
        devices = mesh_utils.create_device_mesh(mesh_shape)
        if data_partition == DataPartitionType.FULL:
            global_batch_size = per_host_batch_size * jax.process_count()
        else:
            assert data_partition == DataPartitionType.REPLICATED
            global_batch_size = per_host_batch_size
        if data_partition == DataPartitionType.FULL and global_batch_size < jax.device_count():
            return
        per_host_input_batch = dict(x=jnp.zeros((per_host_batch_size, 8), dtype=jnp.float32))
        with jax.sharding.Mesh(devices, ("data", "model")):
            global_input_batch = host_to_global_device_array(
                per_host_input_batch, partition=data_partition
            )
            self.assertIsInstance(global_input_batch["x"], Tensor)
            self.assertSequenceEqual(global_input_batch["x"].shape, (global_batch_size, 8))
            partition_spec = data_partition_type_to_spec(data_partition)
            pjit_fn = pjit.pjit(
                lambda x: x,
                in_shardings=(partition_spec,),
                out_shardings=partition_spec,
            )
            output = pjit_fn(global_input_batch)
            self.assertIsInstance(output["x"], Tensor)
            self.assertSequenceEqual(output["x"].shape, (global_batch_size, 8))


if __name__ == "__main__":
    absltest.main()
