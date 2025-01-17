# Copyright © 2024 Apple Inc.

"""Tests base Input interface."""

from functools import partial

import jax
import jax.test_util
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp
from jax.experimental.pjit import pjit

from axlearn.common.input_base import Input, partition_by_path_ndim
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import PartitionSpec, tree_paths


class PartitionByPathNdimTest(TestCase):
    """Tests partition_by_path_ndim."""

    @parameterized.parameters(
        dict(
            config={
                (".*", 2): PartitionSpec("data", "seq"),
                (".*", 1): PartitionSpec("data"),
            },
            expected={
                'custom_call_target="Sharding"': 3,
                "sharding={devices=[2,2]<=[4]}": 2,
                "sharding={devices=[2,2]<=[4] last_tile_dim_replicate}": 1,
            },
        ),
        dict(
            config={
                ("target_labels", 2): PartitionSpec("data", "seq"),
                ("input", 2): PartitionSpec("data", "seq"),  # Ignored: Not a fullmatch.
                (".*", 2): PartitionSpec("data"),
            },
            expected={
                # target_labels sharded over (data, seq), input_ids over (data).
                # target_num_bytes is not constrained.
                'custom_call_target="Sharding"': 2,
                "sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}": 1,
                "sharding={devices=[2,2]<=[4]}": 1,
            },
        ),
        dict(
            config={
                ("target_labels", 2): PartitionSpec("data", "seq"),
                (".*", 1): PartitionSpec("data"),
            },
            expected={
                # target_labels over (data, seq), target_num_bytes over (data).
                'custom_call_target="Sharding"': 2,
                "sharding={devices=[2,2]<=[4]}": 1,
                "sharding={devices=[2,2]<=[4] last_tile_dim_replicate}": 1,
            },
        ),
    )
    # TODO(markblee): Add a pytest marker for multi-device tests.
    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    def test_partition_by_path_ndim(self, config: dict, expected: dict):
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }
        partition_fn = partition_by_path_ndim(config)

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 2)[..., None],
            axis_names=("data", "seq", "model"),
        ):
            # Check that a copy is returned.
            self.assertIsNot(input_batch, partition_fn(input_batch))

            fn = pjit(partition_fn, in_shardings=None)
            hlo_text = fn.lower(input_batch).compiler_ir(dialect="hlo").as_hlo_text()

            for pattern, count in expected.items():
                self.assertEqual(count, hlo_text.count(pattern), msg=f"{pattern=},{count=}")


class InputTest(TestCase):
    """Tests Input."""

    def test_dispatch_global_batch(self):
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }
        input_cfg = Input.default_config().set(name="test")

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 2)[..., None],
            axis_names=("data", "seq", "model"),
        ):

            def dispatch_and_check(ds: Input, input_batch: dict, expected: dict):
                def check_sharding(path, value):
                    jax.debug.inspect_array_sharding(
                        value,
                        callback=lambda sharding: self.assertEqual(expected[path], sharding.spec),
                    )

                @partial(pjit, in_shardings=None)
                def fn(input_batch):
                    output_batch = ds.dispatch_global_batch(input_batch)
                    jax.tree_map(check_sharding, tree_paths(output_batch), output_batch)
                    return output_batch

                fn.lower(input_batch).compile()

            # Without input partitioner, constrain along batch axis names.
            x = input_cfg.instantiate(parent=None)
            dispatch_and_check(
                x,
                input_batch,
                expected={
                    "input_ids": PartitionSpec("data"),
                    "target_labels": PartitionSpec("data"),
                    "target_num_bytes": PartitionSpec("data"),
                },
            )

            # With input partitioner, a subset of keys can be further constrained.
            x = input_cfg.set(
                input_partitioner=partition_by_path_ndim({(".*", 2): PartitionSpec("data", "seq")})
            ).instantiate(parent=None)
            dispatch_and_check(
                x,
                input_batch,
                expected={
                    "input_ids": PartitionSpec("data", "seq"),
                    "target_labels": PartitionSpec("data", "seq"),
                    "target_num_bytes": PartitionSpec("data"),
                },
            )
