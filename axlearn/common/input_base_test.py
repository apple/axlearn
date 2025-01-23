# Copyright © 2024 Apple Inc.

"""Tests base Input interface."""

from functools import partial
from typing import Callable, Union

import jax
import jax.test_util
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp
from jax.experimental.pjit import pjit

from axlearn.common.input_base import Input, PathAndRank, partition_by_path_rank
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Nested, PartitionSpec, Tensor, tree_paths


class PartitionByPathAndRankTest(TestCase):
    """Tests partition_by_path_rank."""

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
                ("target_num_.*", None): PartitionSpec.UNCONSTRAINED,
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
                (None, 2): PartitionSpec.UNCONSTRAINED,
            },
            expected={
                # target_labels over (data, seq), target_num_bytes over (data).
                'custom_call_target="Sharding"': 2,
                "sharding={devices=[2,2]<=[4]}": 1,
                "sharding={devices=[2,2]<=[4] last_tile_dim_replicate}": 1,
            },
        ),
        dict(
            config={
                # Leave all inputs unconstrained.
                (None, None): PartitionSpec.UNCONSTRAINED
            },
            expected={
                'custom_call_target="Sharding"': 0,
            },
        ),
        dict(
            config={},
            expected=ValueError("No rules matched"),
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
    def test_partition_by_path_rank(self, config: dict, expected: Union[dict, Exception]):
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }
        partition_fn = partition_by_path_rank(config)

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 2)[..., None],
            axis_names=("data", "seq", "model"),
        ):
            if isinstance(expected, Exception):
                with self.assertRaisesRegex(type(expected), str(expected)):
                    partition_fn(input_batch)
                return

            # Check that a copy is returned.
            self.assertIsNot(input_batch, partition_fn(input_batch))

            fn = pjit(partition_fn, in_shardings=None)
            hlo_text = fn.lower(input_batch).compiler_ir(dialect="hlo").as_hlo_text()

            for pattern, count in expected.items():
                self.assertEqual(count, hlo_text.count(pattern), msg=f"{pattern=},{count=}")


def dispatch_and_check_sharding(
    cfg: Input.Config,
    *,
    input_batch: Nested[Tensor],
    callback: Callable[[str, jax.sharding.Sharding], None],
):
    """Instantiates an input, dispatches the input batch, and invokes callback with the sharding.

    The callback is invoked with (path: str, sharding: Sharding).
    """
    ds: Input = cfg.instantiate(parent=None)

    def check_sharding(path, value):
        jax.debug.inspect_array_sharding(value, callback=lambda sharding: callback(path, sharding))

    @partial(pjit, in_shardings=None)
    def fn(input_batch):
        output_batch = ds.dispatch_global_batch(input_batch)
        jax.tree_map(check_sharding, tree_paths(output_batch), output_batch)
        return output_batch

    fn.lower(input_batch).compile()


class InputTest(TestCase):
    """Tests Input."""

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
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

            def dispatch_and_check(cfg: Input.Config, expected: dict):
                dispatch_and_check_sharding(
                    cfg,
                    input_batch=input_batch,
                    callback=lambda path, sharding: self.assertEqual(expected[path], sharding.spec),
                )

            # Without input partitioner, constrain along batch axis names.
            dispatch_and_check(
                input_cfg,
                expected={
                    "input_ids": PartitionSpec("data"),
                    "target_labels": PartitionSpec("data"),
                    "target_num_bytes": PartitionSpec("data"),
                },
            )

            # With input partitioner, a subset of keys can be further constrained.
            dispatch_and_check(
                input_cfg.set(
                    input_partitioner=partition_by_path_rank(
                        {
                            PathAndRank(".*", 2): PartitionSpec("data", "seq"),
                            PathAndRank("target_num_bytes", None): PartitionSpec.UNCONSTRAINED,
                        },
                    )
                ),
                expected={
                    "input_ids": PartitionSpec("data", "seq"),
                    "target_labels": PartitionSpec("data", "seq"),
                    "target_num_bytes": PartitionSpec("data"),
                },
            )
