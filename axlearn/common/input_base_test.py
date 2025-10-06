# Copyright © 2024 Apple Inc.

"""Tests base Input interface."""

from functools import partial
from typing import Callable, Union

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental.pjit import pjit

from axlearn.common.input_base import Input, PathAndRank, partition_by_path_rank
from axlearn.common.input_dispatch import BaseInputDispatcher, InputDispatcher, SpmdInputDispatcher
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
                "sharding={devices=[4,2]<=[8]}": 2,
                "sharding={devices=[4,2]<=[8] last_tile_dim_replicate}": 1,
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
                "sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}": 1,
                "sharding={devices=[4,2]<=[8]}": 1,
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
                "sharding={devices=[4,2]<=[8]}": 1,
                "sharding={devices=[4,2]<=[8] last_tile_dim_replicate}": 1,
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
    def test_partition_by_path_rank(self, config: dict, expected: Union[dict, Exception]):
        if len(jax.devices()) != 8:
            self.skipTest("Test requires 8 devices")
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }
        partition_fn = partition_by_path_rank(config)

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(4, 2)[..., None],
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
        jax.tree.map(check_sharding, tree_paths(output_batch), output_batch)
        return output_batch

    fn.lower(input_batch).compile()


class InputTest(TestCase):
    """Tests Input."""

    def test_dispatch_global_batch(self):
        if len(jax.devices()) != 8:
            self.skipTest("Test requires 8 devices")
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
        }
        input_cfg = Input.default_config().set(
            name="test",
            partition_spec=PartitionSpec("data"),
        )

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(4, 2)[..., None],
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

    @parameterized.parameters(
        dict(
            dispatcher=SpmdInputDispatcher,
            # No change with SpmdInputDispatcher (it directly maps logical to logical).
            expected={
                "x": jax.ShapeDtypeStruct((4, 8), dtype=jnp.int32),
                "y": jax.ShapeDtypeStruct((4,), dtype=jnp.int32),
            },
        ),
        dict(
            dispatcher=InputDispatcher,
            # InputDispatcher pads to the feed physical shape.
            expected={
                "x": jax.ShapeDtypeStruct((8, 8), dtype=jnp.int32),
                "y": jax.ShapeDtypeStruct((8,), dtype=jnp.int32),
            },
        ),
    )
    def test_element_spec(
        self,
        dispatcher: type[BaseInputDispatcher],
        expected: Nested[jax.ShapeDtypeStruct],
    ):
        if len(jax.devices()) != 8:
            self.skipTest("Test requires 8 devices")
        input_cfg: Input.Config = Input.default_config().set(
            name="test",
            partition_spec=PartitionSpec("data"),
            input_dispatcher=dispatcher.default_config().set(
                global_logical_batch_size=4,
            ),
        )

        with jax.make_mesh((4, 2), ("data", "model")):
            ds: Input = input_cfg.instantiate(parent=None)

            element_spec = {
                "x": jax.ShapeDtypeStruct((4, 8), dtype=jnp.int32),
                "y": jax.ShapeDtypeStruct((4,), dtype=jnp.int32),
            }
            self.assertNestedEqual(
                expected, ds.input_dispatcher.logical_to_physical_shapes(element_spec)
            )


if __name__ == "__main__":
    absltest.main()
