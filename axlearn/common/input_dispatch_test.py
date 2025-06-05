# Copyright © 2024 Apple Inc.

"""Tests input dispatcher."""

import random

import jax
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common.input_dispatch import InputDispatcher, SpmdInputDispatcher
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import (
    PHYSICAL_TO_LOGICAL_DISPATCH_KEY,
    host_to_global_device_array,
    replicate_to_local_data,
    shapes,
)


class InputDispatcherTest(TestCase):
    @parameterized.parameters(
        # In the most common use cases, users only specify `global_logical_batch_size` and
        # `num_physical_feeds` and let InputDispatcher figure out `global_physical_batch_size`
        # and `logical_feed_indices` automatically.
        (8, None, 2, None),
        (2, None, 16, None),
        # With explicit `global_physical_batch_size`.
        (8, 16, 2, None),
        # With explicit `logical_feed_indices`.
        (8, 16, 2, (0,)),
        (8, 16, 4, (1, 3)),
        (2, 16, 16, (7, 11)),
        # Test a case where `feed_logical_batch_size` < `feed_physical_batch_size`.
        (2, 8, 2, (0,)),
        # Users can specify `global_logical_batch_size` only. In this case,
        # `global_physical_batch_size` is inferred as `device_count`.
        (8, None, None, None),
    )
    def test_input_dispatcher(
        self,
        global_logical_batch_size,
        global_physical_batch_size,
        num_physical_feeds,
        logical_feed_indices,
    ):
        all_physical_batches = []
        dispatcher: InputDispatcher = None
        inferred_num_physical_feeds = num_physical_feeds or jax.process_count()
        for physical_feed_index in range(inferred_num_physical_feeds):
            cfg: InputDispatcher.Config = InputDispatcher.default_config().set(
                global_logical_batch_size=global_logical_batch_size,
                global_physical_batch_size=global_physical_batch_size,
                num_physical_feeds=num_physical_feeds,
                physical_feed_index=physical_feed_index,
                logical_feed_indices=logical_feed_indices,
            )
            dispatcher: InputDispatcher = cfg.set(name="dispatcher").instantiate(parent=None)
            self.assertEqual(inferred_num_physical_feeds, dispatcher.config.num_physical_feeds)
            if logical_feed_indices is not None:
                self.assertEqual(len(logical_feed_indices), dispatcher.num_logical_feeds)
            feed_read_config = dispatcher.feed_read_config()
            if physical_feed_index in dispatcher.config.logical_feed_indices:
                self.assertIsNotNone(dispatcher.logical_feed_index)
                self.assertEqual(
                    {
                        "shard_index": dispatcher.logical_feed_index,
                        "num_shards": dispatcher.num_logical_feeds,
                    },
                    feed_read_config,
                )
            else:
                self.assertIsNone(dispatcher.logical_feed_index)
                self.assertEqual(dispatcher.num_logical_feeds, feed_read_config["num_shards"])
            feed_example_start_index = (
                feed_read_config["shard_index"] * dispatcher.feed_logical_batch_size
            )
            feed_logical_batch = {
                "example_index": feed_example_start_index
                + jnp.arange(dispatcher.feed_logical_batch_size),
            }
            feed_physical_batch = dispatcher.logical_to_physical_batch(feed_logical_batch)
            self.assertEqual(
                (dispatcher.feed_physical_batch_size,), feed_physical_batch["example_index"].shape
            )
            # Dispatch should contain the right range of values.
            if PHYSICAL_TO_LOGICAL_DISPATCH_KEY in feed_physical_batch:
                dispatch = feed_physical_batch[PHYSICAL_TO_LOGICAL_DISPATCH_KEY]
                if physical_feed_index in dispatcher.config.logical_feed_indices:
                    self.assertEqual(
                        dispatcher.feed_logical_batch_size, jnp.count_nonzero(dispatch)
                    )
                else:
                    self.assertNestedEqual(
                        jnp.zeros(
                            [dispatcher.feed_physical_batch_size, global_logical_batch_size],
                            dtype=bool,
                        ),
                        dispatch,
                    )
            all_physical_batches.append(feed_physical_batch)

        # Shuffle the feed batches.
        random.shuffle(all_physical_batches)
        # Form the global batch by concatenating the tensors.
        # pylint: disable-next=no-value-for-parameter
        global_physical_batch = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *all_physical_batches,
        )
        self.assertEqual(
            (dispatcher.config.global_physical_batch_size,),
            global_physical_batch["example_index"].shape,
        )
        global_logical_batch = dispatcher.physical_to_logical_batch(global_physical_batch)
        self.assertEqual(
            {"example_index": (global_logical_batch_size,)}, shapes(global_logical_batch)
        )
        self.assertCountEqual(
            global_logical_batch["example_index"].tolist(), range(global_logical_batch_size)
        )


class SpmdInputDispatcherTest(TestCase):
    """Tests SpmdInputDispatcher."""

    @parameterized.parameters(1, 2, 4)
    def test_every_other_process(self, divisor: int):
        device_count = jax.device_count()
        process_count = jax.process_count()
        print(f"{device_count=}, {process_count=}")
        # E.g., run on v5e-16.
        if process_count % divisor != 0:
            pytest.skip(reason="Incompatible process_count/divisor.")

        with jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("x", "y")) as mesh:
            # Shard dim=0 only along data.
            logical_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(("x",)))

            # Build an array that has dim=0 smaller than num hosts
            self.assertEqual(process_count % divisor, 0)
            global_logical_shape = (process_count // divisor, 2)

            cfg: SpmdInputDispatcher.Config = SpmdInputDispatcher.default_config().set(
                partition_spec=logical_sharding.spec,
                global_logical_batch_size=global_logical_shape[0],
            )
            dispatcher: SpmdInputDispatcher = cfg.set(name="dispatcher").instantiate(parent=None)
            read_cfg = dispatcher.feed_read_config()
            num_feeds, feed_index = read_cfg["num_shards"], read_cfg["shard_index"]

            self.assertEqual(
                dispatcher.feed_logical_batch_size * dispatcher.num_logical_feeds,
                global_logical_shape[0],
            )
            self.assertEqual(dispatcher.num_logical_feeds, num_feeds)
            self.assertEqual(dispatcher.logical_feed_index, feed_index)

            local_shape = (dispatcher.feed_logical_batch_size, *global_logical_shape[1:])
            process_arrays = [
                jax.random.uniform(jax.random.PRNGKey(i), shape=local_shape)
                for i in range(0, num_feeds)
            ]
            # Current process reads its feed idx.
            local_batch = {
                "x": jnp.expand_dims(process_arrays[feed_index], axis=0),
                "idx": jnp.array([feed_index]),
            }
            local_batch = dispatcher.logical_to_physical_batch(local_batch)
            global_physical_batch = host_to_global_device_array(
                local_batch, partition=dispatcher.partition_spec
            )
            global_logical_batch = dispatcher.physical_to_logical_batch(global_physical_batch)

            # Replicate by stacking.
            # stacked_batch = multihost_utils.process_allgather(global_logical_batch, tiled=False)
            stacked_batch = replicate_to_local_data(global_logical_batch)

            # Check that feed_index covers num feeds across workers.
            self.assertEqual(
                np.arange(num_feeds).tolist(), np.unique(stacked_batch["idx"]).tolist()
            )

            # Check that global batch covers all process-local arrays.
            # Use the idx to recover ordering.
            self.assertNestedAllClose(
                np.concatenate(process_arrays, axis=0),
                np.concatenate(stacked_batch["x"][stacked_batch["idx"]], axis=0),
            )

    @pytest.mark.for_8_devices
    def test_validate(self):
        """Tests that we raise if attempting an invalid partition."""

        device_count = jax.device_count()
        assert device_count > 1

        with jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("x", "y")) as mesh:
            # Shard dim=0 only along data.
            logical_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(("x", "y")))

            # Build an array that cannot be partitioned over logical_sharding.
            global_logical_shape = (device_count // 2, 1)

            cfg: SpmdInputDispatcher.Config = SpmdInputDispatcher.default_config().set(
                name="test",
                global_logical_batch_size=global_logical_shape[0],
            )

            with self.assertRaisesRegex(ValueError, "num_partitions"):
                cfg.clone(partition_spec=logical_sharding.spec).instantiate(parent=None)

            with self.assertRaisesRegex(ValueError, "empty"):
                cfg.clone(partition_spec=PartitionSpec()).instantiate(parent=None)
