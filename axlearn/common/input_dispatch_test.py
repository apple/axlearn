# Copyright © 2024 Apple Inc.

"""Tests input dispatcher."""

import random

import jax
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import shapes


class DispatcherTest(TestCase):
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
        for physical_feed_index in range(num_physical_feeds):
            cfg = InputDispatcher.default_config().set(
                global_logical_batch_size=global_logical_batch_size,
                global_physical_batch_size=global_physical_batch_size,
                num_physical_feeds=num_physical_feeds,
                physical_feed_index=physical_feed_index,
                logical_feed_indices=logical_feed_indices,
            )
            dispatcher: InputDispatcher = cfg.set(name="dispatcher").instantiate(parent=None)
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
