# Copyright © 2023 Apple Inc.

"""Tests for summary.py"""
import functools
import os
import tempfile

import chex
import jax
import numpy as np
import pytest
import tensorflow as tf
import wandb
from jax.experimental.pjit import pjit
from tensorboard.backend.event_processing import event_accumulator

from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.summary import CallbackSummary, ImageSummary
from axlearn.common.summary_writer import SummaryWriter, WandBWriter
from axlearn.common.test_utils import TestCase


class SummaryTest(TestCase):
    """Tests for `Summary`."""

    def test_add_summary_image(self):
        tempdir = tempfile.mkdtemp()
        writer: SummaryWriter = (
            SummaryWriter.default_config().set(name="test", dir=tempdir).instantiate(parent=None)
        )
        color_image = jax.numpy.ones((2, 5, 5, 3))
        grayscale_image = jax.numpy.zeros((2, 5, 5))
        writer(
            step=100,
            values={
                "color_image": ImageSummary(color_image),
                "grayscale_image": ImageSummary(grayscale_image),
            },
        )

        ea = event_accumulator.EventAccumulator(tempdir)
        ea.Reload()
        logged_color_image = tf.stack(
            [
                tf.image.decode_image(im)
                for im in tf.make_ndarray(ea.Tensors("color_image")[0].tensor_proto)[2:]
            ]
        )
        chex.assert_trees_all_close(logged_color_image / 255, color_image)
        logged_grayscale_image = tf.stack(
            [
                tf.image.decode_image(im)
                for im in tf.make_ndarray(ea.Tensors("grayscale_image")[0].tensor_proto)[2:]
            ]
        )
        chex.assert_trees_all_close(logged_grayscale_image / 255, grayscale_image[..., None])

    @pytest.mark.skipif(wandb is None, reason="wandb package not installed.")
    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_callback_summary(self):
        class _TestModule(Module):
            """A test `Module`."""

            def forward(self):
                # This logs a 7 row table with two columns where each cell contains a 16 x 16 color
                # image.
                # Shape: num examples x table columns x image height x image width x channels.
                images = jax.numpy.ones((7, 2, 16, 16, 3))

                def create_table(images: np.ndarray):
                    return wandb.Table(
                        ["output", "target"], [[wandb.Image(img) for img in row] for row in images]
                    )

                self.add_summary("my_table", CallbackSummary(create_table, images))

        module = _TestModule.default_config().set(name="tmp").instantiate(parent=None)

        # Test that it works under jit.
        @jax.jit
        def _test():
            _, output_collection = F(
                module, prng_key=jax.random.PRNGKey(0), state={}, inputs=[], is_training=True
            )
            return output_collection

        with tempfile.TemporaryDirectory() as tempdir:
            try:
                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(name="test", exp_name="wandb-testAddSummary", dir=tempdir, mode="offline")
                    .instantiate(parent=None)
                )

                output_collection = _test()
                self.assertTrue("my_table" in output_collection.summaries)
                self.assertTrue(
                    isinstance(output_collection.summaries["my_table"].value(), wandb.Table)
                )

                writer(step=100, values=output_collection.summaries)
                # Not sure how to check it was actually written in offline mode.
            finally:
                wandb.finish()

    def test_with_pjit_out_shardings(self):
        """There is a JAX bug where PJIT with out_shardings doesn't consistently support pytrees
        that validate themselves. This tests that summaries still work under conditions that trigger
        this bug.
        """

        @functools.partial(pjit, out_shardings=(None, None))
        def f():
            return ImageSummary(jax.numpy.ones((4, 5, 6))), None

        f()

    def test_validate(self):
        """Check validate() works in normal conditions."""
        with self.assertRaises(ValueError):
            ImageSummary(jax.numpy.ones((1, 1)))
        ImageSummary(jax.numpy.ones((1, 1, 1)))
