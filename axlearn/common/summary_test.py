# Copyright © 2023 Apple Inc.

"""Tests for summary.py"""

import dataclasses
import functools
import os
import tempfile

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import wandb
from jax.experimental.pjit import pjit
from tensorboard.backend.event_processing import event_accumulator

from axlearn.common import learner, optimizers, trainer_test
from axlearn.common.config import config_for_function
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.summary import AudioSummary, CallbackSummary, ImageSummary
from axlearn.common.summary_writer import SummaryWriter, WandBWriter
from axlearn.common.test_utils import TestCase
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_test import DummyInput
from axlearn.common.utils import Tensor, flatten_items, tree_paths


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

    def test_with_tree_paths(self):
        """Tests that `ImageSummary` works with `tree_paths()`."""
        img = jnp.ones((1, 1, 1, 3))
        s = dict(a=ImageSummary(img), b=ImageSummary(img))
        self.assertEqual(
            tree_paths(s), dict(a=ImageSummary("a/_value"), b=ImageSummary("b/_value"))
        )

    def test_with_flatten_items(self):
        """Tests that `ImageSummary` works with `flatten_items()`."""
        img = jnp.ones((1, 1, 1, 3))
        s = dict(a=ImageSummary(img), b=ImageSummary(img))
        self.assertSequenceEqual(flatten_items(s), [("a/_value", img), ("b/_value", img)])

    def test_end_to_end(self):
        """Tests that `ImageSummary` and `AudioSummary` works with `SpmdTrainer` and
        `SpmdEvaler` in an end-to-end fashion.
        """
        img = jnp.broadcast_to(jnp.array([0.0, 0.5, 1.0]), shape=(1, 1, 3))
        img = jnp.array([img, img])
        audio1 = jax.random.uniform(jax.random.PRNGKey(124), [16000 * 5]) * 2.0 - 1.0
        audio2 = jax.random.uniform(jax.random.PRNGKey(125), [8000 * 5, 3]) * 2.0 - 1.0

        class ImageSummaryModel(trainer_test.DummyModel):
            def forward(self, *args, **kwargs):
                self.add_summary("img", ImageSummary(img))
                self.add_summary("audio1", AudioSummary(audio1))
                self.add_summary("audio2", AudioSummary(audio2, sample_rate=8000))
                return super().forward(*args, **kwargs)

        cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(name="test_trainer")
        with tempfile.TemporaryDirectory() as cfg.dir:
            cfg.mesh_axis_names = ("data", "model")
            cfg.mesh_shape = (1, 1)
            cfg.model = ImageSummaryModel.default_config().set(dtype=jnp.float32)
            cfg.input = trainer_test.DummyInput.default_config()
            cfg.learner = learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1,
                    decouple_weight_decay=True,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            )

            evaler_cfg = SpmdEvaler.default_config()
            evaler_cfg.input = DummyInput.default_config().set(total_num_batches=2)
            evaler_cfg.eval_policy.n = 2
            cfg.evalers = dict(eval_dummy=evaler_cfg)
            cfg.checkpointer.save_policy.n = 5
            cfg.max_step = 8
            trainer: SpmdTrainer = cfg.instantiate(parent=None)
            trainer.run(prng_key=jax.random.PRNGKey(123))

            @dataclasses.dataclass
            class Expected:
                """Information about expected logged image summaries."""

                path: str
                count: int
                key: str

                def shape(self):
                    # The trainer / evaler makes `count` calls to forward().
                    # Each call to forward logs a batch of two images.
                    return (self.count, 2, 1, 1, 3)

                def img(self):
                    return jnp.broadcast_to(img, self.shape())

            expected = [
                Expected(path=os.path.join(cfg.dir, "summaries", "eval_dummy"), count=4, key="img"),
                Expected(
                    path=os.path.join(cfg.dir, "summaries", "train_train"), count=8, key="model/img"
                ),
            ]

            for info in expected:
                ea = event_accumulator.EventAccumulator(info.path)
                ea.Reload()

                logged_evaler_img = tf.stack(
                    [
                        tf.stack(
                            [
                                tf.image.decode_image(im)
                                for im in tf.make_ndarray(event.tensor_proto)[2:]
                            ]
                        )
                        for event in ea.Tensors(info.key)
                    ]
                )
                self.assertEqual(logged_evaler_img.shape, info.shape())
                # TB uses lossy compression.
                self.assertNestedAllClose(logged_evaler_img / 255, info.img(), rtol=0.01, atol=0)

            @dataclasses.dataclass
            class ExpectedAudio:
                """Information about expected logged audio summaries."""

                path: str
                count: int
                key: str
                sr: int
                audio: Tensor

                def shape(self):
                    # The trainer / evaler makes `count` calls to forward().
                    # Each call to forward logs a batch of audios.
                    return [self.count] + list(self.audio.shape)

                def audios(self):
                    return jnp.broadcast_to(self.audio, self.shape())

                def sample_rate(self):
                    return jnp.broadcast_to(self.sr, (self.count,))

            expected = [
                ExpectedAudio(
                    path=os.path.join(cfg.dir, "summaries", "eval_dummy"),
                    count=4,
                    key="audio1",
                    audio=audio1[:, None],
                    sr=16000,
                ),
                ExpectedAudio(
                    path=os.path.join(cfg.dir, "summaries", "train_train"),
                    count=8,
                    key="model/audio1",
                    audio=audio1[:, None],
                    sr=16000,
                ),
                ExpectedAudio(
                    path=os.path.join(cfg.dir, "summaries", "eval_dummy"),
                    count=4,
                    key="audio2",
                    audio=audio2,
                    sr=8000,
                ),
                ExpectedAudio(
                    path=os.path.join(cfg.dir, "summaries", "train_train"),
                    count=8,
                    key="model/audio2",
                    audio=audio2,
                    sr=8000,
                ),
            ]

            for info in expected:
                ea = event_accumulator.EventAccumulator(info.path)
                ea.Reload()

                logged_evaler_audio = []
                logged_evaler_sr = []
                for event in ea.Tensors(info.key):
                    results = tf.make_ndarray(event.tensor_proto)
                    for encoded, _ in results:
                        decoded, sr = tf.audio.decode_wav(encoded)
                        logged_evaler_audio.append(decoded)
                        logged_evaler_sr.append(sr)

                logged_evaler_sr = np.stack(logged_evaler_sr)
                self.assertNestedAllClose(logged_evaler_sr, info.sample_rate())

                logged_evaler_audio = tf.stack(logged_evaler_audio, 0)
                self.assertEqual(logged_evaler_audio.shape, info.shape())
                # TB convert it int16 and convert it back.
                self.assertNestedAllClose(
                    logged_evaler_audio, info.audios(), rtol=0, atol=1 / 65536
                )

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
        good = ImageSummary(jax.numpy.ones((1, 1, 1)))
        good.validate()
        bad = ImageSummary(jax.numpy.ones((1, 1)))
        with self.assertRaises(ValueError):
            bad.validate()
