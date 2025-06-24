# Copyright © 2023 Apple Inc.

# pylint: disable=no-self-use,missing-class-docstring
"""Tests for writers.

To run tests with Weights & Biases writers, run this file with:

    WANDB_API_KEY="..." pytest summary_writer_test.py
"""
import os
import tempfile
from enum import Enum
from unittest import mock

import jax
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from axlearn.common.config import config_class
from axlearn.common.evaler_test import DummyModel
from axlearn.common.metrics import WeightedScalar
from axlearn.common.summary import AudioSummary, ImageSummary
from axlearn.common.summary_writer import (
    CheckpointerAction,
    CompositeWriter,
    SummaryWriter,
    WandBWriter,
)
from axlearn.common.test_utils import TestCase

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class DummyLayerType(Enum):
    """A dummy enum derived type."""

    TYPE1 = 1


class DummyModelForSummaryWriters(DummyModel):
    @config_class
    class Config(DummyModel.Config):
        """Configures DummyModelForSummaryWriters.

        type: int enums are often used in config definition
              added here to test summary writers capability to handle them.
        """

        type: DummyLayerType = DummyLayerType.TYPE1


class SummaryWriterTest(TestCase):
    """Test SummaryWriter."""

    def test_add_summary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            image_n_steps = 4
            cfg: SummaryWriter.Config = SummaryWriter.default_config().set(
                name="test",
                dir=tempdir,
                write_every_n_steps=2,
                write_every_n_steps_map=dict(Image=image_n_steps),
            )
            writer = cfg.instantiate(parent=None)
            for step in (1, 2, 4):
                image = np.ones((1, 2, 3, 1))
                writer(
                    step=step,
                    values={
                        "loss": WeightedScalar(mean=3, weight=16),
                        "accuracy": WeightedScalar(mean=0.7, weight=16),
                        "learner": {"learning_rate": 0.1},
                        "image": ImageSummary(jnp.array(image)),
                    },
                )
                # Compare written summaries against expected.
                event_acc = EventAccumulator(tempdir, size_guidance={"tensors": 0})
                event_acc.Reload()
                summaries = {}
                for summary in event_acc.Tags()["tensors"]:
                    for tensor_event in reversed(event_acc.Tensors(summary)):
                        self.assertEqual(tensor_event.step, step)
                        summaries[summary] = tf.make_ndarray(tensor_event.tensor_proto)
                        break

                if step % cfg.write_every_n_steps != 0:
                    expected = {}
                elif step % image_n_steps == 0:
                    tf_image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
                    encoded_image = tf.image.encode_png(tf_image)
                    dims = tf.stack([tf.as_string(image.shape[2]), tf.as_string(image.shape[1])])
                    expected = {
                        "loss": tf.constant(3.0),
                        "accuracy": tf.constant(0.7),
                        "learner/learning_rate": tf.constant(0.1),
                        "image": tf.concat([dims, encoded_image], 0),
                    }
                else:
                    assert step % cfg.write_every_n_steps == 0
                    expected = {
                        "loss": tf.constant(3.0),
                        "accuracy": tf.constant(0.7),
                        "learner/learning_rate": tf.constant(0.1),
                    }
                self.assertNestedEqual(expected, summaries)

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cfg: SummaryWriter.Config = SummaryWriter.default_config().set(name="test", dir=tempdir)
            writer = cfg.instantiate(parent=None)
            writer.log_config(DummyModelForSummaryWriters.default_config())
            event_acc = EventAccumulator(tempdir, size_guidance={"tensors": 0})
            event_acc.Reload()
            summaries = {}
            for summary in event_acc.Tags()["tensors"]:
                for tensor_event in event_acc.Tensors(summary):
                    self.assertEqual(tensor_event.step, 0)
                    if summary != "trainer_config":
                        # skip the trainer config one, as it is too long.
                        summaries[summary] = tf.make_ndarray(tensor_event.tensor_proto)

            expected = {
                "trainer_config/dtype": tf.constant(b"'jax.numpy.float32'"),
                "trainer_config/klass": tf.constant(
                    b"'axlearn.common.summary_writer_test.DummyModelForSummaryWriters'"
                ),
                "trainer_config/layer.bias": tf.constant(b"False"),
                "trainer_config/layer.input_dim": tf.constant(b"32"),
                "trainer_config/layer.klass": tf.constant(b"'axlearn.common.layers.Linear'"),
                "trainer_config/layer.output_dim": tf.constant(b"32"),
                "trainer_config/layer.param_partition_spec[0]": tf.constant(b"'model'"),
                "trainer_config/layer.param_partition_spec[1]": tf.constant(b"None"),
                "trainer_config/name": tf.constant(b"'DummyModelForSummaryWriters'"),
                "trainer_config/type": tf.constant(b"<DummyLayerType.TYPE1: 1>"),
                "trainer_config/param_init.klass": tf.constant(
                    b"'axlearn.common.param_init.ConstantInitializer'"
                ),
                "trainer_config/param_init.value": tf.constant(b"1.0"),
            }
            self.assertEqual(expected, summaries)


class CompositeWriterTest(TestCase):
    """Tests CompositeWriter."""

    def test_multiple_summary_writers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            writer = (
                CompositeWriter.default_config()
                .set(
                    name="test_multi_writer",
                    dir=tempdir,
                    writers={
                        "writer1": SummaryWriter.default_config(),
                        "writer2": SummaryWriter.default_config(),
                    },
                )
                .instantiate(parent=None)
            )
            writer(
                step=100,
                values={
                    "loss": WeightedScalar(mean=3, weight=16),
                    "accuracy": WeightedScalar(mean=0.7, weight=16),
                    "learner": {"learning_rate": 0.1},
                },
            )

    def test_multiple_summary_writers_checkpoint(self):
        with tempfile.TemporaryDirectory() as tempdir:
            writer = (
                CompositeWriter.default_config()
                .set(
                    name="test_multi_writer",
                    dir=tempdir,
                    writers={
                        "writer1": SummaryWriter.default_config(),
                        "writer2": SummaryWriter.default_config(),
                    },
                )
                .instantiate(parent=None)
            )
            for sub_writer in writer.writers:
                sub_writer.log_checkpoint = mock.Mock()

            writer.log_checkpoint(
                ckpt_dir=tempdir,
                state=dict(x=jnp.zeros([], dtype=jnp.int32)),
                action=CheckpointerAction.SAVE,
                step=100,
            )

            for sub_writer in writer.writers:
                sub_writer.log_checkpoint.assert_called_once()


class WandBWriterTest(TestCase):
    """Tests WandBWriter."""

    def _write_per_step(self, writer: WandBWriter, step: int):
        writer(
            step=step,
            values={
                "loss": WeightedScalar(mean=3, weight=16),
                "accuracy": WeightedScalar(mean=0.7, weight=16),
                "learner": {"learning_rate": 0.1},
                "image": ImageSummary(jax.numpy.ones((2, 5, 5, 3))),
                "audio": AudioSummary(jax.numpy.ones((5, 2), dtype=jnp.float32), sample_rate=12345),
            },
        )

    @parameterized.product(step=[5, 10, 20, 30, 40])
    @pytest.mark.skipif(wandb is None, reason="wandb package not installed.")
    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_add_summary(self, step):
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                write_every_n_steps = 10
                write_every_n_steps_map = dict(Image=20, Audio=40)
                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(
                        name="test",
                        exp_name="wandb-testAddSummary",
                        dir=tempdir,
                        mode="offline",
                        write_every_n_steps=write_every_n_steps,
                        write_every_n_steps_map=write_every_n_steps_map,
                    )
                    .instantiate(parent=None)
                )
                self._write_per_step(writer, step)
                # WandB are written one step late, so an extra log is performed to ensure flushing.
                self._write_per_step(writer, 100)

                if step % write_every_n_steps != 0:
                    self.assertNotIn("loss", wandb.run.summary.keys())
                else:
                    self.assertEqual(wandb.run.summary["loss"], 3)
                    self.assertAlmostEqual(wandb.run.summary["accuracy"], 0.7)
                    self.assertAlmostEqual(wandb.run.summary["learner"]["learning_rate"], 0.1)
                    if step % write_every_n_steps_map["Image"] == 0:
                        self.assertIn("image", wandb.run.summary.keys())
                        self.assertEqual(len(wandb.run.summary["image"].filenames), 2)
                        if step % write_every_n_steps_map["Audio"] == 0:
                            self.assertTrue("audio" in wandb.run.summary.keys())
                            self.assertIsInstance(
                                wandb.run.summary["audio"], wandb.sdk.wandb_summary.SummarySubDict
                            )
                            self.assertGreater(wandb.run.summary["audio"]["size"], 0)
                        else:
                            self.assertNotIn("audio", wandb.run.summary.keys())
                    else:
                        self.assertNotIn("image", wandb.run.summary.keys())
                        self.assertNotIn("audio", wandb.run.summary.keys())
            finally:
                wandb.finish()

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dummy_model_config = DummyModelForSummaryWriters.default_config()
            flat_config = WandBWriter.format_config(
                dummy_model_config.to_flat_dict(omit_default_values={})
            )
            config = WandBWriter.format_config(dummy_model_config.to_dict())
            try:
                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(name="test", exp_name="wandb-testLogConfig", dir=tempdir, mode="offline")
                    .instantiate(parent=None)
                )
                writer.log_config(dummy_model_config)
            finally:
                stored_config = dict(wandb.config)
                wandb.finish()
                stored_flat_config = stored_config.pop(
                    WandBWriter._FLAT_CONFIG_KEY  # pylint: disable=protected-access
                )
                assert stored_flat_config == flat_config
                assert stored_config == config

    @pytest.mark.skipif(wandb is None, reason="wandb package not installed.")
    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_resume(self):
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(name="test", exp_name="wandb-testResume", dir=tempdir)
                    .instantiate(parent=None)
                )
                exp_id = wandb.run.id

                for step in [10, 20, 30, 40]:
                    self._write_per_step(writer, step)
                wandb.finish()

                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(name="test", exp_name="wandb-testResume", dir=tempdir)
                    .instantiate(parent=None)
                )
                assert wandb.run.id == exp_id
                # Because we resume from checkpoints, we may compute metrics
                # for a training step we completed in the previous run.
                # We need to make sure that we can log at the same training step multiple times.
                for step in [30, 40, 50, 60]:
                    self._write_per_step(writer, step)
            finally:
                wandb.finish()


if __name__ == "__main__":
    absltest.main()
