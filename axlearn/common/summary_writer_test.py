# Copyright © 2023 Apple Inc.

# pylint: disable=no-self-use,missing-class-docstring
"""Tests for writers.

To run tests with Weights & Biases writers, run this file with:

    WANDB_API_KEY="..." pytest summary_writer_test.py
"""
import os
import tempfile
from unittest import mock

import jax
import pytest
import tensorflow as tf
from absl.testing import absltest
from jax import numpy as jnp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from axlearn.common.evaler_test import DummyModel
from axlearn.common.metrics import WeightedScalar
from axlearn.common.summary import ImageSummary
from axlearn.common.summary_writer import (
    CheckpointerAction,
    CompositeWriter,
    SummaryWriter,
    WandBWriter,
)

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class SummaryWriterTest(absltest.TestCase):
    """Test SummaryWriter."""

    def test_add_summary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cfg: SummaryWriter.Config = SummaryWriter.default_config().set(name="test", dir=tempdir)
            writer = cfg.instantiate(parent=None)
            writer(
                step=100,
                values={
                    "loss": WeightedScalar(mean=3, weight=16),
                    "accuracy": WeightedScalar(mean=0.7, weight=16),
                    "learner": {"learning_rate": 0.1},
                },
            )
            # Compare written summaries against expected.
            event_acc = EventAccumulator(tempdir, size_guidance={"tensors": 0})
            event_acc.Reload()
            summaries = {}
            for summary in event_acc.Tags()["tensors"]:
                for tensor_event in event_acc.Tensors(summary):
                    self.assertEqual(tensor_event.step, 100)
                    summaries[summary] = tf.make_ndarray(tensor_event.tensor_proto)

            expected = {
                "loss": tf.constant(3.0),
                "accuracy": tf.constant(0.7),
                "learner/learning_rate": tf.constant(0.1),
            }
            self.assertEqual(expected, summaries)

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cfg: SummaryWriter.Config = SummaryWriter.default_config().set(name="test", dir=tempdir)
            writer = cfg.instantiate(parent=None)
            writer.log_config(DummyModel.default_config())
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
                "trainer_config/klass": tf.constant(b"'axlearn.common.evaler_test.DummyModel'"),
                "trainer_config/layer.bias": tf.constant(b"False"),
                "trainer_config/layer.input_dim": tf.constant(b"32"),
                "trainer_config/layer.klass": tf.constant(b"'axlearn.common.layers.Linear'"),
                "trainer_config/layer.output_dim": tf.constant(b"32"),
                "trainer_config/layer.param_partition_spec[0]": tf.constant(b"'model'"),
                "trainer_config/layer.param_partition_spec[1]": tf.constant(b"None"),
                "trainer_config/name": tf.constant(b"'DummyModel'"),
                "trainer_config/param_init.klass": tf.constant(
                    b"'axlearn.common.param_init.ConstantInitializer'"
                ),
                "trainer_config/param_init.value": tf.constant(b"1.0"),
            }
            self.assertEqual(expected, summaries)


class CompositeWriterTest(absltest.TestCase):
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


class WandBWriterTest(absltest.TestCase):
    """Tests WandBWriter."""

    def _write_per_step(self, writer: WandBWriter, step: int):
        writer(
            step=step,
            values={
                "loss": WeightedScalar(mean=3, weight=16),
                "accuracy": WeightedScalar(mean=0.7, weight=16),
                "learner": {"learning_rate": 0.1},
                "image": ImageSummary(jax.numpy.ones((2, 5, 5, 3))),
            },
        )

    @pytest.mark.skipif(wandb is None, reason="wandb package not installed.")
    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_add_summary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                writer: WandBWriter = (
                    WandBWriter.default_config()
                    .set(name="test", exp_name="wandb-testAddSummary", dir=tempdir, mode="offline")
                    .instantiate(parent=None)
                )
                for step in [10, 20, 30, 40]:
                    self._write_per_step(writer, step)

                self.assertEqual(wandb.run.summary["loss"], 3)
                self.assertAlmostEqual(wandb.run.summary["accuracy"], 0.7)
                self.assertAlmostEqual(wandb.run.summary["learner"]["learning_rate"], 0.1)
                self.assertTrue("image" in wandb.run.summary.keys())
                self.assertEqual(len(wandb.run.summary["image"].filenames), 2)
            finally:
                wandb.finish()

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
