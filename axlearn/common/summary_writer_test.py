# Copyright © 2023 Apple Inc.

# pylint: disable=no-self-use
"""Tests for writers.

To run tests with Weights & Biases writers, run this file with:

    WANDB_API_KEY="..." pytest summary_writer_test.py
"""
import os
import tempfile

import pytest
import tensorflow as tf
from absl.testing import absltest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from axlearn.common.metrics import WeightedScalar
from axlearn.common.summary_writer import CompositeWriter, SummaryWriter, WandBWriter

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
                for _, step, tensor in event_acc.Tensors(summary):
                    self.assertEqual(step, 100)
                    summaries[summary] = tf.make_ndarray(tensor)

            expected = {
                "loss": tf.constant(3.0),
                "accuracy": tf.constant(0.7),
                "learner/learning_rate": tf.constant(0.1),
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


class WandBWriterTest(absltest.TestCase):
    """Tests WandBWriter."""

    def _write_per_step(self, writer: WandBWriter, step: int):
        writer(
            step=step,
            values={
                "loss": WeightedScalar(mean=3, weight=16),
                "accuracy": WeightedScalar(mean=0.7, weight=16),
                "learner": {"learning_rate": 0.1},
            },
        )

    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_add_summary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            writer: WandBWriter = (
                WandBWriter.default_config()
                .set(name="test", exp_name="wandb-testAddSummary", dir=tempdir)
                .instantiate(parent=None)
            )
            for step in [10, 20, 30, 40]:
                self._write_per_step(writer, step)
            wandb.finish()

    @pytest.mark.skipif("WANDB_API_KEY" not in os.environ, reason="wandb api key not found.")
    def test_resume(self):
        with tempfile.TemporaryDirectory() as tempdir:
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
            wandb.finish()
