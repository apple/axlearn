# pylint: disable=no-self-use
"""Tests for writers.

To run tests with Weights & Biases writers, run this file with:

    WANDB_API_KEY="..." pytest summary_writer_test.py
"""
import os
import tempfile

import pytest
from absl.testing import absltest

from axlearn.common.metrics import WeightedScalar
from axlearn.common.summary_writer import CompositeWriter, SummaryWriter, WandBWriter

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class SummaryWriterTest(absltest.TestCase):
    """Test SummaryWriter."""

    def test_add_summary(self):
        tempdir = tempfile.mkdtemp()
        writer: SummaryWriter = (
            SummaryWriter.default_config().set(name="test", dir=tempdir).instantiate(parent=None)
        )
        writer(
            step=100,
            values={
                "loss": WeightedScalar(mean=3, weight=16),
                "accuracy": WeightedScalar(mean=0.7, weight=16),
                "learner": {"learning_rate": 0.1},
            },
        )


class CompositeWriterTest(absltest.TestCase):
    """Tests CompositeWriter."""

    def test_multiple_summary_writers(self):
        tempdir = tempfile.mkdtemp()

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
        tempdir = tempfile.mkdtemp()
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
        tempdir = tempfile.mkdtemp()
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
