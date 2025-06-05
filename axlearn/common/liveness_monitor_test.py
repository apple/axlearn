# Copyright © 2023 Apple Inc.

"""Tests liveness monitors."""
# pylint: disable=protected-access

import os
import time
from unittest import mock

import tensorflow as tf

from axlearn.common.liveness_monitor import LivenessMonitor, TfSummaryMonitor
from axlearn.common.test_utils import TestWithTemporaryCWD


class TfSummaryMonitorTest(TestWithTemporaryCWD):
    """Tests TfSummaryMonitor."""

    def test_ping(self):
        # Note: test runs in a temporary cwd. See TestWithTemporaryCWD.
        os.makedirs("train_train", exist_ok=True)
        writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), "train_train"))

        # Construct the monitor.
        cfg = TfSummaryMonitor.default_config().set(
            summary_dir=self._temp_root.name,
            max_timeout_seconds=5,
            max_start_seconds=1,
        )
        monitor: LivenessMonitor = cfg.instantiate()

        # Pinging before monitor starts raises.
        self.assertFalse(monitor.started())
        with self.assertRaisesRegex(RuntimeError, "started"):
            monitor.ping()

        # Set initial timestamp.
        monitor.reset()
        self.assertTrue(monitor.started())

        # Initial timestamp.
        t0 = monitor._latest
        # Within grace period of t0.
        t0_grace_period = t0 + cfg.max_start_seconds
        # Within one timeout window of t0.
        t0_timeout = t0 + cfg.max_timeout_seconds

        # Pinging within grace period always returns True.
        with mock.patch("time.time", mock.MagicMock(return_value=t0_grace_period)):
            self.assertTrue(monitor.ping())

        # Pinging outside grace period succeeds if we're within timeout window.
        with mock.patch("time.time", mock.MagicMock(return_value=t0_timeout)):
            self.assertAlmostEqual(monitor._latest, t0)
            self.assertTrue(monitor.ping())

        # Pinging outside grace period + timeout fails.
        with mock.patch("time.time", mock.MagicMock(return_value=t0_timeout + 1)):
            self.assertAlmostEqual(monitor._latest, t0)
            self.assertFalse(monitor.ping())

        # Try creating some fake summaries.
        # Note: tf.summary does not seem to use time.time, so mocking it doesn't work.
        time.sleep(1)
        with writer.as_default():
            tf.summary.scalar("test", 0.1, step=123)
        self.assertTrue(monitor.ping())
        # Test that latest time is updated.
        self.assertGreater(monitor._latest, t0)

        # Test resetting.
        with mock.patch("time.time", mock.MagicMock(return_value=t0 + 100)):
            self.assertFalse(monitor.ping())

            # Test that resetting the grace period succeeds.
            monitor.reset()
            self.assertTrue(monitor.started())
            self.assertAlmostEqual(monitor._latest, t0 + 100)
            self.assertTrue(monitor.ping())
