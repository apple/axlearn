# Copyright © 2024 Apple Inc.

"""Tests measurement utils for GCP."""
# pylint: disable=protected-access

import contextlib
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp.measurement import GoodputRecorder
from axlearn.common import measurement


class GoodputRecorderTest(parameterized.TestCase):
    """Tests GoodputRecorder."""

    @parameterized.parameters(
        (None,), (["name=test-name", "upload_dir=/test/path/to/upload", "upload_interval=15"],)
    )
    def test_from_flags(self, spec):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        if spec is not None:
            fv.set_default("recorder_spec", spec)
        fv.mark_as_parsed()

        if spec is None:
            ctx = self.assertRaisesRegex(ValueError, "name")
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            recorder = GoodputRecorder.from_flags(fv)
            # Recorder is not instantiated until first event.
            self.assertIsNone(recorder._recorder)

    def test_record_and_monitor(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default(
            "recorder_spec",
            ["name=test-name", "upload_dir=/test/path/to/upload", "upload_interval=15"],
        )
        fv.mark_as_parsed()

        recorder = GoodputRecorder.from_flags(fv)
        recorder._recorder = mock.MagicMock()
        recorder.record(measurement.Event.START_JOB)
        self.assertTrue(recorder._recorder.record_job_start_time.called)

    def test_start_monitoring(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default(
            "recorder_spec",
            ["name=test-name", "upload_dir=/test/path/to/upload", "upload_interval=15"],
        )
        fv.mark_as_parsed()

        recorder = GoodputRecorder.from_flags(fv)
        self.assertIsNone(recorder._monitor)  # Ensure _monitor is initially None

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_goodput_monitor:
            mock_monitor_instance = mock_goodput_monitor.return_value
            recorder.start_monitoring()

            # Check that GoodputMonitor was instantiated
            mock_goodput_monitor.assert_called_once_with(
                job_name="test-name",
                logger_name="goodput_logger_test-name",
                tensorboard_dir="/test/path/to/upload",
                upload_interval=15,
                monitoring_enabled=True,
                include_badput_breakdown=True,
            )

            # Ensure that start_goodput_uploader is called on the monitor instance
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            self.assertIsNotNone(recorder._monitor)

    def test_missing_required_flags(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        # Missing 'upload_dir' and 'upload_interval' from recorder_spec
        fv.set_default("recorder_spec", ["name=test-name"])  # Incomplete config
        fv.mark_as_parsed()

        # Expecting ValueError since 'upload_dir' and 'upload_interval' are required
        with self.assertRaises(ValueError):
            GoodputRecorder.from_flags(fv)

    def test_monitoring_initialization_failure(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default(
            "recorder_spec",
            ["name=test-name", "upload_dir=/test/path/to/upload", "upload_interval=15"],
        )
        fv.mark_as_parsed()

        recorder = GoodputRecorder.from_flags(fv)
        self.assertIsNone(recorder._monitor)

        # Mock a failure in initializing the GoodputMonitor
        with mock.patch(
            "ml_goodput_measurement.monitoring.GoodputMonitor",
            side_effect=Exception("Failed to initialize GoodputMonitor"),
        ):
            with self.assertRaises(Exception):
                recorder.start_monitoring()
            self.assertIsNone(recorder._monitor)

    def test_non_zero_process_index(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default(
            "recorder_spec",
            ["name=test-name", "upload_dir=/test/path/to/upload", "upload_interval=15"],
        )
        fv.mark_as_parsed()

        recorder = GoodputRecorder.from_flags(fv)
        self.assertIsNone(recorder._monitor)

        with mock.patch("jax.process_index") as mock_process_index:
            mock_process_index.return_value = 1  # Simulate a non-zero process index

            try:
                recorder.start_monitoring()
            except AttributeError:
                self.fail("AttributeError was raised unexpectedly.")
