# Copyright © 2024 Apple Inc.

"""Tests measurement utils for GCP."""
# pylint: disable=protected-access

from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp.measurement import GoodputRecorder
from axlearn.common import measurement
from axlearn.common.config import RequiredFieldMissingError


class GoodputRecorderTest(parameterized.TestCase):
    """Tests GoodputRecorder."""

    def test_from_flags_with_spec(self):
        """Tests that flags are correctly parsed."""
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default(
            "recorder_spec",
            [
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_rolling_window_goodput_monitoring=True",
                "rolling_window_size=1,2,3",
            ],
        )
        fv.mark_as_parsed()
        recorder = GoodputRecorder.from_flags(fv)
        self.assertEqual("test-name", recorder.config.name)
        self.assertEqual("/test/path", recorder.config.upload_dir)
        self.assertEqual(15, recorder.config.upload_interval)
        self.assertTrue(recorder.config.enable_rolling_window_goodput_monitoring)
        self.assertEqual([1, 2, 3], recorder.config.rolling_window_size)

    def test_from_flags_missing_required(self):
        """Tests that missing required flags raise an error."""
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default("recorder_spec", ["name=test-name"])  # Missing upload_dir/interval
        fv.mark_as_parsed()
        with self.assertRaisesRegex(RequiredFieldMissingError, "upload_dir"):
            GoodputRecorder.from_flags(fv)

    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_context_manager(self, _):
        """Tests the record_event context manager."""
        recorder = GoodputRecorder(GoodputRecorder.default_config().set(name="test"))
        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder:
            mock_instance = mock_recorder.return_value
            with recorder.record_event(measurement.Event.JOB):
                pass
            mock_recorder.assert_called_once()
            mock_instance.record_job_start_time.assert_called_once()
            mock_instance.record_job_end_time.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput(self, _):
        """Tests the maybe_monitor_goodput context manager."""
        cfg = GoodputRecorder.default_config().set(
            name="test-monitor",
            upload_dir="/test",
            upload_interval=30,
            enable_gcp_goodput_metrics=True,
            enable_pathways_goodput=False,
            include_badput_breakdown=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            with recorder.maybe_monitor_goodput():
                pass

            # Verify that GoodputMonitor was instantiated with the correct parameters.
            mock_monitor_cls.assert_called_once_with(
                job_name="test-monitor",
                logger_name="goodput_logger_test-monitor",
                tensorboard_dir="/test",
                upload_interval=30,
                monitoring_enabled=True,
                pathway_enabled=False,
                include_badput_breakdown=True,
                gcp_options=mock.ANY,
            )
            # Verify the start and stop methods were called.
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            mock_monitor_instance.stop_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_rolling_window(self, _):
        """Tests the rolling window monitoring context manager."""
        cfg = GoodputRecorder.default_config().set(
            name="test-rolling",
            upload_dir="/test",
            upload_interval=30,
            enable_rolling_window_goodput_monitoring=True,
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            with recorder.maybe_monitor_rolling_window_goodput():
                pass

            # Verify that GoodputMonitor was instantiated for rolling window.
            mock_monitor_cls.assert_called_once()
            self.assertEqual(
                "/test/rolling_window_test-rolling",
                mock_monitor_cls.call_args.kwargs["tensorboard_dir"],
            )

            # Verify the correct the start and stop methods were called.
            mock_monitor_instance.start_rolling_window_goodput_uploader.assert_called_with([10, 20])
            mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=1)
    def test_non_zero_process_index_skips_monitoring(
        self, mock_process_index
    ):  # pylint: disable=unused-argument
        """Tests that monitoring is skipped on non-zero process indices."""
        cfg = GoodputRecorder.default_config().set(
            name="test", upload_dir="/test", upload_interval=30
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test maybe_monitor_goodput
            with recorder.maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            # Test maybe_monitor_rolling_window_goodput
            recorder.config.enable_rolling_window_goodput_monitoring = True
            with recorder.maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()
