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

    def setUp(self):
        super().setUp()
        self.mock_flags = mock.MagicMock(spec=flags.FlagValues)
        self.mock_flags.jax_backend = None
        self.patch_flags_global = mock.patch.object(flags, "FLAGS", new=self.mock_flags)
        self.patch_flags_global.start()

    def tearDown(self):
        super().tearDown()
        self.patch_flags_global.stop()

    @parameterized.parameters(
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
            ],
            expected_rolling_window_size=[],
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "rolling_window_size=1,2,3",
            ],
            expected_rolling_window_size=[1, 2, 3],
        ),
    )
    def test_from_flags(
        self,
        recorder_spec,
        expected_rolling_window_size,
    ):
        """Tests that flags are correctly parsed into the config."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = recorder_spec
        mock_fv.jax_backend = "tpu"

        recorder = GoodputRecorder.from_flags(mock_fv)

        self.assertEqual("test-name", recorder.config.name)
        self.assertEqual("/test/path", recorder.config.upload_dir)
        self.assertEqual(15, recorder.config.upload_interval)
        self.assertEqual(expected_rolling_window_size, recorder.config.rolling_window_size)

    def test_from_flags_missing_required(self):
        """Tests that missing required flags raise an error."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = ["name=test-name"]  # Missing upload_dir/interval
        mock_fv.jax_backend = "tpu"
        with self.assertRaisesRegex(RequiredFieldMissingError, "upload_dir"):
            GoodputRecorder.from_flags(mock_fv)

    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_context_manager(self, _):
        """Tests the record_event context manager."""
        cfg = GoodputRecorder.default_config().set(
            name="test",
            upload_dir="/tmp/test",
            upload_interval=1,
        )
        recorder = GoodputRecorder(cfg)
        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value
            with recorder.record_event(measurement.Event.JOB):
                pass
            mock_recorder_cls.assert_called_once()
            mock_instance.record_job_start_time.assert_called_once()
            mock_instance.record_job_end_time.assert_called_once()

    @parameterized.parameters(
        dict(is_pathways_job=False, mock_jax_backend="tpu"),
        dict(is_pathways_job=True, mock_jax_backend="proxy"),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput(self, _, is_pathways_job, mock_jax_backend):
        """Tests the maybe_monitor_goodput context manager."""
        self.mock_flags.jax_backend = mock_jax_backend
        cfg = GoodputRecorder.default_config().set(
            name="test-monitor",
            upload_dir="/test",
            upload_interval=30,
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
                pathway_enabled=is_pathways_job,
                include_badput_breakdown=True,
            )
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            mock_monitor_instance.stop_goodput_uploader.assert_called_once()

    @parameterized.parameters(
        dict(
            is_rolling_window_enabled=True,
            rolling_window_size=[10, 20],
            is_pathways_job=False,
            mock_jax_backend="tpu",
        ),
        dict(
            is_rolling_window_enabled=False,
            rolling_window_size=[],
            is_pathways_job=False,
            mock_jax_backend="tpu",
        ),
        dict(
            is_rolling_window_enabled=True,
            rolling_window_size=[50],
            is_pathways_job=True,
            mock_jax_backend="proxy",
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_rolling_window(
        self,
        mock_process_index,
        is_rolling_window_enabled,
        rolling_window_size,
        is_pathways_job,
        mock_jax_backend,
    ):  # pylint: disable=unused-argument
        """Tests the rolling window monitoring context manager."""
        self.mock_flags.jax_backend = mock_jax_backend
        cfg = GoodputRecorder.default_config().set(
            name="test-rolling",
            upload_dir="/test",
            upload_interval=30,
            rolling_window_size=rolling_window_size,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            if not is_rolling_window_enabled:
                with recorder.maybe_monitor_rolling_window_goodput():
                    pass
                mock_monitor_cls.assert_not_called()
                return
            with recorder.maybe_monitor_rolling_window_goodput():
                pass

            mock_monitor_cls.assert_called_once_with(
                job_name="test-rolling",
                logger_name="goodput_logger_test-rolling",
                tensorboard_dir="/test/rolling_window_test-rolling",
                upload_interval=30,
                monitoring_enabled=True,
                pathway_enabled=is_pathways_job,
                include_badput_breakdown=True,
            )

            mock_monitor_instance.start_rolling_window_goodput_uploader.assert_called_with(
                rolling_window_size
            )
            mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=1)
    def test_non_zero_process_index_skips_monitoring(
        self, mock_process_index
    ):  # pylint: disable=unused-argument
        """Tests that monitoring is skipped on non-zero process indices."""
        self.mock_flags.jax_backend = "tpu"
        cfg = GoodputRecorder.default_config().set(
            name="test", upload_dir="/test", upload_interval=30
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test maybe_monitor_goodput
            with recorder.maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            cfg_rolling = GoodputRecorder.default_config().set(
                name="test-rolling-skip",
                upload_dir="/test",
                upload_interval=30,
                rolling_window_size=[10, 20],
            )
            recorder_rolling = GoodputRecorder(cfg_rolling)
            with recorder_rolling.maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()
