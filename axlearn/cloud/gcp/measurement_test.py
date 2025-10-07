# Copyright © 2024 Apple Inc.

"""Tests measurement utils for GCP."""
# pylint: disable=protected-access

from unittest import mock

from absl import flags, logging
from absl.testing import parameterized

from axlearn.cloud.gcp.measurement import GoodputRecorder
from axlearn.common import measurement
from axlearn.common.config import RequiredFieldMissingError


class GoodputRecorderTest(parameterized.TestCase):
    """Tests GoodputRecorder."""

    @parameterized.parameters(
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "rolling_window_size=1,2,3",
                "jax_backend=proxy",
            ],
            expected_rolling_window_size=[1, 2, 3],
            expected_jax_backend="proxy",
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_monitoring=true",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
            expected_enable_monitoring=True,
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_monitoring=false",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
            expected_enable_monitoring=False,
        ),
    )
    def test_from_flags(
        self,
        recorder_spec,
        expected_rolling_window_size,
        expected_jax_backend,
        expected_enable_monitoring=True,
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
        self.assertEqual(expected_jax_backend, recorder.config.jax_backend)
        self.assertEqual(expected_enable_monitoring, recorder.config.enable_monitoring)

    def test_from_flags_missing_required(self):
        """Tests that missing required flags raise an error."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = ["name=test-name"]  # Missing upload_dir/interval
        mock_fv.jax_backend = "tpu"
        with self.assertRaisesRegex(RequiredFieldMissingError, "upload_dir"):
            GoodputRecorder.from_flags(mock_fv)

    @parameterized.parameters(
        dict(
            event=measurement.EventType.JOB,
            expected_start="record_job_start_time",
            expected_end="record_job_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.STEP,
            expected_start="record_step_start_time",
            expected_end=None,
            args=(123,),
            kwargs={},
            expect_end_call=False,
        ),
        dict(
            event=measurement.EventType.ACCELERATOR_INIT,
            expected_start="record_tpu_init_start_time",
            expected_end="record_tpu_init_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.TRAINING_PREPARATION,
            expected_start="record_training_preparation_start_time",
            expected_end="record_training_preparation_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.DATA_LOADING,
            expected_start="record_data_loading_start_time",
            expected_end="record_data_loading_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.CUSTOM_BADPUT_EVENT,
            expected_start="record_custom_badput_event_start_time",
            expected_end="record_custom_badput_event_end_time",
            args=(),
            kwargs={"custom_badput_event_type": "TEST_TYPE"},
            expect_end_call=True,
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_context_manager_success(
        self, _, event, expected_start, expected_end, args, kwargs, expect_end_call
    ):
        """Tests that record_event calls correct start and end methods with args and kwargs."""
        cfg = GoodputRecorder.default_config().set(
            name="test",
            upload_dir="/tmp/test",
            upload_interval=1,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value

            start_mock = mock.MagicMock()
            setattr(mock_instance, expected_start, start_mock)
            if expect_end_call and expected_end:
                end_mock = mock.MagicMock()
                setattr(mock_instance, expected_end, end_mock)

            with recorder.record_event(event, *args, **kwargs):
                pass

            mock_recorder_cls.assert_called_once()
            start_mock.assert_called_once_with(*args, **kwargs)
            if expect_end_call and expected_end:
                end_mock.assert_called_once_with(*args, **kwargs)

    def test_record_event_context_manager_handles_runtime_error(self):
        cfg = GoodputRecorder.default_config().set(
            name="test",
            upload_dir="/tmp/test",
            upload_interval=1,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("jax.process_index", return_value=0):
            with mock.patch(
                "ml_goodput_measurement.goodput.GoodputRecorder"
            ) as mock_recorder_cls, mock.patch.object(logging, "warning") as mock_warning:
                mock_instance = mock_recorder_cls.return_value

                def raise_runtime_error(*args, **kwargs):
                    raise RuntimeError("mocked error")

                mock_instance.record_job_start_time.side_effect = raise_runtime_error
                mock_instance.record_job_end_time.side_effect = raise_runtime_error
                # Should not crash here.
                with recorder.record_event(measurement.EventType.JOB):
                    pass

                # Assert warnings were logged for start and end failures
                assert mock_warning.call_count == 2
                start_call = mock_warning.call_args_list[0]
                end_call = mock_warning.call_args_list[1]

                assert "Failed to record" in start_call.args[0]
                assert "Failed to record" in end_call.args[0]

    @parameterized.parameters(
        dict(is_pathways_job=False, mock_jax_backend="tpu"),
        dict(is_pathways_job=True, mock_jax_backend="proxy"),
        dict(is_pathways_job=False, mock_jax_backend=None),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput(self, _, is_pathways_job, mock_jax_backend):
        """Tests the _maybe_monitor_goodput context manager."""
        cfg = GoodputRecorder.default_config().set(
            name="test-monitor",
            upload_dir="/test",
            upload_interval=30,
            jax_backend=mock_jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            with recorder._maybe_monitor_goodput():
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
        """Tests the rolling window monitoring."""
        cfg = GoodputRecorder.default_config().set(
            name="test-rolling",
            upload_dir="/test",
            upload_interval=30,
            rolling_window_size=rolling_window_size,
            jax_backend=mock_jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            if not is_rolling_window_enabled:
                with recorder._maybe_monitor_rolling_window_goodput():
                    pass
                mock_monitor_cls.assert_not_called()
                return
            with recorder._maybe_monitor_rolling_window_goodput():
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
        cfg = GoodputRecorder.default_config().set(
            name="test", upload_dir="/test", upload_interval=30, enable_monitoring=True
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test cumulative goodput monitoring.
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            cfg_rolling = GoodputRecorder.default_config().set(
                name="test-rolling-skip",
                upload_dir="/test",
                upload_interval=30,
                rolling_window_size=[10, 20],
                enable_monitoring=True,
            )
            recorder_rolling = GoodputRecorder(cfg_rolling)
            with recorder_rolling._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()

    @parameterized.parameters(
        dict(
            rolling_window_size=[5, 10],
            jax_backend="tpu",
            expected_monitor_calls=2,  # Cumulative & Rolling Window
            expect_rolling=True,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[],
            jax_backend="tpu",
            expected_monitor_calls=1,  # Cumulative only
            expect_rolling=False,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[5, 10],
            jax_backend=None,  # Disables Pathways
            expected_monitor_calls=2,
            expect_rolling=True,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[],
            jax_backend=None,
            expected_monitor_calls=1,
            expect_rolling=False,
            expect_cumulative=True,
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_all(
        self,
        _,
        rolling_window_size,
        jax_backend,
        expected_monitor_calls,
        expect_rolling,
        expect_cumulative,
    ):
        """Tests all goodput monitoring with various configs."""
        cfg = GoodputRecorder.default_config().set(
            name="test-all",
            upload_dir="/test",
            upload_interval=30,
            rolling_window_size=rolling_window_size,
            jax_backend=jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value

            with recorder.maybe_monitor_all():
                pass

            self.assertEqual(mock_monitor_cls.call_count, expected_monitor_calls)

            if expect_cumulative:
                mock_monitor_instance.start_goodput_uploader.assert_called_once()
                mock_monitor_instance.stop_goodput_uploader.assert_called_once()
            else:
                mock_monitor_instance.start_goodput_uploader.assert_not_called()
                mock_monitor_instance.stop_goodput_uploader.assert_not_called()

            if expect_rolling:
                mock_monitor_instance.start_rolling_window_goodput_uploader.assert_called_once_with(
                    rolling_window_size
                )
                mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_called_once()
            else:
                mock_monitor_instance.start_rolling_window_goodput_uploader.assert_not_called()
                mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_not_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_enabled_by_default(self, _):
        """Tests that monitoring is enabled by default (enable_monitoring=True)."""
        cfg = GoodputRecorder.default_config().set(
            name="test-default-enabled",
            upload_dir="/test",
            upload_interval=30,
            # Enable_monitoring defaults to True.
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        # Verify the flag defaults to True
        self.assertTrue(recorder.config.enable_monitoring)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test that cumulative goodput monitoring is active by default.
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_called()

            # Test that rolling window monitoring is active by default.
            with recorder._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_called()

            # Test that maybe_monitor_all is skipped
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_explicitly_disabled(self, _):
        """Tests that monitoring is disabled when enable_monitoring=False."""
        cfg = GoodputRecorder.default_config().set(
            name="test-explicitly-disabled",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=False,  # Explicitly disabled
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test cumulative goodput monitoring is skipped
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            # Test rolling window monitoring is skipped
            with recorder._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            # Test maybe_monitor_all is skipped
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_not_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_explicitly_enabled(self, _):
        """Tests that monitoring works when enable_monitoring=True."""
        cfg = GoodputRecorder.default_config().set(
            name="test-enabled",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=True,  # Explicitly enabled
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value

            # Test cumulative goodput monitoring works
            with recorder._maybe_monitor_goodput():
                pass

            # Should be called once for cumulative monitoring
            self.assertEqual(mock_monitor_cls.call_count, 1)
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            mock_monitor_instance.stop_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_works_with_monitoring_disabled(self, _):
        """Tests that record_event still works when monitoring is disabled."""
        cfg = GoodputRecorder.default_config().set(
            name="test-recording-only",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=False,  # Monitoring disabled
        )
        recorder = GoodputRecorder(cfg)

        # Verify that goodput recording still works (not monitoring/uploading)
        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value
            mock_instance.record_job_start_time = mock.MagicMock()
            mock_instance.record_job_end_time = mock.MagicMock()

            # Record event should work
            with recorder.record_event(measurement.EventType.JOB):
                pass

            # Verify goodput recording happened
            mock_recorder_cls.assert_called_once()
            mock_instance.record_job_start_time.assert_called_once()
            mock_instance.record_job_end_time.assert_called_once()

        # Verify no monitoring/uploading happened
        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_not_called()
