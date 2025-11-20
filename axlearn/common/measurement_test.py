# Copyright © 2024 Apple Inc.

"""Tests measurement utils."""
# pylint: disable=protected-access

from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.common import measurement, measurement_base


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    def setUp(self):
        self._orig_recorder = measurement.global_recorder
        self._orig_recorders = measurement_base._recorders.copy()
        measurement.global_recorder = None

    def tearDown(self):
        measurement.global_recorder = self._orig_recorder
        measurement_base._recorders.clear()
        measurement_base._recorders.update(self._orig_recorders)

    def test_register(self):
        self.assertNotIn("test_register_dummy", measurement_base._recorders)

        @measurement.register_recorder("test_register_dummy")
        class DummyRecorder(measurement.Recorder):
            pass

        self.assertEqual(DummyRecorder, measurement_base._recorders.get("test_register_dummy"))

        # Registering twice should fail.
        with self.assertRaisesRegex(ValueError, "already registered"):
            measurement.register_recorder("test_register_dummy")(DummyRecorder)

    @parameterized.parameters(
        # No-op if no recorder_type provided.
        dict(
            recorder_type=None,
            expected=None,
        ),
        dict(
            recorder_type="test_initialize_mock",
            expected="Mock",
        ),
        # Try initializing from another module.
        dict(
            recorder_type=(
                f"axlearn.experiments.testdata.{__name__.replace('.', '_')}.dummy_recorder:"
                "dummy_recorder"
            ),
            expected="DummyRecorder",
        ),
    )
    def test_initialize(self, recorder_type, expected):
        mock_recorder = mock.MagicMock()
        measurement.register_recorder("test_initialize_mock")(mock_recorder)

        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default("recorder_type", recorder_type)
        fv.mark_as_parsed()

        self.assertIsNone(measurement.global_recorder)
        measurement.initialize(fv)

        if recorder_type is None:
            # global_recorder should not be initialized, and record_event should be no-op.
            self.assertIsNone(measurement.global_recorder)
            measurement.record_event(measurement.Event.START_JOB)
            return

        recorder_name = recorder_type.split(":", 1)[-1]
        if recorder_name == "test_initialize_mock":
            self.assertTrue(mock_recorder.from_flags.called)

        self.assertIn(expected, str(measurement_base._recorders.get(recorder_name, None)))
        self.assertIn(expected, str(measurement.global_recorder))

        # Ensure that record_event does not fail.
        with mock.patch.object(measurement.global_recorder, "record") as mock_record:
            measurement.record_event(measurement.Event.START_JOB)
            self.assertIn(measurement.Event.START_JOB, mock_record.call_args[0])

        # Ensure that start_monitoring does not fail.
        with mock.patch.object(
            measurement.global_recorder, "start_monitoring"
        ) as mock_start_monitoring:
            measurement.start_monitoring()
            mock_start_monitoring.assert_called_once()

        # Ensure that maybe_monitor_all does not fail (just enter and exit context).
        with measurement.global_recorder.maybe_monitor_all():
            pass

    def test_initialize_gcp_goodput_recorder(self):
        """Test that GCP goodput recorder can be dynamically imported."""
        from axlearn.cloud.gcp import measurement as _  # pylint: disable=import-outside-toplevel

        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default("recorder_type", "axlearn.cloud.gcp.measurement:goodput")
        fv.set_default(
            "recorder_spec",
            [
                "name=test-goodput",
                "upload_dir=/tmp/test",
                "upload_interval=30",
            ],
        )
        fv.mark_as_parsed()

        self.assertIsNone(measurement.global_recorder)
        measurement.initialize(fv)

        self.assertIsNotNone(measurement.global_recorder)
        self.assertIn("GoodputRecorder", str(type(measurement.global_recorder)))
