# Copyright © 2024 Apple Inc.

"""Tests measurement utils."""
# pylint: disable=protected-access

import contextlib
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.common import measurement
from axlearn.experiments.testdata.axlearn_common_measurement_test.dummy_recorder import (
    DummyRecorder as RealDummyRecorder,
)


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    def setUp(self):
        super().setUp()
        self._orig_recorder = measurement.global_recorder
        self._orig_recorders = measurement._recorders.copy()
        measurement.global_recorder = None
        measurement._recorders = {}

    def tearDown(self):
        super().tearDown()
        measurement.global_recorder = self._orig_recorder
        measurement._recorders = self._orig_recorders

    def test_register(self):
        self.assertEqual({}, measurement._recorders)

        @measurement.register_recorder("test")
        class DummyRecorder(measurement.Recorder):
            pass

        self.assertEqual(DummyRecorder, measurement._recorders.get("test"))

        with self.assertRaisesRegex(ValueError, "already registered"):
            measurement.register_recorder("test")(DummyRecorder)

    @parameterized.parameters(
        dict(recorder_type=None),
        dict(recorder_type="test"),
        dict(
            recorder_type=(
                "axlearn.experiments.testdata.axlearn_common_measurement_test.dummy_recorder:"
                "dummy_recorder"
            )
        ),
    )
    def test_initialize(self, recorder_type):
        mock_recorder_cls = mock.MagicMock()
        mock_recorder_instance = mock_recorder_cls.from_flags.return_value
        mock_recorder_instance.record_event.return_value = contextlib.nullcontext()
        measurement.register_recorder("test")(mock_recorder_cls)
        measurement.register_recorder("dummy_recorder")(RealDummyRecorder)

        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default("recorder_type", recorder_type)
        fv.mark_as_parsed()

        self.assertIsNone(measurement.global_recorder)
        measurement.initialize(fv)

        if recorder_type is None:
            self.assertIsNone(measurement.global_recorder)
            return

        recorder_name = recorder_type.split(":", 1)[-1]
        if recorder_name == "test":
            self.assertEqual(mock_recorder_instance, measurement.global_recorder)
            mock_recorder_cls.from_flags.assert_called_once()
        elif recorder_name == "dummy_recorder":
            self.assertIsNotNone(measurement.global_recorder)
            self.assertIsInstance(measurement.global_recorder, RealDummyRecorder)

        with mock.patch.object(
            measurement.global_recorder, "start_monitoring"
        ) as mock_start_monitoring:
            measurement.start_monitoring()
            mock_start_monitoring.assert_called_once()
