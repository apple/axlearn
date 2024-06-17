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

    @parameterized.parameters(None, ["name=test-name"])
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

    def test_record(self):
        fv = flags.FlagValues()
        measurement.define_flags(flag_values=fv)
        fv.set_default("recorder_spec", ["name=test-name"])
        fv.mark_as_parsed()

        recorder = GoodputRecorder.from_flags(fv)
        recorder._recorder = mock.MagicMock()
        recorder.record(measurement.Event.START_JOB)
        self.assertTrue(recorder._recorder.record_job_start_time.called)
