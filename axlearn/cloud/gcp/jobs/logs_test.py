# Copyright © 2024 Apple Inc.

"""Tests Cloud Logging utils."""
# pylint: disable=protected-access

from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp.jobs import logs
from axlearn.common.test_utils import TestWithTemporaryCWD


class MainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_private_flags(self):
        fv = flags.FlagValues()
        logs._private_flags(flag_values=fv)
        # Basic sanity check.
        self.assertIsNone(fv["name"].default)
        self.assertIsNotNone(fv["worker"].default)

    @parameterized.parameters(
        dict(start_time=None, end_time=None),
        dict(start_time="start_time", end_time="end_time"),
    )
    def test_logs(self, start_time, end_time):
        fv = flags.FlagValues()
        logs._private_flags(flag_values=fv)
        fv.set_default("name", "test-job")
        fv.set_default("start_time", start_time)
        fv.set_default("end_time", end_time)
        fv.mark_as_parsed()

        mock_client = mock.Mock()
        mock_client.list_entries.return_value = []

        with mock.patch(f"{logs.__name__}._logging_client", return_value=mock_client):
            logs.main(["cli"], flag_values=fv)
            _, kwargs = mock_client.list_entries.call_args
            self.assertIn("test-job", kwargs["filter_"])
            if start_time:
                self.assertIn('timestamp>="start_time"', kwargs["filter_"])
            if end_time:
                self.assertIn('timestamp<="end_time"', kwargs["filter_"])
