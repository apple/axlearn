# Copyright © 2024 Apple Inc.

"""Tests launch utilities."""
# pylint: disable=protected-access

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.job import Job
from axlearn.cloud.gcp.jobs.launch_utils import match_by_regex, serialized_flags_for_job


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    def test_serialized_flags_for_job(self):
        fv = flags.FlagValues()
        flags.DEFINE_string("test_discarded", None, "Test discarded flag", flag_values=fv)

        class DummyJob(Job):
            @classmethod
            def define_flags(cls, fv):
                flags.DEFINE_string("test_kept", "value", "Test kept flag", flag_values=fv)
                flags.DEFINE_multi_string(
                    "test_multi",
                    ["value1", "value2"],
                    "Test kept multi-flag",
                    flag_values=fv,
                )

        DummyJob.define_flags(fv)
        self.assertEqual(
            ["--test_kept=value", "--test_multi=value1", "--test_multi=value2"],
            serialized_flags_for_job(fv, job=DummyJob),
        )

    @parameterized.parameters(
        # Matches any "start" command.
        dict(
            matcher=match_by_regex(match_regex=dict(start=".*")),
            cases=[
                dict(action="start", instance_type="", expected=True),
                dict(action="start", instance_type="test type", expected=True),
                # Missing matcher for list.
                dict(action="list", instance_type="", expected=False),
            ],
        ),
        # Matches TPU types.
        dict(
            matcher=match_by_regex(match_regex=dict(start=r"v(\d)+.*-(\d)+", list="tpu")),
            cases=[
                dict(action="start", instance_type="v4-8", expected=True),
                dict(action="start", instance_type="v5litepod-16", expected=True),
                dict(action="start", instance_type="tpu", expected=False),
                dict(action="list", instance_type="tpu", expected=True),
            ],
        ),
    )
    def test_match_by_regex(self, matcher, cases):
        for case in cases:
            self.assertEqual(case["expected"], matcher(case["action"], case["instance_type"]))
