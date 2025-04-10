# Copyright © 2024 Apple Inc.

"""Tests runner utils."""

from absl.testing import parameterized

from axlearn.cloud.gcp.runners import utils as runner_utils


class UtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(tier=None, reservation=None, expected=False),
        # Demoted -- should be rescheduled.
        dict(tier=None, reservation="test", expected=True),
        # Demoted -- should be rescheduled.
        dict(tier="1", reservation="test", expected=True),
        # Demoted -- should be rescheduled.
        dict(tier=None, reservation="test", expected=True, is_pending=True),
        # Promoted -- do not reschedule. Instead, let pre-emption trigger reschedule.
        dict(tier="0", reservation=None, expected=False),
        # Promoted, but job is pending. Take this opportunity to reschedule.
        dict(tier="0", reservation=None, expected=True, is_pending=True),
    )
    def test_should_recreate_job(self, tier, reservation, expected, is_pending=False):
        self.assertEqual(
            expected, runner_utils.should_recreate_job(tier, reservation, is_pending=is_pending)
        )
