# Copyright © 2023 Apple Inc.

"""Tests bastion VM."""
# pylint: disable=protected-access

from absl import flags

from axlearn.cloud.gcp.jobs import bastion_vm
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.test_utils import TestWithTemporaryCWD


class MainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_private_flags(self):
        with mock_gcp_settings(bastion_vm.__name__, default_mock_settings()):
            fv = flags.FlagValues()
            bastion_vm._private_flags(flag_values=fv)
            # Basic sanity check.
            self.assertIsNotNone(fv["project"].default)
            self.assertIsNotNone(fv["zone"].default)
            self.assertIsNotNone(fv["env_id"].default)
