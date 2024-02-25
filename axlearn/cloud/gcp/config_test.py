# Copyright © 2023 Apple Inc.

"""Tests config utils."""

import os
from unittest import mock

from absl import flags

from axlearn.cloud.common import config
from axlearn.cloud.common.config_test import _setup_fake_repo, create_default_config
from axlearn.cloud.gcp import config as gcp_config
from axlearn.common.test_utils import TestWithTemporaryCWD


class ConfigTest(TestWithTemporaryCWD):
    """Tests config utils."""

    @mock.patch(
        f"{gcp_config.__name__}._flag_values", return_value={"project": "test", "zone": "test"}
    )
    def test_gcp_settings(self, flag_values):
        del flag_values

        temp_dir = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_dir)

        with self.assertRaisesRegex(RuntimeError, expected_regex="FLAGS must be parsed"):
            gcp_config.gcp_settings("project", required=False)

        flags.FLAGS.mark_as_parsed()

        # By default, should fail because no config file exists.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("project")

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("project", required=False))

        # Should not fail if a default exists.
        self.assertEqual(
            "default", gcp_config.gcp_settings("project", required=True, default="default")
        )

        # Create a default config, which should get picked up.
        default_config = create_default_config(temp_dir)

        # Should fail because no config for --project and --zone.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("project")

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("project", required=False))

        # Write some values to the config.
        config.write_configs_with_header(
            str(default_config),
            {gcp_config.CONFIG_NAMESPACE: {"test:test": {"project": "test", "zone": "test"}}},
        )

        # Should fail because key cannot be found.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("unknown_key")

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("unknown_key", required=False))

        # Should succeed.
        self.assertEqual(gcp_config.gcp_settings("project"), "test")
