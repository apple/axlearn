# Copyright © 2023 Apple Inc.

"""Tests config utils."""

import os

from absl import flags

from axlearn.cloud.common import config
from axlearn.cloud.common.config_test import _setup_fake_repo, create_default_config
from axlearn.cloud.gcp import config as gcp_config
from axlearn.common.test_utils import TestWithTemporaryCWD


class ConfigTest(TestWithTemporaryCWD):
    """Tests config utils."""

    def test_gcp_settings(self):
        temp_dir = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_dir)

        flag_values = flags.FlagValues()
        flags.DEFINE_string("project", None, "The project name.", flag_values=flag_values)
        flags.DEFINE_string("zone", None, "The zone name.", flag_values=flag_values)
        flag_values.project = "test"
        flag_values.zone = "test"

        with self.assertRaisesRegex(RuntimeError, expected_regex="fv must be parsed"):
            gcp_config.gcp_settings("bucket", required=False, fv=flag_values)

        flag_values.mark_as_parsed()

        self.assertEqual("test", gcp_config.gcp_settings("project", fv=flag_values))
        self.assertEqual("test", gcp_config.gcp_settings("zone", fv=flag_values))

        # By default, should fail because no config file exists.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("bucket", fv=flag_values)

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("bucket", required=False, fv=flag_values))

        # Should not fail if a default exists.
        self.assertEqual(
            "default",
            gcp_config.gcp_settings("bucket", required=True, default="default", fv=flag_values),
        )

        # Create a default config, which should get picked up.
        default_config = create_default_config(temp_dir)

        # Should fail because no config for --project and --zone.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("bucket", fv=flag_values)

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("bucket", required=False, fv=flag_values))

        # Write some values to the config.
        config.write_configs_with_header(
            str(default_config),
            {
                gcp_config.CONFIG_NAMESPACE: {
                    "test:test": {"project": "test", "zone": "test", "bucket": "test-bucket"}
                }
            },
        )

        # Should fail because key cannot be found.
        with self.assertRaises(SystemExit):
            gcp_config.gcp_settings("unknown_key", fv=flag_values)

        # Should not fail if not required.
        self.assertIsNone(gcp_config.gcp_settings("unknown_key", fv=flag_values, required=False))

        # Should succeed.
        self.assertEqual("test-bucket", gcp_config.gcp_settings("bucket", fv=flag_values))

    def test_gcp_settings_with_active_config(self):
        temp_dir = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_dir)

        flag_values = flags.FlagValues()
        # Flags do not set project/zone.
        flags.DEFINE_string("project", None, "The project name.", flag_values=flag_values)
        flags.DEFINE_string("zone", None, "The zone name.", flag_values=flag_values)
        flag_values.mark_as_parsed()

        self.assertIsNone(gcp_config.default_project())
        self.assertIsNone(gcp_config.default_zone())

        # Create a default config, which should get picked up.
        default_config = create_default_config(temp_dir)
        # Write some values to the config.
        config.write_configs_with_header(
            str(default_config),
            {
                gcp_config.CONFIG_NAMESPACE: {
                    "_active": "test-proj:test-zone",
                    "test-proj:test-zone": {
                        "project": "test-proj",
                        "zone": "test-zone",
                        "bucket": "test-bucket",
                    },
                }
            },
        )
        self.assertEqual("test-proj", gcp_config.default_project())
        self.assertEqual("test-zone", gcp_config.default_zone())

        # We follow the default config.
        self.assertEqual("test-proj", gcp_config.gcp_settings("project", fv=flag_values))
        self.assertEqual("test-zone", gcp_config.gcp_settings("zone", fv=flag_values))
        self.assertEqual("test-bucket", gcp_config.gcp_settings("bucket", fv=flag_values))
