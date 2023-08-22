# Copyright © 2023 Apple Inc.

"""Tests config utils."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest

from axlearn.cloud.common import config
from axlearn.cloud.common.config_test import _setup_fake_repo, create_default_config
from axlearn.cloud.gcp import config as gcp_config


class ConfigTest(absltest.TestCase):
    """Tests config utils."""

    @mock.patch(
        f"{gcp_config.__name__}._flag_values", return_value={"project": "test", "zone": "test"}
    )
    def test_gcp_settings(self, flag_values):
        del flag_values

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)

            # By default, should fail because no config file exists.
            with self.assertRaises(SystemExit):
                gcp_config.gcp_settings("project")

            # Create a default config, which should get picked up.
            default_config = create_default_config(temp_dir)

            # Should fail because no config for --project and --zone.
            with self.assertRaises(SystemExit):
                gcp_config.gcp_settings("project")

            # Write some values to the config.
            config.write_configs_with_header(
                str(default_config),
                {gcp_config.CONFIG_NAMESPACE: {"test:test": {"project": "test", "zone": "test"}}},
            )

            # Should fail because key cannot be found.
            with self.assertRaises(SystemExit):
                gcp_config.gcp_settings("unknown_key")

            # Should succeed.
            self.assertEqual(gcp_config.gcp_settings("project"), "test")
