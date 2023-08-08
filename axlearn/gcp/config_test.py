# Copyright © 2023 Apple Inc.

"""Tests config utils."""

import os
import pathlib
import tempfile
from typing import Union
from unittest import mock

from absl.testing import absltest

from axlearn.gcp import ROOT_MODULE_NAME, config
from axlearn.gcp.config import (
    CONFIG_DIR,
    CONFIG_NAMESPACE,
    DEFAULT_CONFIG_FILE,
    _config_search_paths,
    _default_config_file,
    _locate_user_config_file,
    _repo_root_or_cwd,
    _update_configs,
    gcp_settings,
    load_configs,
    write_configs_with_header,
)


def _setup_fake_repo(temp_dir: Union[pathlib.Path, str]):
    assert os.path.isdir(temp_dir)
    os.makedirs(os.path.join(temp_dir, ".git"))


def _create_default_config(directory: Union[pathlib.Path, str]) -> pathlib.Path:
    f = pathlib.Path(directory) / CONFIG_DIR / DEFAULT_CONFIG_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()
    return f


class ConfigTest(absltest.TestCase):
    """Tests config utils."""

    @mock.patch(f"{config.__name__}._flag_values", return_value={"project": "test", "zone": "test"})
    def test_gcp_settings(self, flag_values):  # pylint: disable=unused-argument
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)

            # By default, should fail because no config file exists.
            with self.assertRaises(SystemExit):
                gcp_settings("project")

            # Create a default config, which should get picked up.
            default_config = _create_default_config(temp_dir)

            # Should fail because no config for --project and --zone.
            with self.assertRaises(SystemExit):
                gcp_settings("project")

            # Write some values to the config.
            write_configs_with_header(
                str(default_config),
                {CONFIG_NAMESPACE: {"test:test": {"project": "test", "zone": "test"}}},
            )

            # Should fail because key cannot be found.
            with self.assertRaises(SystemExit):
                gcp_settings("unknown_key")

            # Should succeed.
            self.assertEqual(gcp_settings("project"), "test")

    def test_repo_root_or_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            os.chdir(temp_dir)
            self.assertEqual(_repo_root_or_cwd(), os.getcwd())
            _setup_fake_repo(temp_dir)
            self.assertEqual(_repo_root_or_cwd(), temp_dir)

    def test_default_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            os.chdir(temp_dir)
            # If no default config exists, return None.
            self.assertIsNone(_default_config_file())

            # Search relative to CWD if not in repo or package root.
            _create_default_config(temp_dir)
            default_config_file = pathlib.Path(_default_config_file())
            self.assertTrue(pathlib.Path.cwd() in default_config_file.parents)
            # Make sure default config is in the expected path.
            self.assertEqual(
                pathlib.Path(default_config_file).relative_to(pathlib.Path.cwd()),
                pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
            )

            # If within a repo, relative to repo root.
            repo_root = pathlib.Path(temp_dir) / "repo_root"
            repo_root.mkdir()
            os.chdir(repo_root)
            _setup_fake_repo(repo_root)
            _create_default_config(repo_root)
            default_config_file = pathlib.Path(_default_config_file())
            self.assertTrue(pathlib.Path(repo_root) in default_config_file.parents)
            # Make sure default config is in the expected path.
            self.assertEqual(
                default_config_file.relative_to(repo_root),
                pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
            )

            # If within a package, relative to package root.
            package_root = pathlib.Path(temp_dir) / ROOT_MODULE_NAME
            package_root.mkdir()
            os.chdir(package_root)
            _create_default_config(package_root)
            default_config_file = pathlib.Path(_default_config_file())
            self.assertTrue(pathlib.Path(package_root) in default_config_file.parents)
            # Make sure default config is in the expected path.
            self.assertEqual(
                default_config_file.relative_to(package_root),
                pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
            )

    def test_locate_user_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)
            temp_dir = pathlib.Path(temp_dir)
            self.assertIsNone(_locate_user_config_file())

            with mock.patch("os.path.expanduser", return_value=os.path.join(temp_dir, "user")):
                paths = _config_search_paths()
                paths = [pathlib.Path(path) for path in paths]
                for path in paths:
                    self.assertTrue(temp_dir in path.parents)

                    # Create the file and verify that we can locate it.
                    path.parent.mkdir(exist_ok=True, parents=True)
                    path.touch()
                    self.assertIsNotNone(_locate_user_config_file())
                    path.unlink()  # Delete.

    def test_load_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)
            temp_dir = pathlib.Path(temp_dir)

            # By default, we have no configs.
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertIsNone(config_file)
            self.assertEqual(configs, {})

            # Create a default config.
            default_config = _create_default_config(temp_dir)
            write_configs_with_header(
                str(default_config), {CONFIG_NAMESPACE: {"a": [123], "c": 123}}
            )

            # load_configs should pickup the default configs.
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertEqual(config_file, str(default_config))
            self.assertEqual(configs, {"a": [123], "c": 123})

            # Files in search paths take precedence.
            with mock.patch("os.path.expanduser", return_value=os.path.join(temp_dir, "user")):
                paths = _config_search_paths()
                paths = [pathlib.Path(path) for path in paths]
                for path in paths:
                    self.assertTrue(temp_dir in path.parents)

                    # Create a user config file.
                    path.parent.mkdir(exist_ok=True, parents=True)
                    write_configs_with_header(str(path), {CONFIG_NAMESPACE: {"a": [321], "b": {}}})

                    # Verify that we can load it, and verify that the content is a union of the
                    # default and user configs.
                    config_file, configs = load_configs(CONFIG_NAMESPACE)
                    self.assertEqual(config_file, str(path))
                    self.assertEqual(configs, {"a": [321], "b": {}, "c": 123})

                    # Delete.
                    path.unlink()

    def test_load_configs_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)
            temp_dir = pathlib.Path(temp_dir)

            # Create a default config.
            default_config = _create_default_config(temp_dir)
            write_configs_with_header(str(default_config), {CONFIG_NAMESPACE: {"a": 1, "b": 2}})

            # Verify configs.
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertEqual(config_file, str(default_config))
            self.assertEqual(configs, {"a": 1, "b": 2})

            # Create a user config file.
            path = pathlib.Path(_config_search_paths()[0])
            path.parent.mkdir(exist_ok=True, parents=True)
            write_configs_with_header(str(path), {CONFIG_NAMESPACE: {"b": 3}})

            # Verify configs.
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertEqual(config_file, str(path))
            self.assertEqual(configs, {"a": 1, "b": 3})

            # Update default config file.
            write_configs_with_header(str(default_config), {CONFIG_NAMESPACE: {"a": 4, "b": 5}})

            # Verify configs.
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertEqual(config_file, str(path))
            self.assertEqual(configs, {"a": 4, "b": 3})

    def test_load_configs_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)

            # By default, should fail because none exists.
            with self.assertRaises(SystemExit):
                load_configs(required=True)

            # Create a default config, which should get picked up.
            _create_default_config(temp_dir)
            load_configs(required=True)

    def test_update_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            _setup_fake_repo(temp_dir)
            os.chdir(temp_dir)

            # Create a default config, which should get picked up.
            default_config = _create_default_config(temp_dir)
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertEqual(config_file, str(default_config))
            self.assertEqual(configs, {})

            # Update and check that it got saved to a different file.
            # The default config should not be modified.
            _update_configs(CONFIG_NAMESPACE, {"test": 123})
            config_file, configs = load_configs(CONFIG_NAMESPACE)
            self.assertNotEqual(config_file, str(default_config))
            self.assertEqual(configs, {"test": 123})
