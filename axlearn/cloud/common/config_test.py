# Copyright © 2023 Apple Inc.

"""Tests config utils."""

import io
import os
import pathlib
from typing import Optional, Union
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud import ROOT_MODULE_NAME
from axlearn.cloud.common.config import (
    CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    _config_search_paths,
    _default_config_file,
    _get_or_prompt_project,
    _get_projects,
    _locate_user_config_file,
    _prompt,
    _prompt_choice,
    _prompt_project,
    _repo_root_or_cwd,
    load_configs,
)
from axlearn.cloud.common.config import main as config_main
from axlearn.cloud.common.config import update_configs, write_configs_with_header
from axlearn.common.test_utils import TestWithTemporaryCWD, temp_chdir


def _setup_fake_repo(temp_dir: Union[pathlib.Path, str]):
    assert os.path.isdir(temp_dir)
    os.makedirs(os.path.join(temp_dir, ".git"))


def create_default_config(
    directory: Union[pathlib.Path, str], *, contents: Optional[dict] = None
) -> pathlib.Path:
    f = pathlib.Path(directory) / CONFIG_DIR / DEFAULT_CONFIG_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()
    if contents:
        write_configs_with_header(str(f), contents)
    return f


class ConfigTest(TestWithTemporaryCWD):
    """Tests config utils."""

    def test_repo_root_or_cwd(self):
        temp_root = os.path.realpath(self._temp_root.name)
        os.makedirs("temp_cwd")
        with temp_chdir("temp_cwd"):
            self.assertNotEqual(temp_root, os.getcwd())
            self.assertEqual(_repo_root_or_cwd(), os.getcwd())
            _setup_fake_repo(temp_root)
            self.assertEqual(_repo_root_or_cwd(), temp_root)

    def test_default_config_file(self):
        temp_root = os.path.realpath(self._temp_root.name)
        # If no default config exists, return None.
        self.assertIsNone(_default_config_file())

        # Search relative to CWD if not in repo or package root.
        create_default_config(temp_root)
        default_config_file = pathlib.Path(_default_config_file())
        self.assertTrue(pathlib.Path.cwd() in default_config_file.parents)
        # Make sure default config is in the expected path.
        self.assertEqual(
            pathlib.Path(default_config_file).relative_to(pathlib.Path.cwd()),
            pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
        )

        # If within a repo, relative to repo root.
        repo_root = pathlib.Path(temp_root) / "repo_root"
        repo_root.mkdir()
        with temp_chdir(repo_root):
            _setup_fake_repo(repo_root)
            create_default_config(repo_root)
            default_config_file = pathlib.Path(_default_config_file())
            self.assertTrue(pathlib.Path(repo_root) in default_config_file.parents)
            # Make sure default config is in the expected path.
            self.assertEqual(
                default_config_file.relative_to(repo_root),
                pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
            )

        # If within a package, relative to package root.
        package_root = pathlib.Path(temp_root) / ROOT_MODULE_NAME
        package_root.mkdir()
        with temp_chdir(package_root):
            create_default_config(package_root)
            default_config_file = pathlib.Path(_default_config_file())
            self.assertTrue(pathlib.Path(package_root) in default_config_file.parents)
            # Make sure default config is in the expected path.
            self.assertEqual(
                default_config_file.relative_to(package_root),
                pathlib.Path(CONFIG_DIR) / DEFAULT_CONFIG_FILE,
            )

    def test_locate_user_config_file(self):
        temp_root = pathlib.Path(os.path.realpath(self._temp_root.name))
        _setup_fake_repo(temp_root)
        self.assertIsNone(_locate_user_config_file())

        with mock.patch("os.path.expanduser", return_value=os.path.join(temp_root, "user")):
            paths = _config_search_paths()
            paths = [pathlib.Path(path) for path in paths]
            for path in paths:
                self.assertTrue(temp_root in path.parents)

                # Create the file and verify that we can locate it.
                path.parent.mkdir(exist_ok=True, parents=True)
                path.touch()
                self.assertIsNotNone(_locate_user_config_file())
                path.unlink()  # Delete.

    def test_load_configs(self):
        temp_root = pathlib.Path(os.path.realpath(self._temp_root.name))
        _setup_fake_repo(temp_root)
        namespace = "test"

        # By default, we have no configs.
        config_file, configs = load_configs(namespace)
        self.assertIsNone(config_file)
        self.assertEqual(configs, {})

        # Create a default config.
        default_config = create_default_config(
            temp_root, contents={namespace: {"a": [123], "c": 123}}
        )

        # load_configs should pickup the default configs.
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(default_config))
        self.assertEqual(configs, {"a": [123], "c": 123})

        # Files in search paths take precedence.
        with mock.patch("os.path.expanduser", return_value=os.path.join(temp_root, "user")):
            paths = _config_search_paths()
            paths = [pathlib.Path(path) for path in paths]
            for path in paths:
                self.assertTrue(temp_root in path.parents)

                # Create a user config file.
                path.parent.mkdir(exist_ok=True, parents=True)
                write_configs_with_header(str(path), {namespace: {"a": [321], "b": {}}})

                # Verify that we can load it, and verify that the content is a union of the
                # default and user configs.
                config_file, configs = load_configs(namespace)
                self.assertEqual(config_file, str(path))
                self.assertEqual(configs, {"a": [321], "b": {}, "c": 123})

                # Delete.
                path.unlink()

    def test_load_configs_merge(self):
        temp_root = pathlib.Path(os.path.realpath(self._temp_root.name))
        _setup_fake_repo(temp_root)
        namespace = "test"

        # Create a default config.
        default_config = create_default_config(temp_root, contents={namespace: {"a": 1, "b": 2}})

        # Verify configs.
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(default_config))
        self.assertEqual(configs, {"a": 1, "b": 2})

        # Create a user config file.
        path = pathlib.Path(_config_search_paths()[0])
        path.parent.mkdir(exist_ok=True, parents=True)
        write_configs_with_header(str(path), {namespace: {"b": 3}})

        # Verify configs.
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(path))
        self.assertEqual(configs, {"a": 1, "b": 3})

        # Update default config file.
        write_configs_with_header(str(default_config), {namespace: {"a": 4, "b": 5}})

        # Verify configs.
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(path))
        self.assertEqual(configs, {"a": 4, "b": 3})

    def test_load_configs_required(self):
        temp_root = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_root)

        # By default, should fail because none exists.
        with self.assertRaises(SystemExit):
            load_configs(required=True)

        # Create a default config, which should get picked up.
        create_default_config(temp_root)
        load_configs(required=True)

    def test_update_configs(self):
        temp_root = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_root)
        namespace = "test"

        # Create a default config, which should get picked up.
        default_config = create_default_config(temp_root)
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(default_config))
        self.assertEqual(configs, {})

        # Update and check that it got saved to a different file.
        # The default config should not be modified.
        update_configs(namespace, {"test": 123})
        config_file, configs = load_configs(namespace)
        self.assertNotEqual(config_file, str(default_config))
        self.assertEqual(configs, {"test": 123})

    def test_update_configs_merge(self):
        temp_root = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_root)
        namespace = "test"

        # Create a user config.
        path = pathlib.Path(_config_search_paths()[0])
        path.parent.mkdir(exist_ok=True, parents=True)
        write_configs_with_header(str(path), {namespace: {"b": 3}})

        # Update and check that configs are merged, not replaced.
        update_configs(namespace, {"a": 1})
        config_file, configs = load_configs(namespace)
        self.assertEqual(config_file, str(path))
        self.assertEqual(configs, {"a": 1, "b": 3})


class CLITest(TestWithTemporaryCWD):
    """Tests CLI utils."""

    @parameterized.parameters(
        # Test basic required case.
        dict(required=True, inputs=[""], default=None, expected=StopIteration()),
        # Test basic success case.
        dict(required=True, inputs=["test"], default=None, expected="test"),
        # Test optional case.
        dict(required=False, inputs=[""], default=None, expected=None),
        # Test default case.
        dict(required=True, inputs=[""], default="test", expected="test"),
        dict(required=False, inputs=[""], default="test", expected="test"),
        # Test multiple input case.
        dict(required=True, inputs=["", "test"], default=None, expected="test"),
        dict(required=True, inputs=["", ""], default=None, expected=StopIteration()),
    )
    def test_prompt(self, inputs, required, default, expected):
        with mock.patch("builtins.input", side_effect=inputs):
            if isinstance(expected, Exception):
                with self.assertRaises(type(expected)):
                    _prompt("test", required=required, default=default)
            else:
                self.assertEqual(expected, _prompt("test", required=required, default=default))

    @parameterized.parameters(
        # Test no choices.
        dict(choices=[], inputs=[], expected=None),
        # Test success case.
        dict(choices=["a", "b"], inputs=["0"], expected=0),
        dict(choices=["a", "b"], inputs=["3", "-1", "1"], expected=1),
        # Test non-integer inputs.
        dict(choices=["a", "b"], inputs=["hello", "world", "1"], expected=1),
        # Test failure case.
        dict(choices=["a", "b"], inputs=[], expected=StopIteration()),
        dict(choices=["a", "b"], inputs=["hello"], expected=StopIteration()),
    )
    def test_prompt_choice(self, choices, inputs, expected):
        with mock.patch("builtins.input", side_effect=inputs):
            if isinstance(expected, Exception):
                with self.assertRaises(type(expected)):
                    _prompt_choice(choices)
            else:
                self.assertEqual(expected, _prompt_choice(choices))

    @parameterized.product(
        [
            dict(
                project_configs={
                    "project0": {"test": 0, "labels": ["a", "b"]},
                    "project1": {"test": 1},
                    "project2": {"test": 2, "labels": ["a"]},
                }
            ),
            dict(
                project_configs={
                    "project0": {"test": 0, "labels": "a, b"},
                    "project1": {"test": 1},
                    "project2": {"test": 2, "labels": "a"},
                }
            ),
            dict(
                project_configs={
                    "project0": {"test": 0, "labels": "  b, a "},
                    "project1": {"test": 1},
                    "project2": {"test": 2, "labels": "a"},
                }
            ),
        ],
        [
            # No labels will match all projects.
            dict(labels=None, expected=["project0", "project1", "project2"]),
            dict(labels=[], expected=["project0", "project1", "project2"]),
            # Test specifying labels as list.
            dict(labels=["a"], expected=["project0", "project2"]),
            dict(labels=["a", "a", "b"], expected=["project0"]),
            dict(labels=["b", "a"], expected=["project0"]),
            dict(labels=["c"], expected=[]),
        ],
    )
    def test_get_projects(self, project_configs, labels, expected):
        matched = _get_projects(project_configs, labels=labels)
        self.assertEqual(expected, list(matched.keys()))
        for k, v in matched.items():
            self.assertEqual(v, project_configs[k])

    @parameterized.parameters(
        # Test basic case.
        dict(inputs=["0"], expected="project0"),
        # Test repeat prompt.
        dict(inputs=["-1", "5", "1"], expected="project1"),
    )
    def test_prompt_project(self, inputs, expected):
        project_configs = {
            "project0": {"test": 0, "labels": ["a", "b"]},
            "project1": {"test": 1},
            "project2": {"test": 2, "labels": ["a"]},
        }
        with mock.patch("builtins.input", side_effect=inputs):
            self.assertEqual(expected, _prompt_project(project_configs))
            # When there's no choices, return None.
            self.assertEqual(None, _prompt_project({}))

    @parameterized.parameters(
        # Test basic case.
        dict(inputs=[], labels=["b"], expected="project0"),
        dict(inputs=["1"], labels=None, expected="project1"),
        # Test ambiguous case.
        dict(inputs=["0"], labels=None, expected="project0"),
        dict(inputs=["0"], labels=["a"], expected="project0"),
        dict(inputs=["-1"], labels=["a"], expected=StopIteration()),
    )
    def test_get_or_prompt_project(self, inputs, labels, expected):
        project_configs = {
            "project0": {"test": 0, "labels": ["a", "b"]},
            "project1": {"test": 1},
            "project2": {"test": 2, "labels": ["a"]},
        }
        with mock.patch("builtins.input", side_effect=inputs):
            if isinstance(expected, Exception):
                with self.assertRaises(type(expected)):
                    _get_or_prompt_project(project_configs, labels=labels)
            else:
                self.assertEqual(expected, _get_or_prompt_project(project_configs, labels=labels))

    def test_main(self):
        temp_root = os.path.realpath(self._temp_root.name)
        namespace = "test"
        project_configs = {
            "project0": {"test": 0, "labels": ["a", "b"]},
            "project1": {"test": 1},
            "project2": {"test": 2, "labels": ["a"]},
        }

        # Create a default config.
        create_default_config(temp_root, contents={namespace: project_configs})

        def _assert_list(labels, expected):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                fv = flags.FlagValues()
                flags.DEFINE_multi_string("label", labels, flag_values=fv, help="")
                fv.mark_as_parsed()
                config_main(["cli", "list"], namespace=namespace, fv=fv)
                self.assertEqual(expected, mock_stdout.getvalue())

        # fmt: off
        # List projects.
        _assert_list([], (
            "[ ] project0 [a, b]\n"
            "[ ] project1\n"
            "[ ] project2 [a]\n"
        ))

        # List projects with labels.
        _assert_list(["a"], (
            "[ ] project0 [a, b]\n"
            "[ ] project2 [a]\n"
        ))
        # fmt: on

        # Activate project without label.
        with mock.patch("builtins.input", side_effect=["-1", "1"]):
            fv = flags.FlagValues()
            flags.DEFINE_multi_string("label", [], flag_values=fv, help="")
            fv.mark_as_parsed()
            config_main(["cli", "activate"], namespace=namespace, fv=fv)

        # fmt: off
        # List projects.
        _assert_list([], (
            "[ ] project0 [a, b]\n"
            "[*] project1\n"
            "[ ] project2 [a]\n"
            "\n"
            "Settings of active project project1:\n"
            "\n"
            "test = 1\n"
            "\n"
        ))
        # fmt: on

        # Activate with label.
        with mock.patch("builtins.input", side_effect=["-1", "1"]):
            fv = flags.FlagValues()
            flags.DEFINE_multi_string("label", ["b"], flag_values=fv, help="")
            fv.mark_as_parsed()
            config_main(["cli", "activate"], namespace=namespace, fv=fv)

        # fmt: off
        # List projects.
        _assert_list([], (
            "[*] project0 [a, b]\n"
            "[ ] project1\n"
            "[ ] project2 [a]\n"
            "\n"
            "Settings of active project project0:\n"
            "\n"
            "test = 0\n"
            'labels = [ "a", "b",]\n'
            "\n"
        ))
        # fmt: on

    def test_main_action_get(self):
        temp_root = os.path.realpath(self._temp_root.name)
        namespace = "test"

        def _assert_get(setting, expected):
            cmd = ["cli", "get"]
            if setting is not None:
                cmd.append(setting)

            if isinstance(expected, Exception):
                with self.assertRaises(type(expected)):
                    config_main(cmd, namespace=namespace, fv=flags.FlagValues())
            else:
                with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                    config_main(cmd, namespace=namespace, fv=flags.FlagValues())
                self.assertEqual(expected, mock_stdout.getvalue())

        create_default_config(temp_root, contents={namespace: {"project0": {"foo", "bar"}}})
        _assert_get("foo", app.UsageError("Please activate a project first via config activate."))

        project_configs = {
            "_active": "project0",
            "project0": {"foo": "bar", "labels": ["a", "b"]},
            "project1": {"foo": "baz"},
        }
        create_default_config(temp_root, contents={namespace: project_configs})

        _assert_get("foo", "bar\n")
        _assert_get("labels", "['a', 'b']\n")
        _assert_get(None, app.UsageError("Usage: config get <setting_name>"))
        _assert_get("invalid_setting", app.UsageError("Unknown setting invalid_setting."))
