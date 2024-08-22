# Copyright © 2023 Apple Inc.

"""Utilities to configure cloud projects.

A configuration file is used for persisting project settings. Configs are namespaced so that
multiple modules can share the same config file.

The config file should be named `.axlearn.config`, and can live in one of the following paths
(searched in order):
1. In `.axlearn/.axlearn.config` relative to the repo root.
2. In `~/.axlearn.config`, which can be used by users who `pip install` the project and may be
   invoking the CLI from outside the repo.
"""

import os
import sys
from collections.abc import Sequence
from typing import Any, Optional

import toml
from absl import app, flags, logging

from axlearn.cloud.common import utils

CONFIG_DIR = ".axlearn"  # Relative to root of project.
CONFIG_FILE = ".axlearn.config"
DEFAULT_CONFIG_FILE = "axlearn.default.config"


def load_configs(
    namespace: Optional[str] = None, required: bool = False
) -> tuple[Optional[str], dict]:
    """Loads configs for the given namespace.

    The strategy is as follows:
    1. Attempt to read the default config file;
    2. Attempt to search for an existing config file in a config search path, and use it to override
       the default config;
    3. If neither config file was found, return an empty config.

    Args:
        namespace: Namespace to group configs under. If None, returns the configs of all namespaces.
        required: Whether we require a config to exist. If True, sys.exit is raised if no config
            file is found.

    Returns:
        A tuple of (config_file, namespace_configs). If no config file could be found and required
        is False, config_file will be None and namespace_configs will be an empty dict.
    """
    configs = {}
    config_file = None

    # If a default config exists, read it.
    default_config_file = _default_config_file()
    if default_config_file is not None:
        utils.merge(configs, toml.load(default_config_file))
        config_file = default_config_file

    # If a user config file exists, use it to override the default.
    user_config_file = _locate_user_config_file()
    if user_config_file:
        utils.merge(configs, toml.load(user_config_file))
        config_file = user_config_file

    if required and config_file is None:
        logging.error(
            "A config file could not be found; please create one first. "
            "Please refer to the docs for instructions: "
            "https://github.com/apple/axlearn/blob/main/docs/01-start.md#preparing-the-cli. "
            "Please also make sure the config file can be found in one of: %s",
            _config_search_paths(),
        )
        sys.exit(1)

    if namespace is not None:
        configs = configs.get(namespace, {})
    return config_file, configs


def write_configs_with_header(config_file: str, configs: dict[str, Any]):
    """Writes configs to a file, with a prepended comment warning users not to modify it.

    Args:
        config_file: Output file path.
        configs: Configs to serialize.
    """
    header = "# WARNING: This is a partially generated file. Modify with care.\n"
    body = toml.dumps(configs)
    with open(config_file, "w", encoding="utf-8") as f:
        f.seek(0, 0)
        f.write(f"{header}\n{body}")


def update_configs(namespace: str, namespace_configs: dict[str, Any]):
    """Update configs for the given namespace.

    Args:
        namespace: Config namespace.
        namespace_configs: Configs to write.
    """
    config_file = _locate_user_config_file()
    if config_file is None:
        # Create a user config in a search path.
        config_file = _config_search_paths()[0]
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, mode="a", encoding="utf-8"):  # Touch.
            pass
    configs = toml.load(config_file) or {}
    # Update namespace configs by merging.
    configs[namespace] = utils.merge(configs.get(namespace, {}), namespace_configs)
    write_configs_with_header(config_file, configs)
    print(f"Configs written to {config_file}")


def _repo_root_or_cwd() -> str:
    """Gets the repo root from CWD, or return CWD if not in a repo.

    Note that this repo can be arbitrary, and may not contain `ROOT_MODULE_NAME`.
    """
    cwd = curr = os.getcwd()
    while curr and curr != "/":
        if os.path.exists(os.path.join(curr, ".git")):
            return curr
        curr = os.path.dirname(curr)
    return cwd


def _default_config_file() -> Optional[str]:
    """Looks up a default config file from package, repo, or relative to CWD."""
    default_config = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_FILE)

    # Check if default config exists in repo root or CWD.
    repo_or_cwd_config = os.path.abspath(os.path.join(_repo_root_or_cwd(), default_config))
    if os.path.exists(repo_or_cwd_config):
        logging.log_first_n(logging.INFO, "Found default config at %s", 1, repo_or_cwd_config)
        return repo_or_cwd_config

    # Check if running from package.
    try:
        package_default_config = os.path.join(utils.get_package_root(), default_config)
        if os.path.exists(package_default_config):
            logging.log_first_n(
                logging.INFO, "Found default config at %s", 1, package_default_config
            )
            return package_default_config
    except ValueError:
        pass

    return None


def _config_search_paths() -> Sequence[str]:
    """Paths to search for config file, ordered by precedence."""
    search_paths = [
        os.path.abspath(os.path.join(_repo_root_or_cwd(), CONFIG_DIR, CONFIG_FILE)),
        os.path.join(os.path.expanduser("~"), CONFIG_FILE),
    ]
    assert _default_config_file() not in search_paths
    return search_paths


def _locate_user_config_file() -> Optional[str]:
    """Looks for the user's config file in the search paths, or returns None if not found.

    A user config file may not exist if e.g. the user has never invoked `activate`.
    """
    search_paths = _config_search_paths()
    config_file = None
    for path in search_paths:
        if os.path.exists(path):
            logging.log_first_n(logging.INFO, "Found user config at %s", 1, path)
            config_file = path
            break
    return config_file


def _prompt(field: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """Prompts a user for a value.

    Args:
        field: The field name to display.
        required: If True, prompts until user provides a truthy value.
        default: Default value to use if falsey value provided.

    Returns:
        The user's response. Can be falsey if required=False.
    """
    value = input(field).strip().strip('"')
    if not value:
        value = default
    while required and not value:
        value = input(field).strip().strip('"')
    return value


def _prompt_choice(choices: Sequence[str]) -> Optional[int]:
    """Prompts the user to select a choice among multiple.

    Args:
        choices: Possible choices.

    Returns:
        The selected index.
    """
    if len(choices) == 0:
        logging.warning("No available options.")
        return None

    for i, choice in enumerate(choices):
        print(f"[{i}] {choice}")

    def _valid_choice(choice):
        try:
            return 0 <= int(choice) < len(choices)
        except (ValueError, TypeError) as e:
            logging.debug("%s", e)
            return False

    choice = _prompt("Select choice: ")
    while not _valid_choice(choice):
        print(f"Invalid choice: {choice}")
        choice = _prompt("Select choice: ")
    return int(choice)


def _get_projects(
    project_configs: dict[str, Any], *, labels: Optional[Sequence[str]] = None
) -> dict[str, Any]:
    """Gets configured project configs, optionally filtering by labels.

    Args:
        project_configs: A mapping from projects to project-specific configs. Keys prefixed with `_`
            are considered internal and always filtered out.
        labels: Optional labels to use for filtering. A project config is considered a match if it
            has a "labels" field (either a sequence or comma separated string) containing all of the
            values supplied in `labels`.

    Returns:
        A dict of key, values from `project_configs` matching the given `labels`.
    """
    projects = {}
    for key, project_cfg in project_configs.items():
        if key.startswith("_"):
            continue
        if labels is not None:
            project_labels = project_cfg.get("labels", [])
            if isinstance(project_labels, str):
                project_labels = [label.strip() for label in project_labels.split(",")]
            if not set(labels).issubset(set(project_labels)):
                continue
        projects[key] = project_cfg
    return projects


def _prompt_project(project_configs: dict[str, Any]) -> Optional[str]:
    """Prompts the user to select a project among the existing projects.

    Args:
        projects: Mapping from project key to project-specific configs.

    Returns:
        The selected key, or None if no project_configs or the user made an invalid choice.
    """
    choices = []
    for project, config in project_configs.items():
        choice = project
        if labels := config.get("labels", None):
            choice += f" [{labels if isinstance(labels, str) else ', '.join(labels)}]"
        choices.append(choice)
    choice = _prompt_choice(choices)
    if choice is None:
        return None
    keys = list(project_configs.keys())
    assert 0 <= choice < len(keys)  # _prompt_choice should give a valid choice.
    return keys[choice]


def _get_or_prompt_project(
    project_configs: dict[str, Any], *, labels: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Gets the project with the provided label(s), or prompt for user to select one if ambiguous.

    See `get_project` and `_prompt_project` for details.
    """
    options = _get_projects(project_configs, labels=labels)
    if len(options) == 0:
        print(f"No projects matched labels: {labels}")
        project = None
    elif len(options) > 1:
        print(f"Multiple projects matched labels: {labels}")
        project = _prompt_project(options)
    else:
        project = list(options.keys())[0]
    return project


def config_flags():
    """Config CLI flags."""
    flags.DEFINE_multi_string("label", None, "Label(s) for filtering list and activate")


def main(argv: Sequence[str], *, namespace: str, fv: flags.FlagValues):
    """Entrypoint for interacting with projects in a config namespace."""
    action = utils.parse_action(
        argv, options=["list", "activate", "cleanup", "get"], default="list"
    )

    # Load existing configs, if any.
    _, project_configs = load_configs(namespace)
    updates = {}

    if action == "list":
        active_project = project_configs.get("_active", None)
        for project, project_config in _get_projects(
            project_configs, labels=fv.get_flag_value("label", None)
        ).items():
            prefix = "[*]" if project == active_project else "[ ]"
            project_str = f"{prefix} {project}"
            if labels := project_config.get("labels", None):
                project_str += f" [{labels if isinstance(labels, str) else ', '.join(labels)}]"
            print(project_str)

        if active_project is not None:
            print()
            print(f"Settings of active project {active_project}:")
            print()
            print(toml.dumps(project_configs[active_project]))
        return

    elif action == "activate":
        project = _get_or_prompt_project(project_configs, labels=fv.get_flag_value("label", None))
        if project is None:
            return
        print(f"Setting {project} to active.")
        updates["_active"] = project

    elif action == "cleanup":
        choice = input("Warning: delete all config files? [y/n] ")
        if choice.lower() != "y":
            print("Aborting.")
            return
        for path in _config_search_paths():
            if os.path.exists(path):
                print(f"Found {path}, deleting.")
                os.remove(path)
        print("Cleanup complete.")
        return

    elif action == "get":
        if len(argv) < 3:
            raise app.UsageError("Usage: config get <setting_name>")
        setting_name = argv[2]
        active_project = project_configs.get("_active", None)
        if not active_project:
            raise app.UsageError("Please activate a project first via config activate.")
        active_settings = project_configs[active_project]
        setting = active_settings.get(setting_name, None)
        if setting is None:
            raise app.UsageError(
                f"Unknown setting {setting_name}. View available settings using config list."
            )
        print(setting)
        return

    else:
        logging.warning("Unrecognized action: %s. Please rerun with --help.", action)
        return

    update_configs(namespace, updates)
