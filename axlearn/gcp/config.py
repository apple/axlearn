# Copyright © 2023 Apple Inc.

"""Configure GCP projects.

A configuration file is used for persisting user-specific configs. Configs are namespaced so that
multiple modules can eventually share the same config file. This module manipulates the "gcp"
namespace of the config.

The config file should be named `.axlearn.config`, and can live in one of the following paths
(searched in order):
1. In `.axlearn/.axlearn.config` relative to the repo root.
2. In `~/.axlearn.config`, which can be used by users who `pip install` the project and may be
   invoking the CLI from outside the repo.

Possible actions: [list|activate|cleanup]

For `list`:
- Lists all configured projects, as well as their labels.
- Specify `--label <label>` to filter among projects. Multiple labels can be specified by repeating
  the flag. The resulting projects will match all labels.

For `activate`:
- Sets a project configuration to be "active". This means that all commands invoked from CLI will
  use the given project, zone, etc. associated with the active config.
- Specify `--label <label>` to narrow down the list of projects to consider for activation. If
  there are multiple candidates, starts an interactive prompt for the user to select one.

For `cleanup`:
- Removes all configuration files. Will prompt for confirmation.
"""

import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import toml
from absl import app, flags, logging

from axlearn.gcp.utils import get_package_root, infer_cli_name

FLAGS = flags.FLAGS
CONFIG_DIR = ".axlearn"  # Relative to root of project.
CONFIG_FILE = ".axlearn.config"
DEFAULT_CONFIG_FILE = "axlearn.default.config"
CONFIG_NAMESPACE = "gcp"


def _private_flags():
    flags.DEFINE_multi_string("label", None, "Label(s) for filtering list and activate")


def _flag_values() -> Dict[str, Any]:
    return FLAGS.flag_values_dict()


def gcp_settings(
    key: str, *, default: Optional[Any] = None, required: bool = True
) -> Optional[str]:
    """Reads a specific value from config file under the "GCP" namespace.

    Args:
        key: The config field.
        default: Optional default value to assign, if the field is not set.
            A default is assigned only if the field is None; explicitly falsey values are kept.
        required: Whether we require the field to exist.

    Returns:
        The config value (possibly None if required is False).

    Raises:
        SystemExit: If a required config could not be read, i.e. the value is None even after
        applying default (if applicable).
    """
    config_file, configs = load_configs(CONFIG_NAMESPACE, required=True)
    flag_values = _flag_values()
    project = flag_values.get("project", None)
    zone = flag_values.get("zone", None)
    config_name = project_config_key(project, zone)
    project_configs = configs.get(config_name, None)
    if project_configs is None:
        # TODO(markblee): Link to docs once available.
        logging.error(
            "Unknown settings for project=%s and zone=%s; "
            "You may want to configure this project first; Please refer to the docs for details.",
            project,
            zone,
        )
        sys.exit(1)

    value = project_configs.get(key)
    # Only set the default value if the field is omitted. Explicitly falsey values should not be
    # defaulted.
    if value is None:
        value = default
    if required and value is None:
        logging.error("Could not find key %s in settings.", key)
        logging.error(
            "Please check that the file %s has properly set a value for it "
            "under the config section %s.",
            config_file,
            config_name,
        )
        sys.exit(1)
    return value


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
        return repo_or_cwd_config

    # Check if running from package.
    try:
        package_default_config = os.path.join(get_package_root(), default_config)
        if os.path.exists(package_default_config):
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
            config_file = path
            break
    return config_file


# TODO(markblee): Consider moving general config handling to a common util.
def load_configs(
    namespace: Optional[str] = None, required: bool = False
) -> Tuple[Optional[str], Dict]:
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
        A tuple of (config_file, namespace_configs), where config_file can be None if no config file
        could be found and required is False.
    """
    configs = {}
    config_file = None

    def merge(base, overrides):
        """Recursively merge overrides into base."""
        if not isinstance(base, dict):
            return overrides
        for k, v in overrides.items():
            base[k] = merge(base.get(k), v)
        return base

    # If a default config exists, read it.
    default_config_file = _default_config_file()
    if default_config_file is not None:
        merge(configs, toml.load(default_config_file))
        config_file = default_config_file

    # If a user config file exists, use it to override the default.
    user_config_file = _locate_user_config_file()
    if user_config_file:
        merge(configs, toml.load(user_config_file))
        config_file = user_config_file

    if required and config_file is None:
        # TODO(markblee): Link to docs once available.
        logging.error(
            "A config file could not be found; please create one first. "
            "Please refer to the docs for instructions. Please also make sure the config file "
            "can be found in one of: %s",
            _config_search_paths(),
        )
        sys.exit(1)

    if namespace is not None:
        configs = configs.get(namespace, {})
    return config_file, configs


def write_configs_with_header(config_file: str, configs: Dict[str, Any]):
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


def _update_configs(namespace: str, namespace_configs: Dict[str, Any]):
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
    configs[namespace] = namespace_configs  # Update namespace configs.
    write_configs_with_header(config_file, configs)
    print(f"Configs written to {config_file}")


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
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("%s", e)
            return False

    choice = _prompt("Select choice: ")
    while not _valid_choice(choice):
        print(f"Invalid choice: {choice}")
        choice = _prompt("Select choice: ")
    return int(choice)


def _get_projects(
    gcp_configs: Dict[str, Any], labels: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Gets configured project configs, optionally filtering by labels.

    Args:
        gcp_configs: A mapping from projects to project-specific configs. This mapping is keyed by
            `project_config_key(project, zone)` -- note that the project name alone is insufficient
            for uniqueness. Keys prefixed with `_` are considered internal and always filtered out.
        labels: Optional labels to use for filtering.

    Returns:
        A dict of key, values from `gcp_configs` matching the given `labels`.
    """
    projects = {}
    for key, config in gcp_configs.items():
        if key.startswith("_"):
            continue
        if labels is not None:
            project_labels = [label.strip() for label in config["labels"].split(",")]
            if not set(labels).issubset(set(project_labels)):
                continue
        projects[key] = config
    return projects


def _prompt_project(projects: Dict[str, Any]) -> Optional[str]:
    """Prompts the user to select a project among the existing projects.

    Args:
        projects: Mapping from `project_config_key(project, zone)` to project-specific configs.

    Returns:
        The selected key, or None if no projects or the user made an invalid choice.
    """
    choice = _prompt_choice(
        [f"{project} [{config['labels']}]" for project, config in projects.items()]
    )
    keys = list(projects.keys())
    return keys[choice]


def _get_or_prompt_project(gcp_configs: Dict[str, Any]) -> Optional[str]:
    """Gets the project with the provided label(s), or prompt for user to select one if ambiguous.

    See `get_project` and `_prompt_project` for details.
    """
    options = _get_projects(gcp_configs, FLAGS.label)
    if len(options) == 0:
        print(f"No projects matched labels: {FLAGS.label}")
        project = None
    elif len(options) > 1:
        print(f"Multiple projects matched labels: {FLAGS.label}")
        project = _prompt_project(options)
    else:
        project = list(options.keys())[0]
    return project


def project_config_key(project: str, zone: str) -> str:
    """Constructs a toml-friendly name uniquely identified by project, zone."""
    return f"{project}:{zone}"


def _parse_action(argv: Sequence[str]):
    actions = ["list", "activate", "cleanup"]
    action = argv[1] if len(argv) == 2 else None
    if action is None:
        print(f"Usage: {infer_cli_name()} gcp config [{','.join(actions)}]")
        sys.exit(1)
    return action


# pylint: disable-next=too-many-branches
def main(argv: Sequence[str]):
    action = _parse_action(argv)

    # Load existing configs, if any.
    gcp_configs = load_configs(CONFIG_NAMESPACE)[1]
    updates = {}

    if action == "list":
        active_project = gcp_configs.get("_active", None)
        for project, config in _get_projects(gcp_configs, FLAGS.label).items():
            prefix = "[*]" if project == active_project else "[ ]"
            print(f"{prefix} {project} [{config['labels']}]")
        return

    elif action == "activate":
        project = _get_or_prompt_project(gcp_configs)
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

    else:
        logging.warning(
            "Unrecognized action: %s. Please see `%s gcp config --help`.", action, infer_cli_name()
        )
        return

    _update_configs(CONFIG_NAMESPACE, updates)


if __name__ == "__main__":
    _private_flags()
    app.run(main)
