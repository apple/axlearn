# Copyright © 2023 Apple Inc.

"""Manages the GCP config namespace.

Possible actions: [list|activate|cleanup|get]

    List:
        - Lists all configured projects, as well as their labels.
        - Specify `--label <label>` to filter among projects. Multiple labels can be specified by
            repeating the flag. The resulting projects will match all labels.

    Activate:
        - Sets a project configuration to be "active". This means that all commands invoked from CLI
            will use the given project, zone, etc. associated with the active config.
        - Specify `--label <label>` to narrow down the list of projects to consider for activation.
            If there are multiple candidates, prompts the user to select one.

    Cleanup:
        - Removes all configuration files. Will prompt for confirmation.

    Get:
        - Prints the specified setting value.

Examples:

    # List available projects.
    axlearn gcp config list

    # List available projects by label.
    axlearn gcp config list --label=my-label

    # Activate a project by prompt.
    axlearn gcp config activate

    # Activate a project by one or more labels.
    axlearn gcp config activate --label=my-label --label=my-other-label

    # Print the setting value for labels.
    axlearn gcp config get labels

"""

import subprocess
import sys
from collections.abc import Sequence
from typing import Any, Optional

from absl import app, flags, logging

from axlearn.cloud.common import config, utils
from axlearn.cloud.gcp.utils import running_from_k8s

FLAGS = flags.FLAGS
CONFIG_NAMESPACE = "gcp"


def _gcp_settings_from_active_config(key: str) -> Optional[str]:
    _, configs = config.load_configs(CONFIG_NAMESPACE, required=False)
    config_name = configs.get("_active", None)
    if not config_name:
        return None
    project_configs = configs.get(config_name, {})
    if not project_configs:
        return None
    return project_configs.get(key, None)


def default_project() -> Optional[str]:
    """Default project from active `gcp_settings`.

    Project is used along with env_id to identify `gcp_settings`.

    Returns: the project in active `gcp_settings` config.
    """

    return _gcp_settings_from_active_config("project")


def default_zone() -> Optional[str]:
    """Default zone from active `gcp_settings`.

    Besides specifying the GCP zone, this value was also used
    along with project to identify `gcp_settings`. It is being replaced by
    env_id. See `default_env_id`.

    Returns: the zone in active `gcp_settings` config.
    """

    return _gcp_settings_from_active_config("zone")


def default_env_id() -> Optional[str]:
    """Default env_id value from active `gcp_settings`.

    Env_id is used along with project to identify `gcp_settings`.

    When env_id is None, fall back to zone for backwards compatibility.

    Returns: the env_id in active `gcp_settings` config; if it doesn't exist, returns the zone.
    """

    return _gcp_settings_from_active_config("env_id") or _gcp_settings_from_active_config("zone")


def gcp_settings(
    key: str,
    *,
    fv: Optional[flags.FlagValues] = FLAGS,
    default: Optional[Any] = None,
    required: bool = True,
) -> Optional[str]:
    """Reads a specific value from config file under the "GCP" namespace.

    Args:
        key: The config field.
        fv: The flag values, which can override project and zone settings. Must be parsed.
        default: Optional default value to assign, if the field is not set.
            A default is assigned only if the field is None; explicitly falsey values are kept.
        required: Whether we require the field to exist.

    Returns:
        The config value (possibly None if required is False).

    Raises:
        RuntimeError: If `fv` have not been parsed and the config field value depends on flags.
        SystemExit: If a required config could not be read, i.e. the value is None even after
            applying default (if applicable).
    """
    if fv is None or not fv.is_parsed():
        raise RuntimeError(f"fv must be parsed before gcp_settings is called for key: {key}")
    flag_values = fv.flag_values_dict()
    project = flag_values.get("project", None)
    if key == "project" and project:
        return project

    # For backwards compatibility, env_id falls back to zone if not specified.
    env_id = flag_values.get("env_id") or flag_values.get("zone", None)
    if key == "env_id" and env_id:
        return env_id

    required = required and default is None
    config_file, configs = config.load_configs(CONFIG_NAMESPACE, required=required)
    if project and env_id:
        config_name = _project_config_key(project, env_id)
    else:
        # Try to infer from active config.
        config_name = configs.get("_active", None)
        logging.info("Inferring active config: %s", config_name)
    project_configs = configs.get(config_name, {})
    if required and not project_configs:
        # TODO(markblee): Link to docs once available.
        logging.error(
            "Unknown settings for project=%s and env_id=%s; "
            "You may want to configure this project first; Please refer to the docs for details.",
            project,
            env_id,
        )
        sys.exit(1)

    # Only set the default value if the field is omitted. Explicitly falsey values should not be
    # defaulted.
    value = project_configs.get(key, default)

    if key == "env_id" and value is None:
        # Fall back to "zone" for backwards compatibility.
        value = project_configs.get("zone")

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


def _project_config_key(project: str, env_id: str) -> str:
    """Constructs a toml-friendly name uniquely identified by project, env_id."""
    return f"{project}:{env_id}"


def main(argv: Sequence[str], *, namespace: str = CONFIG_NAMESPACE, fv: flags.FlagValues = FLAGS):
    """The entrypoint for `gcp config` commands.

    It wraps the common `config.main` implementation by making it k8s aware. In particular, when
    switching configs, it's often necessary to also switch the kube context.
    """
    # Handle all CLI actions as usual.
    config.main(argv, namespace=namespace, fv=fv)

    try:
        # If the user ran `config activate`, we further switch the kube context.
        action = utils.parse_action(argv, options=["activate"])
        assert action == "activate"
        if not running_from_k8s():
            # If the active config has a cluster configured, attempt to obtain credentials and
            # switch kube context. This will allow using kubectl to interact with the right cluster.
            cluster = gcp_settings("gke_cluster", fv=fv, required=False)
            project = gcp_settings("project", fv=fv)
            zone = gcp_settings("zone", fv=fv)
            region = zone.rsplit("-", 1)[0]  # pytype: disable=attribute-error
            if cluster is not None:
                logging.info(
                    "Detected cluster %s for this config. Will attempt to switch cluster contexts.",
                    cluster,
                )
                try:
                    utils.subprocess_run(
                        "gcloud container clusters get-credentials "
                        f"{cluster} --region {region} --project {project}"
                    )
                except subprocess.CalledProcessError:
                    logging.warning("Failed to switch cluster contexts.")
    except app.UsageError:
        # If the user did not run `config activate`, `parse_action` will surface this as
        # `UsageError`. To avoid this, we can specify the universe of options in `parse_action`, but
        # that may become out of sync with `config.main`.
        pass


if __name__ == "__main__":
    config.config_flags()
    app.run(main)
