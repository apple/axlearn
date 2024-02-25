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

import sys
from functools import partial
from typing import Any, Optional

from absl import app, flags, logging

from axlearn.cloud.common import config

FLAGS = flags.FLAGS
CONFIG_NAMESPACE = "gcp"


def gcp_settings(
    key: str,
    *,
    default: Optional[Any] = None,
    required: bool = True,
    fv: flags.FlagValues = FLAGS,
) -> Optional[str]:
    """Reads a specific value from config file under the "GCP" namespace.

    Args:
        key: The config field.
        default: Optional default value to assign, if the field is not set.
            A default is assigned only if the field is None; explicitly falsey values are kept.
        required: Whether we require the field to exist.
        fv: The flag values, which can override project and zone settings. Must be parsed.

    Returns:
        The config value (possibly None if required is False).

    Raises:
        RuntimeError: If `fv` have not been parsed and the config field value depends on flags.
        SystemExit: If a required config could not be read, i.e. the value is None even after
            applying default (if applicable).
    """
    if key not in ("project", "zone") and not fv.is_parsed():
        raise RuntimeError(f"fv must be parsed before gcp_settings is called for key: {key}")
    required = required and default is None
    flag_values = fv.flag_values_dict()
    project = flag_values.get("project", None)
    if key == "project" and project:
        return project
    zone = flag_values.get("zone", None)
    if key == "zone" and zone:
        return zone
    config_file, configs = config.load_configs(CONFIG_NAMESPACE, required=required)
    if project and zone:
        config_name = _project_config_key(project, zone)
    else:
        # Try to infer from active config.
        config_name = configs.get("_active", None)
    project_configs = configs.get(config_name, {})
    if required and not project_configs:
        # TODO(markblee): Link to docs once available.
        logging.error(
            "Unknown settings for project=%s and zone=%s; "
            "You may want to configure this project first; Please refer to the docs for details.",
            project,
            zone,
        )
        sys.exit(1)

    # Only set the default value if the field is omitted. Explicitly falsey values should not be
    # defaulted.
    value = project_configs.get(key, default)
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


def _project_config_key(project: str, zone: str) -> str:
    """Constructs a toml-friendly name uniquely identified by project, zone."""
    return f"{project}:{zone}"


if __name__ == "__main__":
    config.config_flags()
    app.run(partial(config.main, namespace=CONFIG_NAMESPACE, fv=FLAGS))
