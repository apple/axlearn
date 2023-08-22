# Copyright © 2023 Apple Inc.

"""Manages the GCP config namespace.

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

import sys
from functools import partial
from typing import Any, Dict, Optional

from absl import app, flags, logging

from axlearn.cloud.common import config

FLAGS = flags.FLAGS
CONFIG_NAMESPACE = "gcp"


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
    config_file, configs = config.load_configs(CONFIG_NAMESPACE, required=True)
    flag_values = _flag_values()
    project = flag_values.get("project", None)
    zone = flag_values.get("zone", None)
    if project and zone:
        config_name = _project_config_key(project, zone)
    else:
        # Try to infer from active config.
        config_name = configs.get("_active", None)
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
