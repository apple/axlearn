# Copyright © 2023 Apple Inc.

"""Utilities to retrieve quotas."""

from typing import Dict, List

import tensorflow as tf
import toml

QUOTA_CONFIG_PATH = "project-quotas/project-quotas.config"


def get_project_resources(path: str) -> Dict[str, Dict[str, float]]:
    """Attempts to read project resource quotas.

    Args:
        path: Absolute path to the quota config file.

    Returns:
        Nested project quota map. Key is project identifier, value is mapping
        from resource type to allocated amount of resources.

    Raises:
        ValueError: If unable to parse project resource quota file.
    """
    with tf.io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            return cfg["project_resources"]
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")


def get_user_projects(path: str, user_id: str) -> List[str]:
    """Attempts to read project membership for the given user.

    Args:
        path: Absolute path to the quota config file.
        user_id: User's unique identifier.

    Returns:
        A list of project IDs that the given user belongs to.

    Raises:
        ValueError: If unable to parse project resource quota file.
    """
    with tf.io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            user_in_projects = []
            for project_id, project_members in cfg["project_membership"].items():
                if user_id in project_members:
                    user_in_projects.append(project_id.lower())
            return user_in_projects
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")
