# Copyright © 2023 Apple Inc.

"""Utilities to retrieve quotas."""

import re
from collections import defaultdict
from typing import Dict, List

import tensorflow as tf
import toml

QUOTA_CONFIG_PATH = "project-quotas/project-quotas.config"


def get_resource_limits(path: str) -> Dict[str, Dict[str, float]]:
    """Attempts to read resource limits, both total and per-project.

    Args:
        path: Absolute path to the quota config file.

    Returns:
        A dict with the following keys:
        - resource_limits: A mapping from resource type to total resource limits.
        - project_resources: A nested mapping. Key is project identifier, value is mapping from
            resource type to allocated amount of resources (as percentages).

    Raises:
        ValueError: If unable to parse quota config file.
    """
    with tf.io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            total_resources = cfg["total_resources"]
            project_resources = cfg["project_resources"]

            # Project resources are typically expressed as percentages.
            # Here we convert them to actual values.
            total_project_resources = defaultdict(float)
            for resources in project_resources.values():
                for resource_type, fraction in resources.items():
                    value = fraction * total_resources[resource_type]
                    total_project_resources[resource_type] += value
                    resources[resource_type] = value

            for resource_type, total in total_project_resources.items():
                if total > total_resources[resource_type]:
                    raise ValueError(
                        f"Sum of {resource_type} project resources ({total}) "
                        f"exceeds total ({total_resources[resource_type]})"
                    )

            return dict(
                total_resources=total_resources,
                project_resources=project_resources,
            )
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")


def get_user_projects(path: str, user_id: str) -> List[str]:
    """Attempts to read project membership for the given user.

    Args:
        path: Absolute path to the quota config file.
        user_id: User's unique identifier.

    Returns:
        A list of project IDs that the given user belongs to.

    Raises:
        ValueError: If unable to parse quota config file.
    """
    with tf.io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            user_in_projects = []
            for project_id, project_members in cfg["project_membership"].items():
                for member in project_members:
                    if re.fullmatch(member, user_id):
                        user_in_projects.append(project_id.lower())
            return user_in_projects
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")
