# Copyright © 2023 Apple Inc.

"""Utilities to retrieve quotas."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Protocol

import toml
from tensorflow import io as tf_io

from axlearn.cloud.common.types import ProjectResourceMap, ResourceMap

QUOTA_CONFIG_PATH = "project-quotas/project-quotas.config"


@dataclass
class QuotaInfo:
    """Quota information for job scheduling."""

    # A mapping from resource type to total resource limits.
    total_resources: ResourceMap
    # A nested mapping. Key is project identifier, value is mapping from resource type to allocated
    # amount of resources.
    project_resources: ProjectResourceMap


class QuotaFn(Protocol):
    def __call__(self) -> QuotaInfo:
        """A callable that returns quota information for scheduling."""


def get_resource_limits(path: str) -> QuotaInfo:
    """Attempts to read resource limits, both total and per-project.

    Args:
        path: Absolute path to the quota config file.

    Returns:
        QuotaInfo for scheduling.

    Raises:
        ValueError: If unable to parse quota config file.
    """
    with tf_io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            return _convert_and_validate_resources(
                QuotaInfo(
                    total_resources=cfg["total_resources"],
                    project_resources=cfg["project_resources"],
                )
            )
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")


def _convert_and_validate_resources(info: QuotaInfo) -> QuotaInfo:
    # Project resources are typically expressed as percentages.
    # Here we convert them to actual values. Conversion happens in-place.
    total_project_resources = defaultdict(float)
    for resources in info.project_resources.values():
        for resource_type, fraction in resources.items():
            value = fraction * float(info.total_resources.get(resource_type, 0))
            total_project_resources[resource_type] += value
            resources[resource_type] = value

    for resource_type, total in total_project_resources.items():
        if total > info.total_resources[resource_type]:
            raise ValueError(
                f"Sum of {resource_type} project resources ({total}) "
                f"exceeds total ({info.total_resources[resource_type]})"
            )
    return info


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
    with tf_io.gfile.GFile(path, mode="r") as f:
        cfg = toml.loads(f.read())
        if cfg["toml-schema"]["version"] == "1":
            user_in_projects = []
            for project_id, project_members in cfg["project_membership"].items():
                for member in project_members:
                    if re.fullmatch(member, user_id):
                        user_in_projects.append(project_id.lower())
            return user_in_projects
        raise ValueError(f"Unsupported schema version {cfg['toml-schema']['version']}")
