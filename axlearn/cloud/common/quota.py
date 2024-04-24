# Copyright © 2023 Apple Inc.

"""Utilities to retrieve quotas."""

import re
from dataclasses import dataclass
from typing import List, Protocol, Sequence

import toml
from tensorflow import io as tf_io

from axlearn.cloud.common.types import ProjectResourceMap, ResourceMap

QUOTA_CONFIG_PATH = "project-quotas/project-quotas.config"


@dataclass
class QuotaInfo:
    """Quota information for job scheduling."""

    # A sequence of mappings from resource type to total resource limits.
    # Each element in the sequence represents a scheduling tier, where a higher priority/SLO tier is
    # listed before a lower priority/SLO tier.
    # A job can be scheduled with resources across multiple tiers with an SLO corresponding to the
    # lowest-SLO tier used for the job.
    total_resources: Sequence[ResourceMap[float]]
    # A nested mapping. Key is project identifier, value is mapping from resource type to
    # per-project resource proportions.
    project_resources: ProjectResourceMap[float]


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
            total_resources = cfg["total_resources"]
            if not isinstance(total_resources, Sequence):
                total_resources = [total_resources]
            return QuotaInfo(
                total_resources=total_resources,
                project_resources=cfg["project_resources"],
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
