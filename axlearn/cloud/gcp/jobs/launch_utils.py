# Copyright © 2024 Apple Inc.

"""Helper utilities for launching jobs."""

import re
from typing import Any, Dict, Tuple, Type

from absl import flags

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.types import ResourceType


def serialized_flags_for_job(fv: flags.FlagValues, job: Type[Job]) -> list[str]:
    """Returns a list of serialized flags --flag=value used by the input job.

    Args:
        fv: Flag values to extract from.
        job: Job to extract flags for.

    Returns:
        A sequence of --flag=value strings.
    """
    # Save flags values corresponding to our job.
    launch_fv = flags.FlagValues()
    job.define_flags(launch_fv)

    # Convert the user-supplied flags into a space-separated string, which is forwarded to the
    # command executed by bastion. Only flags which are used by the runner are forwarded.
    filtered = []
    for _, module_flags in fv.flags_by_module_dict().items():
        for flag in module_flags:
            if flag.name in launch_fv and flag.value is not None:
                # Multi-flags get serialized with newlines.
                filtered.extend(flag.serialize().split("\n"))
    return filtered


def jobs_table(jobs: Dict[str, BastionJob]) -> Dict[str, Any]:
    """Construct tabular jobs info.

    Args:
        jobs: A mapping from job name to job info.

    Returns:
        A table which can be passed to `format_table`.
    """
    return dict(
        headings=[
            "NAME",
            "USER_ID",
            "JOB_STATE",
            "PROJECT_ID",
            "RESOURCES",
            "PRIORITY",
        ],
        rows=[
            [
                job.spec.name,
                job.spec.metadata.user_id,
                job.state.name,
                job.spec.metadata.project_id,
                str(job.spec.metadata.resources),
                str(job.spec.metadata.priority),
            ]
            for job in jobs.values()
        ],
    )


def usage_table(usage_info: Dict[str, Dict[ResourceType, Tuple[float, int]]]):
    """Construct tabular usage info.

    Args:
        usage_info: A mapping from principal to resource type to
            (total usage, total number of jobs).

    Returns:
        A table which can be passed to `format_table`.
    """
    table = dict(
        headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
        rows=[
            [
                principal or "unknown",
                resource_type,
                usage[0],
                usage[1],
            ]
            for principal, resource_usage in usage_info.items()
            for resource_type, usage in resource_usage.items()
        ],
    )
    # Sort by usage descending.
    table["rows"] = sorted(table["rows"], key=lambda v: (v[1], v[2]), reverse=True)
    return table


def match_by_regex(match_regex: Dict[str, str]):
    """Matches action and instance type by regex.

    For example:

        match_regex={'start': 'pat1', 'list': 'pat2'}

    ... means that the launcher will be used if action is 'start' and --instance_type regex matches
    'pat1', or if action is 'list' and --instance_type regex matches 'pat2'. The launcher will not
    be invoked for any other action.
    """

    def fn(action: str, instance_type: str) -> bool:
        """Returns True iff the launcher supports the given action and instance_type."""
        return action in match_regex and bool(re.match(match_regex[action], instance_type))

    return fn
