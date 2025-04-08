# Copyright © 2024 Apple Inc.

"""Helper utilities for launching jobs."""

import collections
import json
import re
import shlex
from typing import Optional, Protocol

from absl import flags

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobStatus
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.types import ResourceType
from axlearn.cloud.common.utils import Table
from axlearn.cloud.gcp.utils import list_k8s_jobsets


def serialized_flags_for_job(fv: flags.FlagValues, job: type[Job]) -> list[str]:
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


def match_by_regex(*, match_regex: dict[str, str], gcp_api: str, job_type: str):
    """Matches action and instance type by regex.

    Args
        match_regex: Dictionary of regex to match against action and instance type.
        gcp_api: GCP API client.
        job_type: Job type to match, note that it has higher priority
          than other matching conditions.

    For example:

        match_regex={'start': 'pat1', 'list': 'pat2'}

    ... means that the launcher will be used if action is 'start' and --instance_type regex matches
    'pat1', or if action is 'list' and --instance_type regex matches 'pat2'. The launcher will not
    be invoked for any other action.
    """
    match_gcp_api = gcp_api
    match_job_type = job_type

    def fn(*, action: str, instance_type: str, gcp_api: str, job_type: str) -> bool:
        """Returns True iff the launcher supports the given action and instance_type."""

        # job_type has a higher priority then other condition since it will decide which
        # runner in runner.inner to be used.
        if match_job_type != "default":
            return match_job_type.lower() == job_type.lower()

        return (
            gcp_api.lower() == match_gcp_api.lower()
            and action in match_regex
            and bool(re.match(match_regex[action], instance_type))
        )

    return fn


class JobsToTableFn(Protocol):
    def __call__(self, jobs: dict[str, BastionJob]) -> Table:
        """Constructs a printable table for the input jobs."""


def jobs_table(jobs: dict[str, BastionJob]) -> Table:
    """Construct tabular jobs info.

    Args:
        jobs: A mapping from job name to job info.

    Returns:
        A table which can be printed.
    """
    return Table(
        headings=[
            "NAME",
            "USER_ID",
            "JOB_STATE",
            "PROJECT_ID",
            "RESOURCES",
            "PRIORITY",
            "JOB_ID",
            "TIER",
        ],
        rows=[
            [
                job.spec.name,
                job.spec.metadata.user_id,
                job.state.status.name,
                job.spec.metadata.project_id,
                str(job.spec.metadata.resources),
                str(job.spec.metadata.priority),
                job.spec.metadata.job_id,
                str(job.state.metadata.get("tier", "None")),
            ]
            for job in jobs.values()
        ],
    )


def _usage_table(usage_info: dict[str, dict[ResourceType, tuple[float, int]]]) -> Table:
    """Construct tabular usage info.

    Args:
        usage_info: A mapping from principal to resource type to
            (total usage, total number of jobs).

    Returns:
        A table which can be printed.
    """
    table = Table(
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
    table.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return table


def user_usage_table(jobs: dict[str, BastionJob]) -> Table:
    """Computes per-user usage for the given bastion jobs."""
    # Maps user_id -> resource_type -> (total_usage, count).
    usage_by_user = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for job in jobs.values():
        if job.state.status != JobStatus.PENDING:
            user_id = job.spec.metadata.user_id
            resources = job.spec.metadata.resources
            for resource_type, usage in resources.items():
                usage_by_user[user_id][resource_type][0] += usage
                usage_by_user[user_id][resource_type][1] += 1
    return _usage_table(usage_by_user)


def project_usage_table(jobs: dict[str, BastionJob]) -> Table:
    """Computes per-user and per-project usage for the given bastion jobs."""
    # Maps project_id -> resource_type -> (total_usage, count).
    usage_by_project = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for job in jobs.values():
        if job.state.status != JobStatus.PENDING:
            project_id = job.spec.metadata.project_id
            resources = job.spec.metadata.resources
            for resource_type, usage in resources.items():
                usage_by_project[project_id][resource_type][0] += usage
                usage_by_project[project_id][resource_type][1] += 1
    return _usage_table(usage_by_project)


def with_k8s_jobset_state(fn: JobsToTableFn, *, namespace: str) -> JobsToTableFn:
    """Amends the table with column(s) pertaining to state of GKE Jobsets.

    Jobs for which no Jobset state exists will be assigned "PENDING" state.
    """

    def table_fn(jobs: dict[str, BastionJob]) -> Table:
        table = fn(jobs)
        states = _k8s_jobset_state_from_jobs(jobs, namespace=namespace)
        table.add_col("GKE_STATE", states)
        return table

    return table_fn


def _k8s_jobset_state_from_jobs(
    jobs: dict[str, BastionJob], *, namespace: str, k8s_jobsets: Optional[dict[str, list]] = None
) -> list[str]:
    """Retrieves k8s jobset states for the given jobs."""
    if k8s_jobsets is None:
        k8s_jobsets = list_k8s_jobsets(namespace=namespace)

    states = []
    for job in jobs.values():
        # Gather unique states for the given job.
        statuses = ["active", "ready", "failed", "succeeded"]
        if k8s_jobs := k8s_jobsets.get(job.spec.name, []):
            job_states = collections.defaultdict(int)
            for k8s_job in k8s_jobs:
                for status in statuses:
                    k8s_status = getattr(k8s_job, "status", None)
                    job_states[status] += getattr(k8s_status, status, None) or 0
            states.append(json.dumps(dict(job_states)))
        else:
            states.append("PENDING")
    return states


def _parse_resource_flags_from_command(command: str) -> flags.FlagValues:
    """Infer resources flags from launch command.

    It parses the resources flags from the command.

    Args:
        command: The launch command of a job.

    Returns:
        A flags.FlagValues containing the parsed resources flags.
    """
    commands = shlex.split(command)

    fv = flags.FlagValues()
    flags.DEFINE_string("instance_type", default=None, help="", flag_values=fv)
    flags.DEFINE_integer("num_replicas", default=None, help="", flag_values=fv)
    flags.DEFINE_boolean("enable_pre_provisioner", default=None, help="", flag_values=fv)
    flags.DEFINE_alias("num_slices", "num_replicas", flag_values=fv)
    flags.DEFINE_alias("tpu_type", "instance_type", flag_values=fv)
    fv(commands, known_only=True)

    return fv


def validate_resource_flags(original_command: str, updated_command: str):
    """Raise an exception if the resource flags are different
    in the original and updated commands."""

    original_parsed_flags = _parse_resource_flags_from_command(original_command)
    updated_parsed_flags = _parse_resource_flags_from_command(updated_command)

    original_instance_type = original_parsed_flags.instance_type or original_parsed_flags.tpu_type
    updated_instance_type = updated_parsed_flags.instance_type or updated_parsed_flags.tpu_type

    original_num_replicas = original_parsed_flags.num_replicas or original_parsed_flags.num_slices
    updated_num_replicas = updated_parsed_flags.num_replicas or updated_parsed_flags.num_slices

    original_pre_provisioner = original_parsed_flags.enable_pre_provisioner
    updated_pre_provisioner = updated_parsed_flags.enable_pre_provisioner

    if original_instance_type != updated_instance_type:
        raise ValueError(f"Expected {original_instance_type=} to match {updated_instance_type=}.")

    if original_num_replicas != updated_num_replicas:
        raise ValueError(f"Expected {original_num_replicas=} to match {updated_num_replicas=}.")

    if original_pre_provisioner != updated_pre_provisioner:
        raise ValueError(
            f"Expected {original_pre_provisioner=} to match {updated_pre_provisioner=}."
        )
