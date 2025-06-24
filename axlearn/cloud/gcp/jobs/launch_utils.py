# Copyright © 2024 Apple Inc.

"""Helper utilities for launching jobs."""

import collections
import inspect
import json
import re
import shlex
from typing import Optional, Protocol

from absl import flags

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobStatus
from axlearn.cloud.common.types import ResourceType
from axlearn.cloud.common.utils import Table, define_flags
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import GCPAPI, list_k8s_jobsets
from axlearn.common.config import ConfigBase


def infer_module_qualname(cls: type) -> str:
    """Infers the fully-qualified name of `cls.__module__`.

    Using `cls.__module__` naively may return "__main__".
    """
    module = inspect.getmodule(cls)
    if module is None:
        raise ValueError(f"Unable to get module for {cls}.")
    module_spec = module.__spec__  # pytype: disable=attribute-error
    return module_spec.name


def serialized_flags_for_config(cfg: ConfigBase, fv: flags.FlagValues) -> list[str]:
    """Returns a list of serialized flags --flag=value used by the config.

    Args:
        cfg: Config to extract from.
        fv: Flag values to define flags on.

    Returns:
        A sequence of --flag=value strings.
    """
    # Save flags values corresponding to our job.
    launch_fv = flags.FlagValues()
    define_flags(cfg, launch_fv)

    # Convert the user-supplied flags into a space-separated string, which is forwarded to the
    # command executed by bastion. Only flags which are used by the runner are forwarded.
    filtered = []
    for _, module_flags in fv.flags_by_module_dict().items():
        for flag in module_flags:
            if flag.name in launch_fv and flag.value is not None:
                # Multi-flags get serialized with newlines.
                filtered.extend(flag.serialize().split("\n"))
    return filtered


def infer_gcp_api(fv: flags.FlagValues = flags.FLAGS) -> str:
    """Infers `gcp_api` from flags or settings."""

    if getattr(fv, "gcp_api", None) is not None:
        return fv.gcp_api.lower()
    # The return value depends on --zone, so cannot be set as the default value of fv.gcp_api.
    return gcp_settings(
        "launch_gcp_api", default=GCPAPI.GKE.lower(), required=False, fv=fv
    )  # pytype: disable=bad-return-type


class Matcher(Protocol):
    """Matches a launcher using `action` and `flag_values`."""

    def __call__(self, *, action: str, flag_values: flags.FlagValues) -> bool:
        pass


def match_gcp_api(gcp_api: str):
    """Matches against `gcp_api` in a case-insensitive manner."""

    def fn(*, action: str, flag_values: flags.FlagValues) -> bool:
        del action
        requested_gcp_api = infer_gcp_api(flag_values)
        return requested_gcp_api.lower() == gcp_api.lower()

    return fn


def match_by_regex(match_regex: dict[str, str]):
    """Matches flag values by regex.

    Args
        kwargs: A dict mapping flag names to value regex.

    For example:

        match_regex={'instance_type': 'tpu-*', 'job_type': 'default'}

    ... will return True if --instance_type regex matches 'tpu-*' and --job_type matches 'default'.
    If a flag does not exist or is None, the match will return False instead of raising.
    """

    def fn(*, action: str, flag_values: flags.FlagValues) -> bool:
        """Returns True iff the launcher matches the given flag values."""

        for flag, regex in match_regex.items():
            if flag == "action":
                value = action
            else:
                value = getattr(flag_values, flag, "")
            if value is None or not re.fullmatch(regex, value):
                return False
        return True

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


# TODO(ethanli,markblee): Avoid making assumptions about flags being used.
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
