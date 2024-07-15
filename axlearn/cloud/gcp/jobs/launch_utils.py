# Copyright © 2024 Apple Inc.

"""Helper utilities for launching jobs."""

import collections
import json
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

from absl import flags

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobStatus
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.types import ResourceType
from axlearn.cloud.common.utils import Table
from axlearn.cloud.gcp.tpu import TpuInfo, list_tpu_info, tpu_resource
from axlearn.cloud.gcp.utils import get_credentials, list_k8s_jobsets


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


def match_by_regex(match_regex: Dict[str, str], gcp_api: str):
    """Matches action and instance type by regex.

    For example:

        match_regex={'start': 'pat1', 'list': 'pat2'}

    ... means that the launcher will be used if action is 'start' and --instance_type regex matches
    'pat1', or if action is 'list' and --instance_type regex matches 'pat2'. The launcher will not
    be invoked for any other action.
    """
    match_gcp_api = gcp_api

    def fn(*, action: str, instance_type: str, gcp_api: str) -> bool:
        """Returns True iff the launcher supports the given action and instance_type."""
        return (
            gcp_api.lower() == match_gcp_api.lower()
            and action in match_regex
            and bool(re.match(match_regex[action], instance_type))
        )

    return fn


class JobsToTableFn(Protocol):
    def __call__(self, jobs: Dict[str, BastionJob]) -> Table:
        """Constructs a printable table for the input jobs."""


def jobs_table(jobs: Dict[str, BastionJob]) -> Table:
    """Construct tabular jobs info.

    Args:
        jobs: A mapping from job name to job info.

    Returns:
        A table which can be printed.
    """
    return Table(
        headings=["NAME", "USER_ID", "JOB_STATE", "PROJECT_ID", "RESOURCES", "PRIORITY", "JOB_ID"],
        rows=[
            [
                job.spec.name,
                job.spec.metadata.user_id,
                job.state.status.name,
                job.spec.metadata.project_id,
                str(job.spec.metadata.resources),
                str(job.spec.metadata.priority),
                job.spec.metadata.job_id,
            ]
            for job in jobs.values()
        ],
    )


def _usage_table(usage_info: Dict[str, Dict[ResourceType, Tuple[float, int]]]) -> Table:
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


def user_usage_table(jobs: Dict[str, BastionJob]) -> Table:
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


def project_usage_table(jobs: Dict[str, BastionJob]) -> Table:
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


def with_qrm_tpu_state(fn: JobsToTableFn) -> JobsToTableFn:
    """Amends the table with column(s) pertaining to state of QRM TPUs.

    Jobs for which no TPU state exists will be assigned "PENDING" state.
    """

    def table_fn(jobs: Dict[str, BastionJob]) -> Table:
        table: Table = fn(jobs)
        tpu_state = _qrm_tpu_state_from_jobs(jobs)
        table.add_col("QRM_STATE", tpu_state["job_name_to_states"].values())
        return table

    return table_fn


def _qrm_tpu_state_from_jobs(
    jobs: Dict[str, BastionJob], tpu_infos: Optional[List[TpuInfo]] = None
) -> Dict[str, Any]:
    """Retrieves QRM TPU states for the given jobs."""
    if tpu_infos is None:
        tpu_infos = list_tpu_info(tpu_resource(get_credentials()))

    tpu_infos = {tpu_info.name: tpu_info for tpu_info in tpu_infos}
    tpu_to_job_name = {}
    job_name_to_states = collections.defaultdict(set)

    # Gather TPU states for each job.
    for job in jobs.values():
        tpu_names = [job.spec.name]

        # In the multislice case, tpu_names come from job_name-<slice>.
        # TODO(markblee): Don't rely on parsing flags.
        if matches := re.search(r"--(?:num_slices|num_replicas)[= ](\d+)", job.spec.command):
            num_replicas = int(matches[1])
            if num_replicas > 1:
                tpu_names = [f"{job.spec.name}-{slice_idx}" for slice_idx in range(num_replicas)]

        # Gather unique TPU states for the given job.
        for tpu_name in tpu_names:
            if tpu_name in tpu_infos:
                tpu_to_job_name[tpu_name] = job.spec.name
                tpu_state = tpu_infos[tpu_name].state or "UNKNOWN"
            else:
                tpu_state = "PENDING"
            job_name_to_states[job.spec.name].add(tpu_state)

    return dict(
        running_tpu_infos=tpu_infos,
        running_tpu_to_job_name=tpu_to_job_name,
        job_name_to_states=job_name_to_states,
    )


def with_k8s_jobset_state(fn: JobsToTableFn, *, namespace: str) -> JobsToTableFn:
    """Amends the table with column(s) pertaining to state of GKE Jobsets.

    Jobs for which no Jobset state exists will be assigned "PENDING" state.
    """

    def table_fn(jobs: Dict[str, BastionJob]) -> Table:
        table = fn(jobs)
        states = _k8s_jobset_state_from_jobs(jobs, namespace=namespace)
        table.add_col("GKE_STATE", states)
        return table

    return table_fn


def _k8s_jobset_state_from_jobs(
    jobs: Dict[str, BastionJob], *, namespace: str, k8s_jobsets: Optional[Dict[str, list]] = None
) -> List[str]:
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
