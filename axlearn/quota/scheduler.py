# Copyright © 2023 Apple Inc.

"""Utilities to decide whether to schedule jobs according to resource constraints.

The main API is `Scheduler`, which takes run-or-not verdicts for each job, based on available
resources, demands of each job, per-project quotas, and the priorities of jobs within each
project.

`Scheduler` calls `ResourceLimitCalculator` to compute per-project resource limits based on the
total limit, per-project quotas, and demands.

Job priorities with a project can be determined with `ProjectJobSorter`, which sorts the jobs
based on the user id, creation time, and resource demands.
"""
import collections
import dataclasses
import datetime
import queue
from typing import Dict, Mapping, NamedTuple, Optional, Set

from axlearn.common.config import Configurable
from axlearn.quota.types import JobQueue, ProjectJobs, ProjectResourceMap, ResourceMap, ResourceType

_EPSILON = 1e-3


@dataclasses.dataclass
class JobMetadata:
    user_id: str
    project_id: str
    creation_time: datetime.datetime
    resources: Dict[ResourceType, float]
    priority: int = 5  # 1 - highest, 5 - lowest


class ProjectJobSorter(Configurable):
    """Sorts jobs within a project."""

    def sort(self, jobs: Mapping[str, JobMetadata]) -> JobQueue:
        """Sorts jobs into a queue.

        Within a project, jobs are sorted first by priority (1 - highest), then aggregate usages
        of the users, and finally creation times:
        (1) Of jobs of the same priority, between jobs of different users, those created by users
            with less resource usage will be prioritized;
        (2) Between jobs of the same user, the older jobs will be prioritized.

        Args:
            jobs: A mapping from job ids to metadata.

        Returns:
            A queue of jobs to be scheduled, with higher priority jobs in front of lower priority
            ones.
        """
        # Mapping: user_id -> List[(priority, creation_time, job_id)].
        user_job_map = collections.defaultdict(list)
        for job_id, job_metadata in jobs.items():
            user_job_map[job_metadata.user_id].append(
                (job_metadata.priority, job_metadata.creation_time, job_id)
            )
        for job_list in user_job_map.values():
            # Sort by (priority, creation_time, job_id).
            job_list.sort()

        class QueueItem(NamedTuple):
            """An item in the priority queue. Each item corresponds to a user."""

            # First sort by job priority.
            priority: int
            # Then sort by the aggregate usage of the user across resource types.
            usage: float
            # Tie-break by creation time of the next job of the user to be sorted.
            creation_time: datetime.datetime
            # The ID of the next job of the user.
            job_id: str
            # The user id.
            user_id: str

        user_queue = queue.PriorityQueue()
        for user_id, job_list in user_job_map.items():
            job_priority, job_time, job_id = job_list[0]
            user_queue.put(
                QueueItem(
                    priority=job_priority,
                    usage=0,
                    creation_time=job_time,
                    job_id=job_id,
                    user_id=user_id,
                )
            )
        job_queue = []
        while not user_queue.empty():
            queue_item: QueueItem = user_queue.get()
            user_jobs = user_job_map[queue_item.user_id]
            job_priority, job_create_time, job_id = user_jobs.pop(0)
            assert queue_item.priority == job_priority
            assert queue_item.creation_time == job_create_time
            assert queue_item.job_id == job_id
            job_metadata: JobMetadata = jobs[job_id]
            job_queue.append((job_id, job_metadata.resources))
            if user_jobs:
                # The user has more jobs. Add it back to `user_queue`.
                next_priority, next_creation_time, next_job_id = user_jobs[0]
                user_queue.put(
                    QueueItem(
                        priority=next_priority,
                        usage=queue_item.usage + self._aggregate_resources(job_metadata.resources),
                        creation_time=next_creation_time,
                        job_id=next_job_id,
                        user_id=queue_item.user_id,
                    )
                )
        return job_queue

    # pylint: disable-next=no-self-use
    def _aggregate_resources(self, resource_map: ResourceMap) -> float:
        """Subclasses can override this method."""
        return sum(resource_map.values())


class ResourceLimitCalculator(Configurable):
    """Calculates per-project resource limits.

    When some projects do not use all their quotas, this implementation allocates the spare
    capacity proportionally among projects whose demands exceed their quotas.

    If there is still spare capacity left after all demands from the projects with non-zero quotas
    are met, the rest capacity will be evenly divided among projects without quota
    (aka "best-effort quotas").
    """

    def calculate(
        self, *, limit: float, quotas: Dict[str, float], demands: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculates per-project limits on available resources, quotas, and demands.

        Args:
            limit: The total amount of available resources.
            quotas: A mapping from project ids to quotas. If a project id is missing, assume
                quota of 0.
            demands: A mapping from project ids to demands. If a project id is missing, assume
                demand of 0.

        Returns:
            A mapping from project ids to resource limits.

        Raises:
            ValueError: if total quota exceeds `limit`.
        """
        total_quota = sum(quotas.values())
        if total_quota > limit + _EPSILON:
            raise ValueError(f"Total quotas ({total_quota}) exceeds limit ({limit})")
        if not quotas or not demands:
            return {}

        demands_within_quota = set(
            project_id
            for project_id, quota in quotas.items()
            if demands.get(project_id, 0) <= quota + _EPSILON
        )
        if not demands_within_quota:
            # No spare capacity from any project. Set limits to project quotas.
            return {project_id: quotas.get(project_id, 0) for project_id in demands}

        # Mapping from project ids to resource limits.
        project_limits = {}

        # There is some spare capacity. Compute the per-project limits as follows:
        # (1) For each project where demand <= quota, simply set project limit to its demand;
        # (2) Re-compute the project limits with the remaining capacity and projects, where:
        #     new_limit = limit - sum(demands of projects within quota)
        #     new_demands = demands of remaining projects
        #     new_quota = quotas of remaining projects scaled to new_limit
        new_limit = limit
        # Mapping from project ids to demands for projects not in `demands_within_quota`.
        new_demands = {}
        # Mapping from project ids to quotas for projects not in `demands_within_quota`.
        remaining_quotas = {}
        for project_id, demand in demands.items():
            if project_id in demands_within_quota:
                project_limits[project_id] = demand
                new_limit -= demand
            else:
                new_demands[project_id] = demand
                remaining_quotas[project_id] = quotas.get(project_id, 0)

        if new_limit > _EPSILON and remaining_quotas:
            remaining_quota_sum = sum(remaining_quotas.values())
            if remaining_quota_sum == 0:
                # This happens when the only projects whose demands exceed quotas are those with
                # zero quotas (aka "best-effort quotas").
                #
                # In this case we divide new_limit evenly among the remaining projects.
                new_quotas = {
                    project_id: new_limit / len(remaining_quotas) for project_id in remaining_quotas
                }
            else:
                # Scale quotas by (new_limit / remaining_quota_sum).
                new_quotas = {
                    project_id: quota * new_limit / remaining_quota_sum
                    for project_id, quota in remaining_quotas.items()
                }
            # Call `self.calculate` again with the remaining projects.
            new_limits = self.calculate(limit=new_limit, quotas=new_quotas, demands=new_demands)
            # Merge the results into `project_limits`.
            project_limits.update(new_limits)
        return project_limits


@dataclasses.dataclass
class JobVerdict:
    """Describes whether the job should run."""

    def should_run(self):
        return not self.over_limits

    # If the job cannot be scheduled, the set of resource types on which the job's demands exceed
    # the project limits.
    over_limits: Optional[Set[ResourceType]] = None


class Scheduler(Configurable):
    """A job scheduler."""

    class Config(Configurable.Config):
        """Configures Scheduler."""

        limit_calculator: ResourceLimitCalculator.Config = ResourceLimitCalculator.default_config()

    @dataclasses.dataclass
    class ScheduleResults:
        # The effective resource limits.
        project_limits: ProjectResourceMap
        # Mapping: project_id -> (job_id -> run_or_not).
        job_verdicts: Dict[str, Dict[str, JobVerdict]]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self.limit_calculator = cfg.limit_calculator.instantiate()

    def schedule(
        self,
        *,
        resource_limits: ResourceMap,
        project_quotas: ProjectResourceMap,
        project_jobs: ProjectJobs,
    ) -> ScheduleResults:
        """Makes per-job scheduling decisions based on available resources, quotas, and jobs.

        Args:
            resource_limits: A mapping from resource types to the amount of available resources.
            project_quotas: A mapping from project ids to quotas.
            project_jobs: A mapping from project ids to its job queue.

        Returns:
            A mapping from project ids to a mapping of job ids to schedule decisions.
        """
        project_limits: ProjectResourceMap = collections.defaultdict(dict)
        for resource_type, limit in resource_limits.items():
            resource_quotas = {
                project_id: quota_map.get(resource_type, 0)
                for project_id, quota_map in project_quotas.items()
            }
            resource_demands = {
                project_id: sum(job_demands.get(resource_type, 0) for _, job_demands in jobs)
                for project_id, jobs in project_jobs.items()
            }
            resource_limits = self.limit_calculator.calculate(
                limit=limit,
                quotas=resource_quotas,
                demands=resource_demands,
            )
            for project_id, project_limit in resource_limits.items():
                project_limits[project_id][resource_type] = project_limit

        job_verdicts = {}
        for project_id, jobs in project_jobs.items():
            job_verdicts[project_id] = {}
            resource_limits: ResourceMap = project_limits.get(project_id, {})
            resource_usages: ResourceMap = collections.defaultdict(lambda: 0)
            for job_id, job_demands in jobs:
                over_limits = set()
                for resource_type, demand in job_demands.items():
                    if resource_usages[resource_type] + demand > resource_limits.get(
                        resource_type, 0
                    ):
                        over_limits.add(resource_type)
                verdict = JobVerdict()
                if over_limits:
                    verdict.over_limits = over_limits
                else:
                    # The job can fit.
                    for resource_type, demand in job_demands.items():
                        resource_usages[resource_type] += demand
                job_verdicts[project_id][job_id] = verdict

        return Scheduler.ScheduleResults(project_limits=project_limits, job_verdicts=job_verdicts)
