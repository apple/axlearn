# Copyright © 2023 Apple Inc.

"""Utilities to decide whether to schedule jobs according to resource constraints.

The main API is `JobScheduler`, which makes scheduling decisions based on a quota file (see
`quota.py`) and has `Scheduler` and `ProjectJobSorter` as children.

`Scheduler` takes run-or-not verdicts for each job, based on available resources, demands of each
job, per-project quotas, and the priorities of jobs within each project. It uses
`ResourceLimitCalculator` to compute per-project resource limits based on the total limit,
per-project quotas, and demands.

Job priorities with a project can be determined with `ProjectJobSorter`, which sorts the jobs
based on the user id, creation time, and resource demands.
"""

import collections
import dataclasses
import datetime
import queue
from collections import defaultdict
from typing import Dict, Mapping, NamedTuple, Optional, Set

from absl import logging

from axlearn.cloud.common.quota import QuotaFn
from axlearn.cloud.common.types import (
    JobMetadata,
    JobQueue,
    ProjectJobs,
    ProjectResourceMap,
    ResourceMap,
    ResourceType,
)
from axlearn.common.config import REQUIRED, Configurable, InstantiableConfig, Required, config_class


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
            usage: int
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
    def _aggregate_resources(self, resource_map: ResourceMap[int]) -> int:
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
        self, *, limit: int, quotas: Dict[str, float], demands: Dict[str, int]
    ) -> Dict[str, int]:
        """Calculates per-project limits on available resources, quotas, and demands.

        We assume that `limit` and `demands` are all integers, reflecting number of resource units,
        e.g., number of GPUs. The allocations will also be integers.

        Args:
            limit: The total amount of available resources.
            quotas: A mapping from project ids to quotas. If a project id is missing, assume
                quota of 0. Quotas must be non-negative, but do not have to add up to `limit`.
                Available resources will be allocated proportional to quotas.
            demands: A mapping from project ids to demands. If a project id is missing, assume
                demand of 0.

        Returns:
            A mapping from project ids to resource limits for each project id in `demands`.

        Raises:
            ValueError: if any quota is negative.
        """
        for project_id, quota in quotas.items():
            if quota < 0:
                raise ValueError(f"Negative quota for {project_id}: {quota}")

        project_limits: Dict[str, int] = {project_id: 0 for project_id in demands}
        remaining_demands = {
            project_id: demand for project_id, demand in demands.items() if demand > 0
        }
        # Below we take a multi-pass approach to distribute `limit` to `project_limits` according
        # to `quotas` and `demands`. In each pass we compute `active_quotas` according to
        # `remaining_demands` and allocate resources proportional to quotas, approximately
        # `min(demand, limit * (active_quota / active_quota_sum))`, to each project.
        #
        # As John Peebles pointed out, this is also roughly equivalent to allocating resources in
        # one pass in the ascending order of `demand / quota`.
        while limit > 0 and remaining_demands:
            # A project is "active" if it has some remaining demand.
            active_quotas = {
                project_id: quotas.get(project_id, 0)
                for project_id, demand in remaining_demands.items()
                if demand > 0
            }
            active_quota_sum = sum(active_quotas.values())
            if active_quota_sum == 0:
                # When only best-effort quotas remain, allocate limits evenly.
                active_quotas = {project_id: 1 for project_id in remaining_demands}
                active_quota_sum = sum(active_quotas.values())
            logging.vlog(
                1,
                "limit=%s active_quotas=%s remaining_demands=%s",
                limit,
                active_quotas,
                remaining_demands,
            )
            # Sort projects by descending quotas.
            project_id_order = [
                project_id
                for _, project_id in sorted(
                    [(quota, project_id) for project_id, quota in active_quotas.items()],
                    reverse=True,
                )
            ]

            def _allocate(allocation: int, *, project_id: str) -> int:
                project_limits[project_id] += allocation
                remaining_demands[project_id] -= allocation
                if remaining_demands[project_id] <= 0:
                    # Remove from `remaining_demands` if the demand is now fully met.
                    del remaining_demands[project_id]
                return allocation

            new_limit = limit
            # Try to allocate resources in the order of `project_id_order`.
            for project_id in project_id_order:
                if project_id not in remaining_demands:
                    continue
                # The limit we can allocate to `project_id` in this round is proportional to
                # its active quota but no more than `new_limit`. We round the limit, assuming
                # resources can only be allocated by whole units (like GPUs).
                available_limit = min(
                    new_limit, int(round(limit * active_quotas[project_id] / active_quota_sum))
                )
                allocation = min(available_limit, remaining_demands[project_id])
                logging.vlog(
                    2, "Allocating %s (<=%s) to '%s'", allocation, available_limit, project_id
                )
                new_limit -= _allocate(allocation, project_id=project_id)
            if new_limit == limit:
                # Allocate to the first project.
                new_limit -= _allocate(limit, project_id=project_id_order[0])
            limit = new_limit
        return project_limits


@dataclasses.dataclass
class JobVerdict:
    """Describes whether the job should run."""

    def should_run(self):
        return not self.over_limits

    # If the job cannot be scheduled, the set of resource types on which the job's demands exceed
    # the project limits.
    over_limits: Optional[Set[ResourceType]] = None

    def __bool__(self):
        return self.should_run()

    def __or__(self, other: Optional["JobVerdict"]) -> Optional["JobVerdict"]:
        return self if self.should_run() else other


class Scheduler(Configurable):
    """A job scheduler."""

    @config_class
    class Config(Configurable.Config):
        """Configures Scheduler."""

        limit_calculator: ResourceLimitCalculator.Config = ResourceLimitCalculator.default_config()

    @dataclasses.dataclass
    class ScheduleResults:
        # The effective resource limits.
        project_limits: ProjectResourceMap[int]
        # The resource usages.
        project_usages: ProjectResourceMap[int]
        # Mapping: project_id -> (job_id -> run_or_not).
        job_verdicts: Dict[str, Dict[str, JobVerdict]]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self.limit_calculator = cfg.limit_calculator.instantiate()

    def schedule(
        self,
        *,
        resource_limits: ResourceMap[int],
        project_quotas: ProjectResourceMap[float],
        project_jobs: ProjectJobs,
    ) -> ScheduleResults:
        """Makes per-job scheduling decisions based on available resources, quotas, and jobs.

        Args:
            resource_limits: A mapping from resource types to the integer amount of available
                resources.
            project_quotas: A mapping from project ids to quotas. Quotas do not need to add up to
                the limits. Resources will be allocated according to the relative proportions
                between quotas.
            project_jobs: A mapping from project ids to its job queue.

        Returns:
            A mapping from project ids to a mapping of job ids to schedule decisions.
        """
        project_limits: ProjectResourceMap[int] = collections.defaultdict(dict)
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

        project_usages: ProjectResourceMap[int] = {}
        job_verdicts: Dict[str, Dict[str, JobVerdict]] = {}
        for project_id, jobs in project_jobs.items():
            job_verdicts[project_id] = {}
            resource_limits: ResourceMap[int] = project_limits.get(project_id, {})
            resource_usages: ResourceMap[int] = collections.defaultdict(lambda: 0)
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
            project_usages[project_id] = resource_usages

        return Scheduler.ScheduleResults(
            project_limits=project_limits,
            project_usages=project_usages,
            job_verdicts=job_verdicts,
        )


# TODO(markblee): Consider merging with sorter and scheduler.
class JobScheduler(Configurable):
    """Schedules jobs."""

    @config_class
    class Config(Configurable.Config):
        """Configures JobScheduler."""

        # A config that instantiates to a QuotaFn.
        quota: Required[InstantiableConfig[QuotaFn]] = REQUIRED
        # Sorter that decides ordering of jobs-to-schedule.
        sorter: ProjectJobSorter.Config = ProjectJobSorter.default_config()
        # Scheduler that decides whether to resume/suspend jobs.
        scheduler: Scheduler.Config = Scheduler.default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Instantiate children.
        self._quota = cfg.quota.instantiate()
        self._sorter = cfg.sorter.instantiate()
        self._scheduler = cfg.scheduler.instantiate()

    def schedule(
        self,
        jobs: Dict[str, JobMetadata],
        *,
        dry_run: bool = False,
        verbosity: int = 0,
    ) -> Scheduler.ScheduleResults:
        """Schedules jobs according to quotas.

        Scheduling is determined in two passes.

        First, we determine per-project resource limits according to total resources and
        per-project quotas. Then we schedule jobs per project up to its limits.

        Second, (aka "left-over quota scheduling"), if some jobs in `jobs` that are not scheduled
        to run by the first pass, but they would fit into the unused resources, the jobs will be
        selected to run in the returned results. Specifically, the jobs across projects will be
        sorted by `self._sorter` (e.g., according to user id and creation time) and scheduling will
        be attempted in that order. This scheduling pass does *not* take projects or their quotas
        into account.

        Args:
            jobs: A mapping from {job_name: job_metadata}.
            dry_run: Whether to enable dry-run mode, i.e. everything gets scheduled.
                Typically used with higher verbosity to debug scheduling.
            verbosity: Whether to log scheduling report.

        Returns:
            The scheduling results.
        """
        # Group jobs by project.
        project_jobs = defaultdict(dict)
        for job_name, job_metadata in jobs.items():
            project_jobs[job_metadata.project_id][job_name] = job_metadata

        # Sort jobs according to priority.
        for project_id, jobs_to_sort in project_jobs.items():
            project_jobs[project_id] = self._sorter.sort(jobs_to_sort)

        # Fetch quotas each time.
        quota_info = self._quota()
        total_resources = quota_info.total_resources
        project_resources = quota_info.project_resources

        # Decide whether each job should run.
        schedule_results: Scheduler.ScheduleResults = self._scheduler.schedule(
            resource_limits=total_resources,
            project_quotas=project_resources,
            project_jobs=project_jobs,
        )
        # Some resources may remain after scheduling jobs within per-project limits due to per-job
        # demand sizes. Try to schedule jobs on the left-over resources.
        schedule_results = self._schedule_leftover_quotas(
            jobs,
            total_resources=total_resources,
            schedule_results=schedule_results,
        )

        # Log the job verdicts.
        # TODO(markblee): Move to util/reuse this block if we have multiple scheduler
        # implementations.
        if verbosity > 0:
            logging.info("")
            logging.info("==Begin scheduling report")
            logging.info("Total resource limits: %s", total_resources)
            for project_id, project_verdicts in schedule_results.job_verdicts.items():
                logging.info(
                    "Verdicts for Project [%s] Quota [%s] Effective limits [%s]:",
                    project_id,
                    project_resources.get(project_id, {}),
                    schedule_results.project_limits.get(project_id, {}),
                )
                for job_name, job_verdict in project_verdicts.items():
                    logging.info(
                        "Job %s: Resources [%s] Over limits [%s] Should Run? [%s]",
                        job_name,
                        jobs[job_name].resources,
                        job_verdict.over_limits,
                        job_verdict.should_run(),
                    )
            logging.info("==End of scheduling report")
            logging.info("")

        # Construct mock verdicts allowing everything to be scheduled.
        if dry_run:
            project_usages = defaultdict(
                lambda: {resource_type: 0 for resource_type in total_resources}
            )
            for job_metadata in jobs.values():
                for resource_type, usage in job_metadata.resources.items():
                    project_usages[job_metadata.project_id][resource_type] += usage
            schedule_results = Scheduler.ScheduleResults(
                project_limits=schedule_results.project_limits,
                project_usages=project_usages,
                job_verdicts={
                    project_id: {job_name: JobVerdict() for job_name in project_verdicts}
                    for project_id, project_verdicts in schedule_results.job_verdicts.items()
                },
            )
        return schedule_results

    def _schedule_leftover_quotas(
        self,
        jobs: Dict[str, JobMetadata],
        *,
        total_resources: Dict[ResourceType, int],
        schedule_results: Scheduler.ScheduleResults,
    ) -> Scheduler.ScheduleResults:
        """Schedules the left-over quotas.

        See comments on `schedule` on the behavior of left-over scheduling.

        Args:
            jobs: The jobs to be scheduled.
            total_resources: The total amount of resources per type, including resources already
                allocated to jobs in `schedule_results`.
            schedule_results: The current scheduling results.

        Returns:
            Updated scheduling results. Every job scheduled to run in `schedule_results` will
            continue to be scheduled. Zero or more jobs not scheduled to run `scheduled_results`
            may be scheduled to run in the updated results.
        """
        total_usages = defaultdict(int)
        for usages in schedule_results.project_usages.values():
            for resource_type, usage in usages.items():
                total_usages[resource_type] += usage
        leftover_resources = {
            resource_type: total_resources[resource_type] - total_usages[resource_type]
            for resource_type in total_resources
        }
        leftover_jobs = {}
        for verdicts in schedule_results.job_verdicts.values():
            for job_name, verdict in verdicts.items():
                if not verdict.should_run():
                    leftover_jobs[job_name] = jobs[job_name]
        # Sort leftover jobs together, without regard to their projects.
        leftover_job_queue = self._sorter.sort(leftover_jobs)
        leftover_schedule_results: Scheduler.ScheduleResults = self._scheduler.schedule(
            resource_limits=leftover_resources,
            project_quotas={"leftover": {resource_type: 1 for resource_type in total_resources}},
            project_jobs={"leftover": leftover_job_queue},
        )
        merged_verdicts = {}
        merged_usages = {}
        for job_name, job_metadata in jobs.items():
            project_id = job_metadata.project_id
            if project_id not in merged_verdicts:
                merged_verdicts[project_id] = {}
                merged_usages[project_id] = defaultdict(int)
            orig_verdict = schedule_results.job_verdicts[project_id][job_name]
            merged_verdict = orig_verdict | leftover_schedule_results.job_verdicts["leftover"].get(
                job_name
            )
            if merged_verdict:
                for resource_type in total_resources:
                    merged_usages[project_id][resource_type] += job_metadata.resources.get(
                        resource_type, 0
                    )
            merged_verdicts[project_id][job_name] = merged_verdict
        return Scheduler.ScheduleResults(
            project_limits=schedule_results.project_limits,
            project_usages=merged_usages,
            job_verdicts=merged_verdicts,
        )
