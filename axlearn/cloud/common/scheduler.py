# Copyright © 2023 Apple Inc.

"""Utilities to decide whether to schedule jobs according to resource constraints.

The main API is `JobScheduler`, which makes scheduling decisions based on a quota file (see
`quota.py`) and a corresponding `BaseScheduler` implementation, configured as a child.

See corresponding class docstrings for details.
"""

import collections
import dataclasses
import datetime
import queue
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, Optional, Protocol

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
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    Configurable,
    Required,
    config_class,
    maybe_instantiate,
)


class ProjectJobSorter(Configurable):
    """Sorts jobs within a project based on the user id, creation time, and resource demands."""

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
            job_queue.append((job_id, job_metadata))
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


@dataclasses.dataclass
class JobVerdict:
    """Describes whether the job should run.

    Attributes:
        over_limits: If the job cannot be scheduled, the set of resource types on which the job's
            demands exceed the project limits.
        metadata: Metadata for each verdict. Defaults to an empty dict.
    """

    over_limits: Optional[set[ResourceType]] = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def should_run(self):
        return not self.over_limits

    def __bool__(self):
        return self.should_run()

    def __or__(self, other: Optional["JobVerdict"]) -> Optional["JobVerdict"]:
        return self if self.should_run() else other


class BaseScheduler(Configurable):
    """The job scheduler interface."""

    @dataclasses.dataclass
    class ScheduleResults:
        """Scheduling results.

        Attributes:
            project_limits: The effective resource limits.
            project_usages: The resource usages.
            job_verdicts: A mapping of job_id -> run_or_not.
                The entries will be ordered by descending scheduling priorities (not necessarily
                JobMetadata.priority), where the higher priority jobs will be scheduled before
                lower priority ones. The jobs not getting scheduled will also be ordered.
            unused_limits: Unused resources compared to the limits. Depends on the specific
                implementation of scheduler and resource management, "unused resource" might not
                always be something reasonable or available. Thus, this field is optional.
                Set it to "None" when the scheduler subclass decides not to report unused limits.
        """

        project_limits: ProjectResourceMap[int]
        project_usages: ProjectResourceMap[int]
        job_verdicts: dict[str, JobVerdict]
        unused_limits: Optional[Sequence[ResourceMap[int]]] = None

    def schedule(
        self,
        *,
        resource_limits: Sequence[ResourceMap[int]],
        project_quotas: ProjectResourceMap[float],
        project_jobs: ProjectJobs,
        verbosity: int = 0,
    ) -> ScheduleResults:
        """Makes per-job scheduling decisions based on available resources, quotas, and jobs.

        Args:
            resource_limits: A sequence of mappings from resource types to the integer amount of
                available resources.
            project_quotas: A mapping from project ids to quotas.
            project_jobs: A mapping from project ids to its job queue.
            verbosity: The logging verbosity.

        Returns:
            Scheduling results consisting of:
            * project_limits: The resource limits assigned to each project.
            * project_usages: The total resource usage by project and resource type.
            * job_verdicts: The run-or-not verdicts for each job.
        """
        raise NotImplementedError(type(self))


def _normalize_quotas(
    quotas: ProjectResourceMap, resource_limits: ResourceMap[int]
) -> ProjectResourceMap:
    """Converts the quotas (expressed as proportions) to absolute resource values."""
    quota_sums = collections.defaultdict(float)
    for project_quotas in quotas.values():
        for resource_type, quota in project_quotas.items():
            quota_sums[resource_type] += quota

    normalized_quotas = collections.defaultdict(lambda: collections.defaultdict(float))
    for project_id, project_quotas in quotas.items():
        for resource_type, quota in project_quotas.items():
            normalized_quotas[project_id][resource_type] = (
                quota / max(quota_sums[resource_type], 1e-8)
            ) * resource_limits.get(resource_type, 0)
    return normalized_quotas


def _job_verdict(demands: dict[ResourceType, int], limits: ResourceMap[int]) -> JobVerdict:
    """Constructs a verdict for the job."""
    over_limits = set()
    for resource_type, demand in demands.items():
        if demand > limits.get(resource_type, 0):
            over_limits.add(resource_type)
    verdict = JobVerdict()
    if over_limits:
        verdict.over_limits = over_limits
    return verdict


def _recursively_to_dict(x: Any) -> Any:
    """Recursively converts defaultdicts to dicts."""
    if isinstance(x, collections.defaultdict):
        x = {k: _recursively_to_dict(v) for k, v in x.items()}
    return x


def _compute_total_limits(resource_limits: Sequence[ResourceMap[int]]) -> ResourceMap[int]:
    """Computes total limits from a sequence of limits."""
    total_limits = {}
    for limits in resource_limits:
        for resource_type, limit in limits.items():
            total_limits[resource_type] = total_limits.get(resource_type, 0) + limit
    return total_limits


def _demote_unschedulable_jobs(jobs: JobQueue, *, limits: ResourceMap[int]) -> JobQueue:
    schedulable = []
    unschedulable = []
    for job_name, job_metadata in jobs:
        resources = job_metadata.resources
        is_schedulable = True
        for resource_type, demand in resources.items():
            if demand > limits.get(resource_type, 0):
                is_schedulable = False
                break
        if is_schedulable:
            schedulable.append((job_name, job_metadata))
        else:
            logging.info("Unschedulable job: %s: %s", job_name, job_metadata)
            unschedulable.append((job_name, job_metadata))
    # Put unscheduable jobs after the schedable ones.
    return schedulable + unschedulable


class TierScheduler(BaseScheduler):
    """A scheduler which greedily assigns jobs to tiers.

    Tiers can be used to express different reservation types, such as:
    1. Reserved quota: instances scheduled onto this tier are SLO backed.
    2. Reserved pre-emptible quota: instances scheduled onto this tier may be pre-empted, e.g. by
        defragmentation, machine repairs, or other events, but still belong to a reservation.
    3. On-demand quota: instances scheduled onto this tier are provisioned as capacity allows,
        possibly outside of a reservation.

    We make some basic assumptions:
    1. Jobs can tolerate scheduling on any given tier.
    2. Tiers have an implicit priority order.
    3. Tiers should partition the full quota, i.e., they must be disjoint and add up to the full
        resources.
    4. Lower priority tiers can borrow from higher priority tiers.

    With these assumptions, scheduling proceeds as follows:
    1. Projects are ordered based on the ratio `cumulative project usage / project quota`, breaking
        ties by creation time of the job to-be-scheduled for each project (first come first serve).
        If there are multiple resource types, we take the max usage ratio across resource types.
    2. Within each project, jobs are assumed to be already ordered.
    3. A job-to-be-scheduled will greedily acquire resources starting from the highest priority tier
        to the lowest priority tier. Once its demands are met, it will be scheduled on the last
        (lowest priority) tier which contributed quota.
    """

    def schedule(
        self,
        *,
        resource_limits: Sequence[ResourceMap[int]],
        project_quotas: ProjectResourceMap,
        project_jobs: ProjectJobs,
        verbosity: int = 0,
    ) -> BaseScheduler.ScheduleResults:
        """See `BaseScheduler.schedule` for details."""
        if not isinstance(resource_limits, Sequence):
            raise ValueError(f"Expected resource tiers to be a sequence, got {resource_limits=}.")
        if len(resource_limits) < 1:
            raise ValueError(f"Expected at least one tier, got {resource_limits=}.")

        project_queue = queue.PriorityQueue()
        # Avoid modifying in-place.
        resource_limits = [{**limits} for limits in resource_limits]
        # Maps resource_type -> total limits.
        remaining_limits = _compute_total_limits(resource_limits)
        # Maps project_id -> resource_type -> quota.
        project_quotas = _normalize_quotas(project_quotas, remaining_limits)
        # Maps project_id -> resource_type -> usage.
        project_usages = collections.defaultdict(lambda: collections.defaultdict(int))
        # Maps project_id -> deque of (job_id, job_metadata).
        project_jobs: dict[str, collections.deque[tuple[str, JobMetadata]]] = {
            project_id: collections.deque(
                _demote_unschedulable_jobs(sorted_jobs, limits=remaining_limits)
            )
            for project_id, sorted_jobs in project_jobs.items()
        }

        def project_queue_item(project_id: str) -> tuple[float, datetime.datetime, str]:
            """Constructs a queue entry for the given project."""
            assert len(project_jobs[project_id]) > 0
            usages = project_usages[project_id]
            _, next_job = project_jobs[project_id][0]
            usage_ratios = [
                (usages[resource_type] + usage)
                / max(project_quotas[project_id][resource_type], 1e-8)
                for resource_type, usage in next_job.resources.items()
            ]
            # Smaller values have higher priority.
            return max((0, *usage_ratios)), next_job.creation_time, project_id

        # Initialize queue with projects.
        for project_id, sorted_jobs in project_jobs.items():
            if sorted_jobs:
                project_queue.put(project_queue_item(project_id))

        def traverse_tiers(
            tier_limits: dict[int, ResourceMap], demands: ResourceMap
        ) -> dict[int, ResourceMap]:
            """Visits tiers in the order specified in tier_limits, until we can satisfy demands."""
            tier_usages = collections.defaultdict(lambda: collections.defaultdict(int))
            demands = {**demands}
            for tier, limits in tier_limits.items():
                for resource_type in list(demands.keys()):
                    if tier_usage := min(demands[resource_type], limits.get(resource_type, 0)):
                        tier_usages[tier][resource_type] += tier_usage
                        demands[resource_type] -= tier_usage
                        if demands[resource_type] <= 0:
                            del demands[resource_type]
                if not demands:
                    break
            return tier_usages

        job_verdicts = {}
        while not project_queue.empty():
            project_usage_ratio, _, project_id = project_queue.get()
            job_id, job_metadata = project_jobs[project_id].popleft()

            # Admit the highest priority job within the project.
            verdict = _job_verdict(job_metadata.resources, remaining_limits)
            if verdict:
                # In the forward pass, we greedily identify the minimum set of tiers that are
                # required to satisfy the job's demands.
                tier_usages = traverse_tiers(
                    dict(enumerate(resource_limits)), job_metadata.resources
                )
                final_tier = max((0, *tier_usages.keys()))
                # In the backward pass, we greedily acquire resources from the lowest-priority tier
                # first, since the job will ultimately be scheduled on the lowest-priority tier.
                tier_usages = traverse_tiers(
                    dict(reversed(list(enumerate(resource_limits[: final_tier + 1])))),
                    job_metadata.resources,
                )
                verdict = JobVerdict(metadata={"tier": final_tier})

                # Update resource_limits, remaining_limits and project_usages.
                for tier, usages in tier_usages.items():
                    for resource_type, usage in usages.items():
                        resource_limits[tier][resource_type] -= usage
                        remaining_limits[resource_type] -= usage
                        if remaining_limits[resource_type] <= 0:
                            del remaining_limits[resource_type]
                        project_usages[project_id][resource_type] += usage

            if verbosity > 0:
                logging.info(
                    "Schedule %s(%s)/%s: %s", project_id, project_usage_ratio, job_id, verdict
                )

            job_verdicts[job_id] = verdict
            if project_jobs[project_id]:
                project_queue.put(project_queue_item(project_id))

        return BaseScheduler.ScheduleResults(
            # Treat the usages as the limits.
            project_limits=_recursively_to_dict(project_usages),
            project_usages=_recursively_to_dict(project_usages),
            job_verdicts=_recursively_to_dict(job_verdicts),
            unused_limits=_recursively_to_dict(resource_limits),
        )


class ReporterFn(Protocol):
    def __call__(
        self,
        *,
        schedule_results: BaseScheduler.ScheduleResults,
        resource_limits: Sequence[ResourceMap],
        project_quotas: ProjectResourceMap,
        project_jobs: ProjectJobs,
        verbosity: int,
    ):
        """A callable that handles schedule inputs and results."""


def logging_reporter(
    *,
    schedule_results: BaseScheduler.ScheduleResults,
    resource_limits: Sequence[ResourceMap],
    project_quotas: ProjectResourceMap,
    project_jobs: ProjectJobs,
    verbosity: int,
):
    """An implementation of ReporterFn which logs schedule verdicts by associating them
    with schedule inputs."""
    if verbosity < 1:
        return

    # Log the job verdicts.
    logging.info("")
    logging.info("==Begin scheduling report")
    logging.info("Total resource limits: %s", resource_limits)
    for project_id, project_job_queue in project_jobs.items():
        logging.info(
            "Verdicts for Project [%s] Quota [%s] Effective limits [%s]:",
            project_id,
            project_quotas.get(project_id, {}),
            schedule_results.project_limits.get(project_id, {}),
        )
        for job_name, job_metadata in project_job_queue:
            job_verdict = schedule_results.job_verdicts[job_name]
            logging.info(
                "Job %s: Resources [%s] Over limits [%s] Should Run? [%s] Metadata [%s]",
                job_name,
                job_metadata.resources,
                job_verdict.over_limits,
                job_verdict.should_run(),
                job_verdict.metadata,
            )
    logging.info("==End of scheduling report")
    logging.info("")


def composite_reporter(reporters: Sequence[ConfigOr[ReporterFn]]) -> ReporterFn:
    """Composite a list of inner ReporterFn into one ReporterFn. Execute inner ones in parallel."""
    if not reporters:
        raise ValueError(f"Got empty {reporters=} to composite.")

    reporters = [maybe_instantiate(reporter) for reporter in reporters]
    if len(reporters) == 1:
        return reporters[0]

    def report_fn(**kwargs):
        # Parallel reporter execution.
        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda func: func(**kwargs), reporters))

    return report_fn


class ReportingScheduler(BaseScheduler):
    """A scheduler that wraps an inner scheduler (e.g. TierScheduler).

    It delegates the scheduling responsibility to the inner scheduler, and handles reporting of
    shedule inputs and schedule results.
    """

    @config_class
    class Config(BaseScheduler.Config):
        # Inner schedule to which the scheduling responsibility is delegated.
        inner: BaseScheduler.Config = TierScheduler.default_config()
        # Reporter function to handle customized reporting.
        reporter: Required[ConfigOr[ReporterFn]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._inner: BaseScheduler = cfg.inner.instantiate()
        self._reporter: ReporterFn = maybe_instantiate(cfg.reporter)

    def schedule(
        self,
        **kwargs,
    ) -> BaseScheduler.ScheduleResults:
        """Delegate the scheduling logic to inner scheduler and handles the reporting logic.

        See `BaseScheduler.schedule` for details of the interface.
        "kwargs" are used for more explicit and convenient delegation.
        """
        schedule_results = self._inner.schedule(**kwargs)

        # Handle reports.
        self._reporter(schedule_results=schedule_results, **kwargs)
        return schedule_results


class JobScheduler(Configurable):
    """Schedules jobs, possibly onto multiple tiers of quotas."""

    @config_class
    class Config(Configurable.Config):
        """Configures JobScheduler.

        Attributes:
            quota: A config that instantiates to a QuotaFn.
            sorter: Sorter that decides ordering of jobs-to-schedule.
            scheduler: Scheduler that decides whether to resume/suspend jobs.
        """

        quota: Required[ConfigOr[QuotaFn]] = REQUIRED
        sorter: ProjectJobSorter.Config = ProjectJobSorter.default_config()
        scheduler: BaseScheduler.Config = TierScheduler.default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._quota = maybe_instantiate(cfg.quota)
        self._sorter: ProjectJobSorter = cfg.sorter.instantiate()
        self._scheduler: BaseScheduler = cfg.scheduler.instantiate()

    def schedule(
        self,
        jobs: dict[str, JobMetadata],
        *,
        dry_run: bool = False,
        verbosity: int = 0,
    ) -> BaseScheduler.ScheduleResults:
        """Schedules jobs according to quotas.

        The scheduling behavior depends on the configured `cfg.scheduler`.

        Args:
            jobs: A mapping from {job_name: job_metadata}.
            dry_run: Whether to enable dry-run mode, i.e. everything gets scheduled.
                Typically used with higher verbosity to debug scheduling.
            verbosity: Whether to log scheduling report.

        Returns:
            The scheduling results.
        """
        # Group jobs by project.
        project_jobs = collections.defaultdict(dict)
        for job_name, job_metadata in jobs.items():
            project_jobs[job_metadata.project_id][job_name] = job_metadata

        # Sort jobs according to priority.
        for project_id, jobs_to_sort in project_jobs.items():
            project_jobs[project_id] = self._sorter.sort(jobs_to_sort)

        # Fetch quotas each time.
        quota_info = self._quota()
        resource_limits = quota_info.total_resources
        project_quotas = quota_info.project_resources

        schedule_results = self._scheduler.schedule(
            resource_limits=resource_limits,
            project_quotas=project_quotas,
            project_jobs=project_jobs,
            verbosity=verbosity,
        )

        # Construct mock verdicts allowing everything to be scheduled.
        if dry_run:
            project_usages = collections.defaultdict(lambda: collections.defaultdict(int))
            for job_metadata in jobs.values():
                for resource_type, usage in job_metadata.resources.items():
                    project_usages[job_metadata.project_id][resource_type] += usage
            schedule_results = BaseScheduler.ScheduleResults(
                project_limits=schedule_results.project_limits,
                project_usages=project_usages,
                job_verdicts={job_name: JobVerdict() for job_name in schedule_results.job_verdicts},
            )
        return schedule_results
