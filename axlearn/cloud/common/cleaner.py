# Copyright © 2023 Apple Inc.

"""Utilities to clean resources."""

from collections.abc import Sequence
from enum import Enum

from axlearn.cloud.common.scheduler import JobScheduler
from axlearn.cloud.common.types import JobSpec
from axlearn.common.config import (
    REQUIRED,
    Configurable,
    Required,
    config_class,
    config_for_function,
)


class Cleaner(Configurable):
    """A basic cleaner interface."""

    # pylint: disable-next=unused-argument,no-self-use
    def sweep(self, jobs: dict[str, JobSpec]) -> Sequence[str]:
        """Removes resources in a non-blocking manner."""
        raise NotImplementedError(type(self))


class AggregationType(Enum):
    """The aggregation rule for CompositeCleaner.

    Attributes:
        UNION: Consider a job cleaned if any cleaner reports the job as cleaned.
        INTERSECTION: Consider a job cleaned only if all cleaners report the job as cleaned.
    """

    UNION = "union"
    INTERSECTION = "intersection"


class CompositeCleaner(Cleaner):
    """A cleaner that runs multiple cleaners on the jobs and returns the union
    or intersection of the results."""

    @config_class
    class Config(Cleaner.Config):
        cleaners: Required[Sequence[Cleaner.Config]] = REQUIRED
        aggregation: AggregationType = AggregationType.UNION

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._cleaners = [cleaner.instantiate() for cleaner in cfg.cleaners]
        self._aggregation = cfg.aggregation

    def sweep(self, jobs: dict[str, JobSpec]) -> Sequence[str]:
        """Apply all cleaners and return the union or intersection of the results."""
        if len(self._cleaners) < 1:
            raise ValueError("There should be at least one cleaner.")
        cleaned_jobs = set(self._cleaners[0].sweep(jobs))
        for cleaner in self._cleaners[1:]:
            if self._aggregation == AggregationType.UNION:
                cleaned_jobs.update(cleaner.sweep(jobs))
            elif self._aggregation == AggregationType.INTERSECTION:
                cleaned_jobs.intersection_update(cleaner.sweep(jobs))
            else:
                raise ValueError(f"Unknown aggregation type: {self._aggregation}")
        return list(cleaned_jobs)


class UnschedulableCleaner(Cleaner):
    """A cleaner that cleans jobs that can never possibly be scheduled."""

    @config_class
    class Config(Cleaner.Config):
        scheduler: Required[JobScheduler.Config] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._quota_fn = cfg.scheduler.quota.instantiate()

    def sweep(self, jobs: dict[str, JobSpec]) -> Sequence[str]:
        """Return unschedulable jobs."""
        cfg = self.config
        # Only call quota function once per sweep.
        quota = self._quota_fn()
        quota_cfg = config_for_function(lambda: lambda: quota)
        scheduler = cfg.scheduler.set(quota=quota_cfg).instantiate()
        result = []
        for job_name, job_spec in jobs.items():
            schedule_result = scheduler.schedule(
                dict(my_job=job_spec.metadata),
            )
            if schedule_result.job_verdicts["my_job"].over_limits:
                result.append(job_name)
        return result
