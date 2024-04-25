# Copyright © 2023 Apple Inc.

"""Utilities to clean resources."""

from typing import Dict, Sequence

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
    def sweep(self, jobs: Dict[str, JobSpec]) -> Sequence[str]:
        """Removes resources in a non-blocking manner."""
        raise NotImplementedError(type(self))


class CompositeCleaner(Cleaner):
    """A cleaner that runs multiple cleaners on the jobs and returns the union of the results."""

    @config_class
    class Config(Cleaner.Config):
        cleaners: Required[Sequence[Cleaner.Config]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._cleaners = [cleaner.instantiate() for cleaner in cfg.cleaners]

    def sweep(self, jobs: Dict[str, JobSpec]) -> Sequence[str]:
        """Apply all cleaners and return the union of the results."""
        cleaned_jobs = set()
        for cleaner in self._cleaners:
            cleaned_jobs.update(cleaner.sweep(jobs))
        return list(cleaned_jobs)


class UnschedulableCleaner(Cleaner):
    """A cleaner that cleans jobs that can never possibly be scheduled."""

    @config_class
    class Config(Cleaner.Config):
        scheduler: Required[JobScheduler.Config] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._quota_fn = cfg.scheduler.quota.instantiate()

    def sweep(self, jobs: Dict[str, JobSpec]) -> Sequence[str]:
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
            if schedule_result.job_verdicts[job_spec.metadata.project_id]["my_job"].over_limits:
                result.append(job_name)
        return result
