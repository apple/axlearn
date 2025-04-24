# Copyright © 2025 Apple Inc.

"""A helper module to launch and manage Pathways jobset on GKE."""

from typing import Any

from axlearn.cloud.gcp.job import GKEJob
from axlearn.cloud.gcp.pathways_utils import (
    _PATHWAYS_HEAD_REPLICATED_JOB_NAME,
    PathwaysReplicatedJob,
)
from axlearn.common.utils import Nested


class GKEPathwaysJobSet(GKEJob):
    """A Job that manages Pathways jobset"""

    def __init__(self, cfg: GKEJob.Config, *, bundler):
        super().__init__(cfg, bundler=bundler)
        # TODO(ethanli): Refactor to generalize so we don't need the special case here.
        if not isinstance(cfg.builder, PathwaysReplicatedJob.Config):
            raise NotImplementedError(type(cfg.builder))

    def _build_jobset(self) -> Nested[Any]:
        jobset = super()._build_jobset()

        # TODO (ethanli): Consider refactoring with the modifiers pattern.
        jobset["spec"]["coordinator"] = dict(
            replicatedJob=_PATHWAYS_HEAD_REPLICATED_JOB_NAME,
            jobIndex=0,
            podIndex=0,
        )

        jobset["spec"]["successPolicy"] = dict(
            operator="All",
            targetReplicatedJobs=[
                _PATHWAYS_HEAD_REPLICATED_JOB_NAME,
            ],
        )

        return jobset
