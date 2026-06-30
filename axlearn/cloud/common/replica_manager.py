# Copyright © 2026 Apple Inc.

"""Abstract interface for managing elastic replica state.

A `ReplicaManager` bridges elastic replica state between Bastion and an
external store (Kubernetes CRDs, files, an HTTP API, etc.). Bastion
consults the manager once per scheduling cycle:

  1. refresh()                         # optional prefetch hook
  2. get_desired_replicas(job_name)    # before scheduling, per elastic job
  3. ... scheduler runs ...
  4. set_granted_replicas(job_name, ...) # after scheduling, per elastic verdict

Implementations are responsible for any caching needed to make per-job
calls efficient — typically by listing the backing store in `refresh()`.
"""

from collections.abc import Mapping
from typing import Optional

from axlearn.common.config import Configurable, config_class

# Key under which desired replica counts are stored on `JobStateMetadata`
# by the bastion (for the scheduler to read during expansion).
DESIRED_REPLICAS_KEY = "desired_replicas"

# Key under which granted replica counts are stored on verdict
# `metadata` by the scheduler (for the bastion to forward to the
# replica manager after scheduling).
GRANTED_REPLICAS_KEY = "granted_replicas"


class ReplicaManager(Configurable):
    """Manages desired/granted replica counts for elastic jobs.

    Default method bodies raise NotImplementedError; concrete
    implementations override `get_desired_replicas` and
    `set_granted_replicas`. `refresh` is optional and defaults to no-op.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures a ReplicaManager."""

    def refresh(self) -> None:
        """Optional prefetch hook called once per scheduling cycle.

        Override to fetch state from the backing store in bulk before
        per-job lookups. Default is a no-op.
        """

    def get_desired_replicas(self, job_name: str) -> Optional[Mapping[str, int]]:
        """Returns desired replicas per scaling group for one job.

        Args:
            job_name: Bastion job name.

        Returns:
            A mapping from scaling-group name to desired replica count, or
            None if the manager has no entry for the job (e.g. the job is
            not elastic). When None is returned, the scheduler falls back
            to the job's `min_replicas` from its `ScalingSpec`.
        """
        raise NotImplementedError(type(self))

    def set_granted_replicas(self, job_name: str, granted_replicas: Mapping[str, int]) -> None:
        """Persists granted replicas per scaling group for one job.

        Args:
            job_name: Bastion job name.
            granted_replicas: Mapping from scaling-group name to admitted
                replica count.
        """
        raise NotImplementedError(type(self))
