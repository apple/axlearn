# Copyright © 2025 Apple Inc.

"""Pod mutator interface for customizing Kubernetes pod specs."""

from typing import Any, Optional

from axlearn.cloud.common.types import JobSpec
from axlearn.common.config import Configurable
from axlearn.common.utils import Nested


class PodMutator(Configurable):
    """A plugin interface for mutating a pod spec after it is built.

    Subclasses can inject init containers, sidecar containers, volumes,
    environment variables, or any other pod-level customization.
    """

    def mutate(self, job_spec: Optional[JobSpec], pod: Nested[Any]) -> Nested[Any]:
        """Mutates the pod spec dict in place and returns it.

        Args:
            job_spec: The deserialized JobSpec from the bastion environment, or None if
                not running under bastion (e.g. local launch or tests).
            pod: A nested dict with keys "metadata" and "spec", corresponding
                to a Kubernetes Pod template.

        Returns:
            The (possibly modified) pod dict.
        """
        raise NotImplementedError(type(self))
