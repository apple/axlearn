# Copyright © 2023 Apple Inc.

"""Type definitions for quota management utilities.

Terminology:
- quota: The amount of resources allocated to a project.
- limit: The maximum amount of resources that can be used by a project. The limit can be higher
  than the quota when there is spare capacity.
"""

from typing import Dict, Sequence, Tuple

ResourceType = str


# Mapping from resource types to the amount of resources.
# Can be used to specify quota/limit/demand/usage per resource type.
ResourceMap = Dict[ResourceType, float]


# Mapping from project ids to resource quota/limit/usage of the project.
ProjectResourceMap = Dict[str, ResourceMap]


# A sequence of (job_id, resource_demand) pairs. The higher priority jobs are listed before the
# lower priority ones.
JobQueue = Sequence[Tuple[str, ResourceMap]]


# A mapping from project ids to its job queue.
ProjectJobs = Dict[str, JobQueue]
