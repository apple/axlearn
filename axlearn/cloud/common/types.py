# Copyright © 2023 Apple Inc.

"""Type definitions for quota management utilities.

Terminology:
- quota: The amount of resources allocated to a project.
- limit: The maximum amount of resources that can be used by a project. The limit can be higher
  than the quota when there is spare capacity.
"""

import dataclasses
import datetime
from typing import Dict, Optional, Sequence, Tuple, TypeVar

ResourceType = str


@dataclasses.dataclass
class JobMetadata:
    """Metadata for a bastion job."""

    user_id: str
    project_id: str
    creation_time: datetime.datetime
    resources: Dict[ResourceType, int]
    priority: int = 5  # 1 - highest, 5 - lowest
    # ID of the job, which can be used externally for tracking purposes.
    # It is not used by the bastion directly.
    # TODO(haijing-fu): make it as a required field.
    job_id: Optional[str] = None


@dataclasses.dataclass
class JobSpec:
    """Represents a job that is executed by bastion."""

    # Version to handle schema changes.
    version: int
    # Name of the job (aka job_name).
    name: str
    # Command to run.
    command: str
    # Command to run when job completes (either normally or cancelled).
    cleanup_command: Optional[str]
    # Environment variables. Will be merged into os.environ and applied for both
    # command and cleanup_command.
    env_vars: Optional[Dict[str, str]]
    # Metadata related to a bastion job.
    metadata: JobMetadata


# Mapping from resource types to the amount of resources.
# Can be used to specify quota/limit/demand/usage per resource type.
#
# Use ResourceMap[float] when specifying quotas and ResourceMap[int] when specifying
# limit/demand/usage.
_T = TypeVar("_T", int, float)
ResourceMap = Dict[ResourceType, _T]

# Mapping from project ids to resource quota/limit/usage of the project.
ProjectResourceMap = Dict[str, ResourceMap]

# A sequence of (job_id, job_metadata) pairs. The higher priority jobs are listed before the
# lower priority ones.
JobQueue = Sequence[Tuple[str, JobMetadata]]

# A mapping from project ids to its job queue.
ProjectJobs = Dict[str, JobQueue]
