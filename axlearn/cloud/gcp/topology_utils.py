# Copyright © 2025 Apple Inc.

"""Utilities for TPU topology assignment and selection."""

import json
import logging
import os
from typing import Optional

from axlearn.cloud.common.bastion import BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR


def get_topology_from_env() -> Optional[list[list[str]]]:
    """Retrieves TPU topology assignments from the environment variable.

    Returns:
        A list of lists of strings representing topology assignments, where each inner list
        contains slice identifiers for a particular job replica. Returns None if the
        environment variable is not set or if parsing fails.

    Example:
        [["sub-block-id-1", "sub-block-id-2"], ["sub-block-id-3"]]
    """
    topology_assignments_env = os.environ.get(BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR)
    if not topology_assignments_env:
        logging.debug("No %s environment variable set.", BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR)
        return None

    try:
        topology = json.loads(topology_assignments_env)
        logging.info("Retrieved topology_assignment from env var: %s", topology)
        return topology
    except json.JSONDecodeError as e:
        logging.warning(
            "Failed to parse topology assignments from env var %s, value: %s, error: %s",
            BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR,
            topology_assignments_env,
            e,
        )
        return None
