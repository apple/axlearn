# Copyright © 2025 Apple Inc.

"""A collection of built-in bastion runners."""

from typing import Optional, Union

from axlearn.cloud.gcp.job import GKEJob, exclusive_topology_annotations
from axlearn.cloud.gcp.job_flink import FlinkTPUGKEJob
from axlearn.cloud.gcp.jobset_utils import A3ReplicatedJob, TPUReplicatedJob
from axlearn.cloud.gcp.node_pool_provisioner import TPUNodePoolProvisioner
from axlearn.cloud.gcp.runners.base import BaseRunnerJob
from axlearn.cloud.gcp.runners.gke import FlinkGKERunnerJob, GKERunnerJob
from axlearn.common.config import config_for_function


def named_runner_configs(
    runner: Optional[str] = None,
) -> Union[BaseRunnerJob.Config, dict[str, BaseRunnerJob.Config]]:
    """Returns runner config(s) optionally filtered by name."""

    runners = {
        "gke_tpu_single": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=TPUReplicatedJob.default_config(),
                annotations=config_for_function(exclusive_topology_annotations),
            ),
            pre_provisioner=TPUNodePoolProvisioner.default_config(),
        ),
        "gke_gpu_a3_single": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=A3ReplicatedJob.default_config(),
            ),
        ),
        "gke_tpu_flink": FlinkGKERunnerJob.default_config().set(
            inner=FlinkTPUGKEJob.default_config().set(
                builder=TPUReplicatedJob.default_config(),
            ),
            pre_provisioner=TPUNodePoolProvisioner.default_config(),
        ),
    }

    # Returning all runners is useful for users to discover runners (e.g. CLI help).
    if runner is None:
        return runners
    elif runner in runners:
        return runners[runner]
    raise ValueError(f"Unrecognized runner: {runner}")
