# Copyright © 2025 Apple Inc.

"""A collection of built-in bastion runners."""

from typing import Optional, Union

from axlearn.cloud.gcp.job import GKEJob, exclusive_topology_annotations
from axlearn.cloud.gcp.job_flink import FlinkTPUGKEJob
from axlearn.cloud.gcp.jobset_utils import (
    A3HighReplicatedJob,
    A3MegaReplicatedJob,
    A3UltraReplicatedJob,
    A4HighReplicatedJob,
    TPUReplicatedJob,
)
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
        "gke_tpu_flink": FlinkGKERunnerJob.default_config().set(
            inner=FlinkTPUGKEJob.default_config().set(
                builder=TPUReplicatedJob.default_config(),
            ),
            pre_provisioner=TPUNodePoolProvisioner.default_config(),
        ),
        "gke_gpu_a3_high": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=A3HighReplicatedJob.default_config(),
            ),
        ),
        "gke_gpu_a3_mega": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=A3MegaReplicatedJob.default_config(),
            ),
        ),
        "gke_gpu_a3_ultra": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=A3UltraReplicatedJob.default_config(),
            ),
        ),
        "gke_gpu_a4_high": GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(
                builder=A4HighReplicatedJob.default_config(),
            ),
        ),
    }

    # Returning all runners is useful for users to discover runners (e.g. CLI help).
    if runner is None:
        return runners
    elif runner in runners:
        return runners[runner]
    raise ValueError(f"Unrecognized runner: {runner}")
