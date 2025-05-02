# Copyright © 2025 Apple Inc.

"""A collection of built-in bastion runners."""

from typing import Optional, Union

from axlearn.cloud.gcp.job import GKEJob, exclusive_topology_annotations
from axlearn.cloud.gcp.job_flink import FlinkTPUGKEJob
from axlearn.cloud.gcp.job_pathways import GKEPathwaysJobSet
from axlearn.cloud.gcp.jobset_utils import (
    A3HighReplicatedJob,
    A3MegaReplicatedJob,
    A3UltraReplicatedJob,
    A4HighReplicatedJob,
    TPUReplicatedJob,
)
from axlearn.cloud.gcp.node_pool_provisioner import TPUNodePoolProvisioner
from axlearn.cloud.gcp.pathways_utils import PathwaysMultiheadReplicatedJob, PathwaysReplicatedJob
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
        "gke_tpu_pathways": GKERunnerJob.default_config().set(
            inner=GKEPathwaysJobSet.default_config().set(
                builder=PathwaysReplicatedJob.default_config()
            ),
        ),
        "gke_tpu_pathways_multihead": GKERunnerJob.default_config().set(
            inner=GKEPathwaysJobSet.default_config().set(
                builder=PathwaysMultiheadReplicatedJob.default_config()
            ),
        ),
    }

    # Get the GPU runners from the helper function
    runners.update(_get_gpu_runners())

    # Returning all runners is useful for users to discover runners (e.g. CLI help).
    if runner is None:
        return runners
    elif runner in runners:
        return runners[runner]
    raise ValueError(f"Unrecognized runner: {runner}")


def _get_gpu_runners() -> dict[str, BaseRunnerJob.Config]:
    """Creates a list of GPU runners."""

    runner_list = {
        "gke_gpu_a3_high_single": A3HighReplicatedJob,
        "gke_gpu_a3_mega_single": A3MegaReplicatedJob,
        "gke_gpu_a3_ultra_single": A3UltraReplicatedJob,
        "gke_gpu_a4_high_single": A4HighReplicatedJob,
    }

    runner_configs = {}

    for name, config in runner_list.items():
        runner_configs[name] = GKERunnerJob.default_config().set(
            inner=GKEJob.default_config().set(builder=config.default_config())
        )

    return runner_configs
