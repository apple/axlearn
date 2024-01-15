"""Utilities for ahead-of-time compilation.

Reference:
https://docs.google.com/document/d/1Y5IdmvAZA7UtMHAWkRh8k2PscVoG5FvMH9-E6hygsyY/
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TPU_SKIP_MDS_QUERY"] = "1"

import copy
from dataclasses import dataclass
from typing import Callable, Dict

import jax
jax.config.update("JAX_PLATFORMS", "cpu")
import jax.random
import numpy as np
from jax.experimental.topologies import get_topology_desc

from axlearn.common.trainer import SpmdTrainer


@dataclass
class SystemCharacteristics:
    platform: str
    topology_name: str
    chip_config_name: str  # 'megacore' or 'default'
    chips_per_host_bounds: tuple
    devices_per_slice: int


USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS = {
    "v5e-16": SystemCharacteristics("tpu", "v5e:4x4", "default", (2, 2, 1), 16),
    "v5e-32": SystemCharacteristics("tpu", "v5e:4x8", "default", (2, 2, 1), 32),
    "v5e-64": SystemCharacteristics("tpu", "v5e:8x8", "default", (2, 2, 1), 64),
    "v5e-128": SystemCharacteristics("tpu", "v5e:8x16", "default", (2, 2, 1), 128),
    "v5e-256": SystemCharacteristics("tpu", "v5e:16x16", "default", (2, 2, 1), 256),
    "v4-8": SystemCharacteristics("tpu", "v4:2x2x1", "megacore", (2, 2, 1), 4),
    "v4-16": SystemCharacteristics("tpu", "v4:2x2x2", "megacore", (2, 2, 1), 8),
    "v4-32": SystemCharacteristics("tpu", "v4:2x2x4", "megacore", (2, 2, 1), 16),
    "v4-64": SystemCharacteristics("tpu", "v4:2x4x4", "megacore", (2, 2, 1), 32),
    "v4-128": SystemCharacteristics("tpu", "v4:4x4x4", "megacore", (2, 2, 1), 64),
    "v4-256": SystemCharacteristics("tpu", "v4:4x4x8", "megacore", (2, 2, 1), 128),
    "v4-512": SystemCharacteristics("tpu", "v4:4x8x8", "megacore", (2, 2, 1), 256),
    "v4-1024": SystemCharacteristics("tpu", "v4:8x8x8", "megacore", (2, 2, 1), 512),
    "v4-1536": SystemCharacteristics("tpu", "v4:8x8x12", "megacore", (2, 2, 1), 768),
    "v4-2048": SystemCharacteristics("tpu", "v4:8x8x16", "megacore", (2, 2, 1), 1024),
    "v4-4096": SystemCharacteristics("tpu", "v4:8x16x16", "megacore", (2, 2, 1), 2048),
}


def compile_trainer_programs(
    trainer_config: SpmdTrainer.Config, *, topology: str, topology_num_slices: int = 1
) -> Dict[str, Callable]:
    """Returns compiled XLA programs for the given trainer.

    Args:
        trainer_config: the trainer config.
        topology: a string representing the TPU topology, e.g., "v4-8". Must be a key in
            USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
        topology_num_slices: number of TPU slices.

    Returns:
        A dict containing the following programs:
        * "train_step": a program to run a single training step.
    """
    if topology is not None:
        target_hardware = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[topology]
        topology_devices = get_topology_desc(
            platform=target_hardware.platform,
            topology_name=target_hardware.topology_name,
            chip_config_name=target_hardware.chip_config_name,
            chips_per_host_bounds=target_hardware.chips_per_host_bounds,
            num_slices=topology_num_slices,
        ).devices
    else:
        topology_devices = jax.devices()

    cfg = copy.deepcopy(trainer_config)
    cfg.dir = "NOT_USED"
    cfg.mesh_shape = [len(topology_devices)] + [1] * (len(cfg.mesh_axis_names) - 1)
    topology_devices = np.reshape(topology_devices, cfg.mesh_shape)
    trainer: SpmdTrainer = cfg.instantiate(parent=None, devices=topology_devices)
    compiled_train_step = trainer.compile_train_step()
    return {"train_step": compiled_train_step}
