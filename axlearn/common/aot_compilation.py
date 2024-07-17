# Copyright © 2023 Apple Inc.

"""Utilities for ahead-of-time compilation.

Reference:
https://docs.google.com/document/d/1Y5IdmvAZA7UtMHAWkRh8k2PscVoG5FvMH9-E6hygsyY/
"""

import os
from dataclasses import dataclass
from typing import Dict

import jax
import jax.random
import numpy as np
from jax.experimental.topologies import get_topology_desc

from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import HybridMeshShape, MeshShape, infer_mesh_shape

os.environ["TPU_SKIP_MDS_QUERY"] = "1"

# To avoid error: Unable to initialize backend 'tpu'.
jax.config.update("jax_platforms", "cpu")


@dataclass
class SystemCharacteristics:
    platform: str
    topology_name: str
    chip_config_name: str  # 'megacore' or 'default'
    chips_per_host_bounds: tuple
    devices_per_slice: int


# TODO(markblee): Dedup with `axlearn/cloud/gcp/system_characteristics.py`.
# Reference: https://github.com/google/maxtext/blob/main/MaxText/accelerator_to_spec_map.py
USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS = {
    # v5e
    "v5e-16": SystemCharacteristics("tpu", "v5e:4x4", "default", (2, 2, 1), 16),
    "v5e-32": SystemCharacteristics("tpu", "v5e:4x8", "default", (2, 2, 1), 32),
    "v5e-64": SystemCharacteristics("tpu", "v5e:8x8", "default", (2, 2, 1), 64),
    "v5e-128": SystemCharacteristics("tpu", "v5e:8x16", "default", (2, 2, 1), 128),
    "v5e-256": SystemCharacteristics("tpu", "v5e:16x16", "default", (2, 2, 1), 256),
    # v4
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
    # v5p
    "v5p-8": SystemCharacteristics("tpu", "v5:2x2x1", "megacore", (2, 2, 1), 4),
    "v5p-16": SystemCharacteristics("tpu", "v5:2x2x2", "megacore", (2, 2, 1), 8),
    "v5p-32": SystemCharacteristics("tpu", "v5:2x2x4", "megacore", (2, 2, 1), 16),
    "v5p-64": SystemCharacteristics("tpu", "v5:2x4x4", "megacore", (2, 2, 1), 32),
    "v5p-128": SystemCharacteristics("tpu", "v5:4x4x4", "megacore", (2, 2, 1), 64),
    "v5p-256": SystemCharacteristics("tpu", "v5:4x4x8", "megacore", (2, 2, 1), 128),
    "v5p-384": SystemCharacteristics("tpu", "v5:4x4x12", "megacore", (2, 2, 1), 192),
    "v5p-512": SystemCharacteristics("tpu", "v5:4x8x8", "megacore", (2, 2, 1), 256),
    "v5p-640": SystemCharacteristics("tpu", "v5:4x4x20", "megacore", (2, 2, 1), 320),
    "v5p-768": SystemCharacteristics("tpu", "v5:4x8x12", "megacore", (2, 2, 1), 384),
    "v5p-896": SystemCharacteristics("tpu", "v5:4x4x28", "megacore", (2, 2, 1), 448),
    "v5p-1024": SystemCharacteristics("tpu", "v5:8x8x8", "megacore", (2, 2, 1), 512),
    "v5p-1152": SystemCharacteristics("tpu", "v5:4x12x12", "megacore", (2, 2, 1), 576),
    "v5p-1280": SystemCharacteristics("tpu", "v5:4x8x20", "megacore", (2, 2, 1), 640),
    "v5p-1408": SystemCharacteristics("tpu", "v5:4x4x44", "megacore", (2, 2, 1), 704),
    "v5p-1536": SystemCharacteristics("tpu", "v5:8x8x12", "megacore", (2, 2, 1), 768),
    "v5p-1664": SystemCharacteristics("tpu", "v5:4x4x52", "megacore", (2, 2, 1), 832),
    "v5p-1792": SystemCharacteristics("tpu", "v5:4x8x28", "megacore", (2, 2, 1), 896),
    "v5p-1920": SystemCharacteristics("tpu", "v5:4x12x20", "megacore", (2, 2, 1), 960),
    "v5p-2048": SystemCharacteristics("tpu", "v5:8x8x16", "megacore", (2, 2, 1), 1024),
    "v5p-2176": SystemCharacteristics("tpu", "v5:4x4x68", "megacore", (2, 2, 1), 1088),
    "v5p-2304": SystemCharacteristics("tpu", "v5:8x12x12", "megacore", (2, 2, 1), 1152),
    "v5p-2432": SystemCharacteristics("tpu", "v5:4x4x76", "megacore", (2, 2, 1), 1216),
    "v5p-2560": SystemCharacteristics("tpu", "v5:8x8x20", "megacore", (2, 2, 1), 1280),
    "v5p-2688": SystemCharacteristics("tpu", "v5:4x12x28", "megacore", (2, 2, 1), 1344),
    "v5p-2816": SystemCharacteristics("tpu", "v5:4x8x44", "megacore", (2, 2, 1), 1408),
    "v5p-2944": SystemCharacteristics("tpu", "v5:4x4x92", "megacore", (2, 2, 1), 1472),
    "v5p-3072": SystemCharacteristics("tpu", "v5:4x12x16", "megacore", (2, 2, 1), 1536),
    "v5p-3200": SystemCharacteristics("tpu", "v5:4x20x20", "megacore", (2, 2, 1), 1600),
    "v5p-3328": SystemCharacteristics("tpu", "v5:4x8x52", "megacore", (2, 2, 1), 1664),
    "v5p-3456": SystemCharacteristics("tpu", "v5:12x12x12", "megacore", (2, 2, 1), 1728),
    "v5p-3584": SystemCharacteristics("tpu", "v5:8x8x28", "megacore", (2, 2, 1), 1792),
    "v5p-3712": SystemCharacteristics("tpu", "v5:4x4x116", "megacore", (2, 2, 1), 1856),
    "v5p-3840": SystemCharacteristics("tpu", "v5:8x12x20", "megacore", (2, 2, 1), 1920),
    "v5p-3968": SystemCharacteristics("tpu", "v5:4x4x124", "megacore", (2, 2, 1), 1984),
    "v5p-4096": SystemCharacteristics("tpu", "v5:8x16x16", "megacore", (2, 2, 1), 2048),
    "v5p-4224": SystemCharacteristics("tpu", "v5:4x12x44", "megacore", (2, 2, 1), 2112),
    "v5p-4352": SystemCharacteristics("tpu", "v5:4x8x68", "megacore", (2, 2, 1), 2176),
    "v5p-4480": SystemCharacteristics("tpu", "v5:4x20x28", "megacore", (2, 2, 1), 2240),
    "v5p-4608": SystemCharacteristics("tpu", "v5:12x12x16", "megacore", (2, 2, 1), 2304),
    "v5p-4736": SystemCharacteristics("tpu", "v5:4x4x148", "megacore", (2, 2, 1), 2368),
    "v5p-4864": SystemCharacteristics("tpu", "v5:4x8x76", "megacore", (2, 2, 1), 2432),
    "v5p-4992": SystemCharacteristics("tpu", "v5:4x12x52", "megacore", (2, 2, 1), 2496),
    "v5p-5120": SystemCharacteristics("tpu", "v5:8x16x20", "megacore", (2, 2, 1), 2560),
    "v5p-5248": SystemCharacteristics("tpu", "v5:4x4x164", "megacore", (2, 2, 1), 2624),
    "v5p-5376": SystemCharacteristics("tpu", "v5:8x12x28", "megacore", (2, 2, 1), 2688),
    "v5p-5504": SystemCharacteristics("tpu", "v5:4x4x172", "megacore", (2, 2, 1), 2752),
    "v5p-5632": SystemCharacteristics("tpu", "v5:8x8x44", "megacore", (2, 2, 1), 2816),
    "v5p-5760": SystemCharacteristics("tpu", "v5:12x12x20", "megacore", (2, 2, 1), 2880),
    "v5p-5888": SystemCharacteristics("tpu", "v5:4x8x92", "megacore", (2, 2, 1), 2944),
    "v5p-6016": SystemCharacteristics("tpu", "v5:4x4x188", "megacore", (2, 2, 1), 3008),
    "v5p-6144": SystemCharacteristics("tpu", "v5:12x16x16", "megacore", (2, 2, 1), 3072),
    "v5p-6272": SystemCharacteristics("tpu", "v5:4x28x28", "megacore", (2, 2, 1), 3136),
    "v5p-6400": SystemCharacteristics("tpu", "v5:8x20x20", "megacore", (2, 2, 1), 3200),
    "v5p-6528": SystemCharacteristics("tpu", "v5:4x12x68", "megacore", (2, 2, 1), 3264),
    "v5p-6656": SystemCharacteristics("tpu", "v5:8x8x52", "megacore", (2, 2, 1), 3328),
    "v5p-6784": SystemCharacteristics("tpu", "v5:4x4x212", "megacore", (2, 2, 1), 3392),
    "v5p-6912": SystemCharacteristics("tpu", "v5:12x12x24", "megacore", (2, 2, 1), 3456),
    "v5p-7040": SystemCharacteristics("tpu", "v5:4x20x44", "megacore", (2, 2, 1), 3520),
    "v5p-7168": SystemCharacteristics("tpu", "v5:8x16x28", "megacore", (2, 2, 1), 3584),
    "v5p-7296": SystemCharacteristics("tpu", "v5:4x12x76", "megacore", (2, 2, 1), 3648),
    "v5p-7424": SystemCharacteristics("tpu", "v5:4x8x116", "megacore", (2, 2, 1), 3712),
    "v5p-7552": SystemCharacteristics("tpu", "v5:4x4x236", "megacore", (2, 2, 1), 3776),
    "v5p-7680": SystemCharacteristics("tpu", "v5:12x16x20", "megacore", (2, 2, 1), 3840),
    "v5p-7808": SystemCharacteristics("tpu", "v5:4x4x244", "megacore", (2, 2, 1), 3904),
    "v5p-7936": SystemCharacteristics("tpu", "v5:4x8x124", "megacore", (2, 2, 1), 3968),
    "v5p-8064": SystemCharacteristics("tpu", "v5:12x12x28", "megacore", (2, 2, 1), 4032),
    "v5p-8192": SystemCharacteristics("tpu", "v5:16x16x16", "megacore", (2, 2, 1), 4096),
    "v5p-8320": SystemCharacteristics("tpu", "v5:4x20x52", "megacore", (2, 2, 1), 4160),
    "v5p-8448": SystemCharacteristics("tpu", "v5:8x12x44", "megacore", (2, 2, 1), 4224),
    "v5p-8704": SystemCharacteristics("tpu", "v5:8x8x68", "megacore", (2, 2, 1), 4352),
    "v5p-8832": SystemCharacteristics("tpu", "v5:4x12x92", "megacore", (2, 2, 1), 4416),
    "v5p-8960": SystemCharacteristics("tpu", "v5:8x20x28", "megacore", (2, 2, 1), 4480),
    "v5p-9216": SystemCharacteristics("tpu", "v5:12x16x24", "megacore", (2, 2, 1), 4608),
    "v5p-9472": SystemCharacteristics("tpu", "v5:4x8x148", "megacore", (2, 2, 1), 4736),
    "v5p-9600": SystemCharacteristics("tpu", "v5:12x20x20", "megacore", (2, 2, 1), 4800),
    "v5p-9728": SystemCharacteristics("tpu", "v5:8x8x76", "megacore", (2, 2, 1), 4864),
    "v5p-9856": SystemCharacteristics("tpu", "v5:4x28x44", "megacore", (2, 2, 1), 4928),
    "v5p-9984": SystemCharacteristics("tpu", "v5:8x12x52", "megacore", (2, 2, 1), 4992),
    "v5p-10240": SystemCharacteristics("tpu", "v5:16x16x20", "megacore", (2, 2, 1), 5120),
    "v5p-10368": SystemCharacteristics("tpu", "v5:12x12x36", "megacore", (2, 2, 1), 5184),
    "v5p-10496": SystemCharacteristics("tpu", "v5:4x8x164", "megacore", (2, 2, 1), 5248),
    "v5p-10752": SystemCharacteristics("tpu", "v5:12x16x28", "megacore", (2, 2, 1), 5376),
    "v5p-10880": SystemCharacteristics("tpu", "v5:4x20x68", "megacore", (2, 2, 1), 5440),
    "v5p-11008": SystemCharacteristics("tpu", "v5:4x8x172", "megacore", (2, 2, 1), 5504),
    "v5p-11136": SystemCharacteristics("tpu", "v5:4x12x116", "megacore", (2, 2, 1), 5568),
    "v5p-11264": SystemCharacteristics("tpu", "v5:8x16x44", "megacore", (2, 2, 1), 5632),
    "v5p-11520": SystemCharacteristics("tpu", "v5:12x20x24", "megacore", (2, 2, 1), 5760),
    "v5p-11648": SystemCharacteristics("tpu", "v5:4x28x52", "megacore", (2, 2, 1), 5824),
    "v5p-11776": SystemCharacteristics("tpu", "v5:8x8x92", "megacore", (2, 2, 1), 5888),
    "v5p-11904": SystemCharacteristics("tpu", "v5:4x12x124", "megacore", (2, 2, 1), 5952),
    "v5p-12032": SystemCharacteristics("tpu", "v5:4x8x188", "megacore", (2, 2, 1), 6016),
    "v5p-12160": SystemCharacteristics("tpu", "v5:4x20x76", "megacore", (2, 2, 1), 6080),
    "v5p-12288": SystemCharacteristics("tpu", "v5:16x16x24", "megacore", (2, 2, 1), 6144),
    "v5p-13824": SystemCharacteristics("tpu", "v5:12x24x24", "megacore", (2, 2, 1), 6912),
    "v5p-17920": SystemCharacteristics("tpu", "v5:16x20x28", "megacore", (2, 2, 1), 8960),
}


def compile_trainer_programs(
    trainer_config: SpmdTrainer.Config, *, topology: str, topology_num_slices: int = 1
) -> Dict[str, jax.stages.Compiled]:
    """Returns compiled XLA programs for the given trainer.

    Args:
        trainer_config: The trainer config.
        topology: A string representing the TPU topology, e.g., "v4-8". Must be a key in
            USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
            If None, use CPU devices.
        topology_num_slices: The number of TPU slices.

    Returns:
        A dict containing the following programs:
        * "train_step": a program to run a single training step.

    Raises:
        NotImplementedError: if `topology` is not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
    """
    if topology is not None:
        if topology not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(
                f"Unsupported topology {topology}. "
                f"Supported values are {USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.keys()}"
            )
        target_hardware = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[topology]
        devices_per_slice = target_hardware.devices_per_slice
        topology_devices = get_topology_desc(
            platform=target_hardware.platform,
            topology_name=target_hardware.topology_name,
            chip_config_name=target_hardware.chip_config_name,
            chips_per_host_bounds=target_hardware.chips_per_host_bounds,
            num_slices=topology_num_slices,
        ).devices
    else:
        topology_devices = jax.devices()
        assert topology_num_slices == 1
        devices_per_slice = len(topology_devices)

    cfg = trainer_config.clone(dir="NOT_USED")
    if cfg.mesh_shape is None:
        cfg.mesh_shape = [len(topology_devices)] + [1] * (len(cfg.mesh_axis_names) - 1)
    elif isinstance(cfg.mesh_shape, MeshShape):
        cfg.mesh_shape = infer_mesh_shape(cfg.mesh_shape, num_devices=len(topology_devices))
        topology_devices = np.reshape(topology_devices, cfg.mesh_shape)
    else:
        assert isinstance(cfg.mesh_shape, HybridMeshShape)
        cfg.mesh_shape = HybridMeshShape(
            ici_mesh_shape=infer_mesh_shape(
                cfg.mesh_shape.ici_mesh_shape, num_devices=devices_per_slice
            ),
            dcn_mesh_shape=infer_mesh_shape(
                cfg.mesh_shape.dcn_mesh_shape, num_devices=topology_num_slices
            ),
        )
        topology_devices = np.reshape(
            topology_devices,
            tuple(
                x * y for x, y in zip(cfg.mesh_shape.ici_mesh_shape, cfg.mesh_shape.dcn_mesh_shape)
            ),
        )
    trainer: SpmdTrainer = cfg.instantiate(parent=None, devices=topology_devices)
    compiled_train_step = trainer.compile_train_step()
    return {"train_step": compiled_train_step}
