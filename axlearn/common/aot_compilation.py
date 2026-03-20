# Copyright © 2023 Apple Inc.

"""Utilities for ahead-of-time compilation.

Reference:
https://docs.google.com/document/d/1Y5IdmvAZA7UtMHAWkRh8k2PscVoG5FvMH9-E6hygsyY/
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.topologies import get_topology_desc

from axlearn.common.inference import InferenceRunner
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import HybridMeshShape, MeshShape, Nested, TensorSpec, infer_mesh_shape

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


# pylint: disable=invalid-name
def _create_system_characteristics(
    topology_name_prefix: str,
    chip_config_name: str,
    chips_per_host_bounds: tuple,
    topology_dims: str,
    devices_per_slice: int,
) -> SystemCharacteristics:
    return SystemCharacteristics(
        platform="tpu",
        topology_name=f"{topology_name_prefix}:{topology_dims}",
        chip_config_name=chip_config_name,
        chips_per_host_bounds=chips_per_host_bounds,
        devices_per_slice=devices_per_slice,
    )


_create_v6e_characteristics = partial(_create_system_characteristics, "v6e", "default", (2, 2, 1))
_create_v5e_characteristics = partial(_create_system_characteristics, "v5e", "default", (2, 2, 1))
_create_v4_characteristics = partial(_create_system_characteristics, "v4", "megacore", (2, 2, 1))
_create_v5p_characteristics = partial(_create_system_characteristics, "v5", "megacore", (2, 2, 1))
_create_7x_characteristics = partial(_create_system_characteristics, "tpu7x", "default", (2, 2, 1))
# pylint: enable=invalid-name

# TODO(markblee): Dedup with `axlearn/cloud/gcp/system_characteristics.py`.
# Reference: https://github.com/google/maxtext/blob/main/MaxText/accelerator_to_spec_map.py
USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS = {
    # v6e: one core per chip with 32 GB HBM
    "v6e-1": _create_system_characteristics("v6e", "default", (1, 1, 1), "1x1", 1),
    "v6e-4": _create_v6e_characteristics("2x2", 4),
    "v6e-8": _create_v6e_characteristics("2x4", 8),
    "v6e-16": _create_v6e_characteristics("4x4", 16),
    "v6e-32": _create_v6e_characteristics("4x8", 32),
    "v6e-64": _create_v6e_characteristics("8x8", 64),
    "v6e-128": _create_v6e_characteristics("8x16", 128),
    "v6e-256": _create_v6e_characteristics("16x16", 256),
    # v5e
    "v5e-1": _create_system_characteristics("v5e", "default", (1, 1, 1), "1x1", 1),
    "v5e-4": _create_v5e_characteristics("2x2", 4),
    "v5e-8": _create_v5e_characteristics("2x4", 8),
    "v5e-16": _create_v5e_characteristics("4x4", 16),
    "v5e-32": _create_v5e_characteristics("4x8", 32),
    "v5e-64": _create_v5e_characteristics("8x8", 64),
    "v5e-128": _create_v5e_characteristics("8x16", 128),
    "v5e-256": _create_v5e_characteristics("16x16", 256),
    # v4
    "v4-8": _create_v4_characteristics("2x2x1", 4),
    "v4-16": _create_v4_characteristics("2x2x2", 8),
    "v4-32": _create_v4_characteristics("2x2x4", 16),
    "v4-64": _create_v4_characteristics("2x4x4", 32),
    "v4-128": _create_v4_characteristics("4x4x4", 64),
    "v4-256": _create_v4_characteristics("4x4x8", 128),
    "v4-512": _create_v4_characteristics("4x8x8", 256),
    "v4-1024": _create_v4_characteristics("8x8x8", 512),
    "v4-1536": _create_v4_characteristics("8x8x12", 768),
    "v4-2048": _create_v4_characteristics("8x8x16", 1024),
    "v4-4096": _create_v4_characteristics("8x16x16", 2048),
    # v5p
    "v5p-8": _create_v5p_characteristics("2x2x1", 4),
    "v5p-16": _create_v5p_characteristics("2x2x2", 8),
    "v5p-32": _create_v5p_characteristics("2x2x4", 16),
    "v5p-64": _create_v5p_characteristics("2x4x4", 32),
    "v5p-128": _create_v5p_characteristics("4x4x4", 64),
    "v5p-256": _create_v5p_characteristics("4x4x8", 128),
    "v5p-384": _create_v5p_characteristics("4x4x12", 192),
    "v5p-512": _create_v5p_characteristics("4x8x8", 256),
    "v5p-640": _create_v5p_characteristics("4x4x20", 320),
    "v5p-768": _create_v5p_characteristics("4x8x12", 384),
    "v5p-896": _create_v5p_characteristics("4x4x28", 448),
    "v5p-1024": _create_v5p_characteristics("8x8x8", 512),
    "v5p-1152": _create_v5p_characteristics("4x12x12", 576),
    "v5p-1280": _create_v5p_characteristics("4x8x20", 640),
    "v5p-1408": _create_v5p_characteristics("4x4x44", 704),
    "v5p-1536": _create_v5p_characteristics("8x8x12", 768),
    "v5p-1664": _create_v5p_characteristics("4x4x52", 832),
    "v5p-1792": _create_v5p_characteristics("4x8x28", 896),
    "v5p-1920": _create_v5p_characteristics("4x12x20", 960),
    "v5p-2048": _create_v5p_characteristics("8x8x16", 1024),
    "v5p-2176": _create_v5p_characteristics("4x4x68", 1088),
    "v5p-2304": _create_v5p_characteristics("8x12x12", 1152),
    "v5p-2432": _create_v5p_characteristics("4x4x76", 1216),
    "v5p-2560": _create_v5p_characteristics("8x8x20", 1280),
    "v5p-2688": _create_v5p_characteristics("4x12x28", 1344),
    "v5p-2816": _create_v5p_characteristics("4x8x44", 1408),
    "v5p-2944": _create_v5p_characteristics("4x4x92", 1472),
    "v5p-3072": _create_v5p_characteristics("8x12x16", 1536),
    "v5p-3200": _create_v5p_characteristics("4x20x20", 1600),
    "v5p-3328": _create_v5p_characteristics("4x8x52", 1664),
    "v5p-3456": _create_v5p_characteristics("12x12x12", 1728),
    "v5p-3584": _create_v5p_characteristics("8x8x28", 1792),
    "v5p-3712": _create_v5p_characteristics("4x4x116", 1856),
    "v5p-3840": _create_v5p_characteristics("8x12x20", 1920),
    "v5p-3968": _create_v5p_characteristics("4x4x124", 1984),
    "v5p-4096": _create_v5p_characteristics("8x16x16", 2048),
    "v5p-4224": _create_v5p_characteristics("4x12x44", 2112),
    "v5p-4352": _create_v5p_characteristics("4x8x68", 2176),
    "v5p-4480": _create_v5p_characteristics("4x20x28", 2240),
    "v5p-4608": _create_v5p_characteristics("12x12x16", 2304),
    "v5p-4736": _create_v5p_characteristics("4x4x148", 2368),
    "v5p-4864": _create_v5p_characteristics("4x8x76", 2432),
    "v5p-4992": _create_v5p_characteristics("4x12x52", 2496),
    "v5p-5120": _create_v5p_characteristics("8x16x20", 2560),
    "v5p-5248": _create_v5p_characteristics("4x4x164", 2624),
    "v5p-5376": _create_v5p_characteristics("8x12x28", 2688),
    "v5p-5504": _create_v5p_characteristics("4x4x172", 2752),
    "v5p-5632": _create_v5p_characteristics("8x8x44", 2816),
    "v5p-5760": _create_v5p_characteristics("12x12x20", 2880),
    "v5p-5888": _create_v5p_characteristics("4x8x92", 2944),
    "v5p-6016": _create_v5p_characteristics("4x4x188", 3008),
    "v5p-6144": _create_v5p_characteristics("12x16x16", 3072),
    "v5p-6272": _create_v5p_characteristics("4x28x28", 3136),
    "v5p-6400": _create_v5p_characteristics("8x20x20", 3200),
    "v5p-6528": _create_v5p_characteristics("4x12x68", 3264),
    "v5p-6656": _create_v5p_characteristics("8x8x52", 3328),
    "v5p-6784": _create_v5p_characteristics("4x4x212", 3392),
    "v5p-6912": _create_v5p_characteristics("12x12x24", 3456),
    "v5p-7040": _create_v5p_characteristics("4x20x44", 3520),
    "v5p-7168": _create_v5p_characteristics("8x16x28", 3584),
    "v5p-7296": _create_v5p_characteristics("4x12x76", 3648),
    "v5p-7424": _create_v5p_characteristics("4x8x116", 3712),
    "v5p-7552": _create_v5p_characteristics("4x4x236", 3776),
    "v5p-7680": _create_v5p_characteristics("12x16x20", 3840),
    "v5p-7808": _create_v5p_characteristics("4x4x244", 3904),
    "v5p-7936": _create_v5p_characteristics("4x8x124", 3968),
    "v5p-8064": _create_v5p_characteristics("12x12x28", 4032),
    "v5p-8192": _create_v5p_characteristics("16x16x16", 4096),
    "v5p-8320": _create_v5p_characteristics("4x20x52", 4160),
    "v5p-8448": _create_v5p_characteristics("8x12x44", 4224),
    "v5p-8704": _create_v5p_characteristics("8x8x68", 4352),
    "v5p-8832": _create_v5p_characteristics("4x12x92", 4416),
    "v5p-8960": _create_v5p_characteristics("8x20x28", 4480),
    "v5p-9216": _create_v5p_characteristics("12x16x24", 4608),
    "v5p-9472": _create_v5p_characteristics("4x8x148", 4736),
    "v5p-9600": _create_v5p_characteristics("12x20x20", 4800),
    "v5p-9728": _create_v5p_characteristics("8x8x76", 4864),
    "v5p-9856": _create_v5p_characteristics("4x28x44", 4928),
    "v5p-9984": _create_v5p_characteristics("8x12x52", 4992),
    "v5p-10240": _create_v5p_characteristics("16x16x20", 5120),
    "v5p-10368": _create_v5p_characteristics("12x12x36", 5184),
    "v5p-10496": _create_v5p_characteristics("4x8x164", 5248),
    "v5p-10752": _create_v5p_characteristics("12x16x28", 5376),
    "v5p-10880": _create_v5p_characteristics("4x20x68", 5440),
    "v5p-11008": _create_v5p_characteristics("4x8x172", 5504),
    "v5p-11136": _create_v5p_characteristics("4x12x116", 5568),
    "v5p-11264": _create_v5p_characteristics("8x16x44", 5632),
    "v5p-11520": _create_v5p_characteristics("12x20x24", 5760),
    "v5p-11648": _create_v5p_characteristics("4x28x52", 5824),
    "v5p-11776": _create_v5p_characteristics("8x8x92", 5888),
    "v5p-11904": _create_v5p_characteristics("4x12x124", 5952),
    "v5p-12032": _create_v5p_characteristics("4x8x188", 6016),
    "v5p-12160": _create_v5p_characteristics("4x20x76", 6080),
    "v5p-12288": _create_v5p_characteristics("16x16x24", 6144),
    "v5p-13824": _create_v5p_characteristics("12x24x24", 6912),
    "v5p-17920": _create_v5p_characteristics("16x20x28", 8960),
    # v7x: one chip has 2 TensorCores exposed as separate devices
    "tpu7x-2": _create_system_characteristics("tpu7x", "default", (1, 1, 1), "1x1x1", 2),
    "tpu7x-8": _create_7x_characteristics("2x2x1", 8),
    "tpu7x-16": _create_7x_characteristics("2x2x2", 16),
    "tpu7x-32": _create_7x_characteristics("2x2x4", 32),
    "tpu7x-64": _create_7x_characteristics("2x4x4", 64),
    "tpu7x-128": _create_7x_characteristics("4x4x4", 128),
    "tpu7x-256": _create_7x_characteristics("4x4x8", 256),
    "tpu7x-384": _create_7x_characteristics("4x4x12", 384),
    "tpu7x-512": _create_7x_characteristics("4x8x8", 512),
    "tpu7x-640": _create_7x_characteristics("4x4x20", 640),
    "tpu7x-768": _create_7x_characteristics("4x8x12", 768),
    "tpu7x-896": _create_7x_characteristics("4x4x28", 896),
    "tpu7x-1024": _create_7x_characteristics("8x8x8", 1024),
    "tpu7x-1152": _create_7x_characteristics("4x12x12", 1152),
    "tpu7x-1280": _create_7x_characteristics("4x8x20", 1280),
    "tpu7x-1408": _create_7x_characteristics("4x4x44", 1408),
    "tpu7x-1536": _create_7x_characteristics("8x8x12", 1536),
    "tpu7x-1664": _create_7x_characteristics("4x4x52", 1664),
    "tpu7x-1792": _create_7x_characteristics("4x8x28", 1792),
    "tpu7x-1920": _create_7x_characteristics("4x12x20", 1920),
    "tpu7x-2048": _create_7x_characteristics("8x8x16", 2048),
    "tpu7x-2176": _create_7x_characteristics("4x4x68", 2176),
    "tpu7x-2304": _create_7x_characteristics("8x12x12", 2304),
    "tpu7x-2432": _create_7x_characteristics("4x4x76", 2432),
    "tpu7x-2560": _create_7x_characteristics("8x8x20", 2560),
    "tpu7x-2688": _create_7x_characteristics("4x12x28", 2688),
    "tpu7x-2816": _create_7x_characteristics("4x8x44", 2816),
    "tpu7x-2944": _create_7x_characteristics("4x4x92", 2944),
    "tpu7x-3072": _create_7x_characteristics("8x12x16", 3072),
    "tpu7x-3200": _create_7x_characteristics("4x20x20", 3200),
    "tpu7x-3328": _create_7x_characteristics("4x8x52", 3328),
    "tpu7x-3456": _create_7x_characteristics("12x12x12", 3456),
    "tpu7x-3584": _create_7x_characteristics("8x8x28", 3584),
    "tpu7x-3712": _create_7x_characteristics("4x4x116", 3712),
    "tpu7x-3840": _create_7x_characteristics("8x12x20", 3840),
    "tpu7x-3968": _create_7x_characteristics("4x4x124", 3968),
    "tpu7x-4096": _create_7x_characteristics("8x16x16", 4096),
    "tpu7x-4224": _create_7x_characteristics("4x12x44", 4224),
    "tpu7x-4352": _create_7x_characteristics("4x8x68", 4352),
    "tpu7x-4480": _create_7x_characteristics("4x20x28", 4480),
    "tpu7x-4608": _create_7x_characteristics("12x12x16", 4608),
    "tpu7x-4736": _create_7x_characteristics("4x4x148", 4736),
    "tpu7x-4864": _create_7x_characteristics("4x8x76", 4864),
    "tpu7x-4992": _create_7x_characteristics("4x12x52", 4992),
    "tpu7x-5120": _create_7x_characteristics("8x16x20", 5120),
    "tpu7x-5248": _create_7x_characteristics("4x4x164", 5248),
    "tpu7x-5376": _create_7x_characteristics("8x12x28", 5376),
    "tpu7x-5504": _create_7x_characteristics("4x4x172", 5504),
    "tpu7x-5632": _create_7x_characteristics("8x8x44", 5632),
    "tpu7x-5760": _create_7x_characteristics("12x12x20", 5760),
    "tpu7x-5888": _create_7x_characteristics("4x8x92", 5888),
    "tpu7x-6016": _create_7x_characteristics("4x4x188", 6016),
    "tpu7x-6144": _create_7x_characteristics("12x16x16", 6144),
    "tpu7x-6272": _create_7x_characteristics("4x28x28", 6272),
    "tpu7x-6400": _create_7x_characteristics("8x20x20", 6400),
    "tpu7x-6528": _create_7x_characteristics("4x12x68", 6528),
    "tpu7x-6656": _create_7x_characteristics("8x8x52", 6656),
    "tpu7x-6784": _create_7x_characteristics("4x4x212", 6784),
    "tpu7x-6912": _create_7x_characteristics("12x12x24", 6912),
    "tpu7x-7040": _create_7x_characteristics("4x20x44", 7040),
    "tpu7x-7168": _create_7x_characteristics("8x16x28", 7168),
    "tpu7x-7296": _create_7x_characteristics("4x12x76", 7296),
    "tpu7x-7424": _create_7x_characteristics("4x8x116", 7424),
    "tpu7x-7552": _create_7x_characteristics("4x4x236", 7552),
    "tpu7x-7680": _create_7x_characteristics("12x16x20", 7680),
    "tpu7x-7808": _create_7x_characteristics("4x4x244", 7808),
    "tpu7x-7936": _create_7x_characteristics("4x8x124", 7936),
    "tpu7x-8064": _create_7x_characteristics("12x12x28", 8064),
    "tpu7x-8192": _create_7x_characteristics("16x16x16", 8192),
    "tpu7x-8320": _create_7x_characteristics("4x20x52", 8320),
    "tpu7x-8448": _create_7x_characteristics("8x12x44", 8448),
    "tpu7x-8704": _create_7x_characteristics("8x8x68", 8704),
    "tpu7x-8832": _create_7x_characteristics("4x12x92", 8832),
    "tpu7x-8960": _create_7x_characteristics("8x20x28", 8960),
    "tpu7x-9216": _create_7x_characteristics("12x16x24", 9216),
    "tpu7x-9472": _create_7x_characteristics("4x8x148", 9472),
    "tpu7x-9600": _create_7x_characteristics("12x20x20", 9600),
    "tpu7x-9728": _create_7x_characteristics("8x8x76", 9728),
    "tpu7x-9856": _create_7x_characteristics("4x28x44", 9856),
    "tpu7x-9984": _create_7x_characteristics("8x12x52", 9984),
    "tpu7x-10240": _create_7x_characteristics("16x16x20", 10240),
    "tpu7x-10368": _create_7x_characteristics("12x12x36", 10368),
    "tpu7x-10496": _create_7x_characteristics("4x8x164", 10496),
    "tpu7x-10752": _create_7x_characteristics("12x16x28", 10752),
    "tpu7x-10880": _create_7x_characteristics("4x20x68", 10880),
    "tpu7x-11008": _create_7x_characteristics("4x8x172", 11008),
    "tpu7x-11136": _create_7x_characteristics("4x12x116", 11136),
    "tpu7x-11264": _create_7x_characteristics("8x16x44", 11264),
    "tpu7x-11520": _create_7x_characteristics("12x20x24", 11520),
    "tpu7x-11648": _create_7x_characteristics("4x28x52", 11648),
    "tpu7x-11776": _create_7x_characteristics("8x8x92", 11776),
    "tpu7x-11904": _create_7x_characteristics("4x12x124", 11904),
    "tpu7x-12032": _create_7x_characteristics("4x8x188", 12032),
    "tpu7x-12160": _create_7x_characteristics("4x20x76", 12160),
    "tpu7x-12288": _create_7x_characteristics("16x16x24", 12288),
    "tpu7x-13824": _create_7x_characteristics("12x24x24", 13824),
    "tpu7x-16384": _create_7x_characteristics("16x16x32", 16384),
    "tpu7x-17920": _create_7x_characteristics("16x20x28", 17920),
    "tpu7x-18432": _create_7x_characteristics("16x24x24", 18432),
}


def get_devices_for_topology(
    topology: str, topology_num_slices: int = 1
) -> Tuple[List[jax.Device], int]:
    """Returns a list of XLA devices for the given topology.

    Args:
        topology: A string representing the TPU topology, e.g., "v4-8". Must be a key in
            USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
            If None, use CPU devices.
        topology_num_slices: The number of TPU slices.

    Returns:
        A list of devices, and number of devices per slice.
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
    return topology_devices, devices_per_slice


def reshape_devices(
    *,
    devices: List[jax.Device],
    mesh_shape: Union[MeshShape, HybridMeshShape],
    devices_per_slice: int,
    num_slices: int,
) -> tuple[np.ndarray, Union[MeshShape, HybridMeshShape]]:
    """Reshape device list based on mesh_shape.

    Args:
        devices: A list of devices.
        mesh_shape: Mesh shape of the devices. Missing specifications (-1) will be inferred.
        devices_per_slice: Number of devices per slice.
        num_slices: Number of slices.

    Returns:
        A shaped ndarray of devices, and the inferred mesh shape.
    """
    if isinstance(mesh_shape, MeshShape):
        mesh_shape = infer_mesh_shape(mesh_shape, num_devices=devices_per_slice * num_slices)
        devices = np.reshape(devices, mesh_shape)
    elif isinstance(mesh_shape, HybridMeshShape):
        mesh_shape = HybridMeshShape(
            ici_mesh_shape=infer_mesh_shape(
                mesh_shape.ici_mesh_shape, num_devices=devices_per_slice
            ),
            dcn_mesh_shape=infer_mesh_shape(mesh_shape.dcn_mesh_shape, num_devices=num_slices),
        )
        devices = np.reshape(
            devices,
            tuple(x * y for x, y in zip(mesh_shape.ici_mesh_shape, mesh_shape.dcn_mesh_shape)),
        )
    else:
        raise ValueError(f"Unknown mesh_shape type: {type(mesh_shape)}")
    return devices, mesh_shape


def compile_trainer_programs(
    trainer_config: SpmdTrainer.Config,
    *,
    topology: str,
    topology_num_slices: int = 1,
    compiler_options: Optional[Dict[str, Union[str, bool]]] = None,
) -> Dict[str, jax.stages.Compiled]:
    """Returns compiled XLA programs for the given trainer.

    Args:
        trainer_config: The trainer config.
        topology: A string representing the TPU topology, e.g., "v4-8". Must be a key in
            USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
            If None, use CPU devices.
        topology_num_slices: The number of TPU slices.
        compiler_options: Options to pass to XLA. See `compiler_options.py` for examples.

    Returns:
        A dict containing the following programs:
        * "train_step": a program to run a single training step.

    Raises:
        NotImplementedError: if `topology` is not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
    """
    topology_devices, devices_per_slice = get_devices_for_topology(topology, topology_num_slices)

    cfg = trainer_config.clone(dir="NOT_USED")
    cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
    # Use a default mesh_shape if None or REQUIRED.
    cfg.mesh_shape = cfg.mesh_shape or [len(topology_devices)] + [1] * (
        len(cfg.mesh_axis_names) - 1
    )

    topology_devices, mesh_shape = reshape_devices(
        devices=topology_devices,
        mesh_shape=cfg.mesh_shape,
        devices_per_slice=devices_per_slice,
        num_slices=topology_num_slices,
    )
    cfg.mesh_shape = mesh_shape

    trainer: SpmdTrainer = cfg.instantiate(parent=None, devices=topology_devices)
    compiled_train_step = trainer.compile_train_step(compiler_options=compiler_options)
    return {"train_step": compiled_train_step}


def compile_inference_programs(
    inferencer_config: InferenceRunner.Config,
    *,
    input_batch_spec: Nested[TensorSpec],
    topology: str,
    topology_num_slices: int = 1,
    compiler_options: Optional[Dict[str, Union[str, bool]]] = None,
    method: str = "sample_decode",
) -> Dict[str, jax.stages.Compiled]:
    """Returns compiled XLA programs for the given inference runner.

    Args:
        inferencer_config: The inference runner config.
        input_batch_spec: A nested TensorSpec of input batch.
        topology: A string representing the TPU topology, e.g., "v4-8". Must be a key in
            USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
            If None, use CPU devices.
        topology_num_slices: The number of TPU slices.
        compiler_options: Options to pass to XLA. See `compiler_options.py` for examples.
        method: The method name to compile.

    Returns:
        A dict containing the following programs:
        * "sample_decode": a program to run a sample_decode loop.

    Raises:
        NotImplementedError: if `topology` is not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS.
    """
    topology_devices, devices_per_slice = get_devices_for_topology(topology, topology_num_slices)

    cfg = inferencer_config.clone()
    cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")

    # Use a default mesh_shape if None or REQUIRED.
    cfg.mesh_shape = cfg.mesh_shape or [len(topology_devices)] + [1] * (
        len(cfg.mesh_axis_names) - 1
    )

    topology_devices, mesh_shape = reshape_devices(
        devices=topology_devices,
        mesh_shape=cfg.mesh_shape,
        devices_per_slice=devices_per_slice,
        num_slices=topology_num_slices,
    )
    cfg.mesh_shape = mesh_shape

    inferencer: InferenceRunner = cfg.instantiate(
        parent=None,
        devices=topology_devices,
        inference_runner_state=True,
    )

    method_runner = inferencer.create_method_runner(method=method)

    with inferencer.mesh():
        jitted_fn = cast(
            partial, method_runner._jit_run_on_batch  # pylint: disable=protected-access
        )
        prng_key = jax.ShapeDtypeStruct(dtype=jnp.uint32, shape=[4])
        input_batch = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                shape=x.shape,
                dtype=x.dtype,
                sharding=x.sharding,
            ),
            input_batch_spec,
        )
        lowered = jitted_fn.func.lower(
            jitted_fn.args[0], prng_key, input_batch
        )  # pytype: disable=attribute-error
        compiled = lowered.compile(compiler_options=compiler_options)

    return {"sample_decode": compiled}
