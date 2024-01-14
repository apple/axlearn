# Copyright © 2023 Apple Inc.

"""AoT (ahead-of-time) compilation config tests.

pip install 'jax[tpu]==0.4.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

export TPU_SKIP_MDS_QUERY=1
python axlearn/experiments/aot_test.py

Reference:
https://docs.google.com/document/d/1Y5IdmvAZA7UtMHAWkRh8k2PscVoG5FvMH9-E6hygsyY/
"""
import os
import pickle

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import copy
from typing import Optional

import jax

jax.config.update("jax_platforms", "cpu")
from dataclasses import dataclass

import jax.random
from jax import numpy as jnp
from absl.testing import absltest
from jax.experimental.topologies import get_topology_desc
from jax.experimental.serialize_executable import serialize
import tensorflow as tf

from axlearn.common import test_utils
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import set_data_dir
from axlearn.experiments.text.gpt import c4_trainer


@dataclass
class SystemCharacteristics:
    platform: str
    topology_name: str
    chip_config_name: str  # 'megacore' or 'default'
    chips_per_host_bounds: tuple
    devices_per_slice: int


UserFacingNameToSystemCharacteristics = {
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


def get_system_characteristics(user_facing_name):
    return UserFacingNameToSystemCharacteristics.get(user_facing_name)


class AoTCompilationTest(test_utils.TrainerConfigTestCase):
    """Tests ahead-of-time (AoT) compilation."""

    def _jax_backend(self) -> Optional[str]:
        return "cpu"

    def _test_aot(
        self,
        trainer_config: SpmdTrainer.Config,
        *,
        compile_topology: Optional[str],
        compile_topology_num_slices: int = 1,
    ):
        if compile_topology is not None:
            target_hardware = get_system_characteristics(compile_topology)
            topology_devices = get_topology_desc(
                platform=target_hardware.platform,
                topology_name=target_hardware.topology_name,
                chip_config_name=target_hardware.chip_config_name,
                chips_per_host_bounds=target_hardware.chips_per_host_bounds,
                num_slices=compile_topology_num_slices,
            ).devices
        else:
            topology_devices = jax.devices(self._jax_backend())

        with jax.checking_leaks(), set_data_dir("FAKE"):
            cfg = copy.deepcopy(trainer_config)
            cfg.dir = "NOT_USED"
            cfg.mesh_shape = [len(topology_devices)] + [1] * (len(cfg.mesh_axis_names) - 1)
            topology_devices = np.reshape(topology_devices, cfg.mesh_shape)
            trainer: SpmdTrainer = cfg.instantiate(parent=None, devices=topology_devices)
            compiled_train_step = trainer.compile_train_step()
            self.assertIsNotNone(compiled_train_step)

            # Serialization does not work for CPU devices:
            #     UNIMPLEMENTED: Not an XLA Runtime executable
            if compile_topology is not None:
                serialized_compiled, in_tree, out_tree = serialize(compiled_train_step)
                with open("/tmp/aot_compiled", "wb") as f:
                    pickle.dump(serialized_compiled, f)
                print(serialized_compiled)

    def test_fuji_test(self):
        self._test_aot(
            c4_trainer.named_trainer_configs()["fuji-test"](),
            compile_topology=None,
            # compile_topology="v4-8",
        )


if __name__ == "__main__":
    absltest.main()
