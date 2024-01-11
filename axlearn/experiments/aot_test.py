# Copyright © 2023 Apple Inc.

"""AoT (ahead-of-time) compilation config tests.

E             RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to open /Users/rpang/miniforge3/envs/axlearn/lib/python3.9/site-packages/libtpu/libtpu.so: dlopen(/Users/rpang/miniforge3/envs/axlearn/lib/python3.9/site-packages/libtpu/libtpu.so, 0x0001): tried: '/Users/rpang/miniforge3/envs/axlearn/lib/python3.9/site-packages/libtpu/libtpu.so' (not a mach-o file), '/System/Volumes/Preboot/Cryptexes/OS/Users/rpang/miniforge3/envs/axlearn/lib/python3.9/site-packages/libtpu/libtpu.so' (no such file), '/Users/rpang/miniforge3/envs/axlearn/lib/python3.9/site-packages/libtpu/libtpu.so' (not a mach-o file) (set JAX_PLATFORMS='' to automatically choose an available backend)
"""
import os

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import copy
import tempfile
from typing import Optional

import jax

jax.config.update("jax_platforms", "cpu")
from dataclasses import dataclass

import jax.random
from absl import logging
from absl.testing import absltest
from jax.experimental.topologies import get_topology_desc

from axlearn.common import test_utils
from axlearn.common.checkpointer import every_n_steps_policy
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
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
        compile_topology: str,
        compile_topology_num_slices: int = 1,
    ):
        target_hardware = get_system_characteristics(compile_topology)
        topology_devices = get_topology_desc(
            platform=target_hardware.platform,
            topology_name=target_hardware.topology_name,
            chip_config_name=target_hardware.chip_config_name,
            chips_per_host_bounds=target_hardware.chips_per_host_bounds,
            num_slices=compile_topology_num_slices,
        ).devices

        with jax.checking_leaks(), set_data_dir("FAKE"):
            cfg = copy.deepcopy(trainer_config)
            cfg.dir = cfg.dir or tempfile.mkdtemp()
            cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
            cfg.mesh_shape = [len(topology_devices)] + [1] * (len(cfg.mesh_axis_names) - 1)
            topology_devices = np.reshape(topology_devices, cfg.mesh_shape)
            cfg.max_step = 3
            for evaler_cfg in cfg.evalers.values():
                if getattr(evaler_cfg.eval_policy, "fn", None) is eval_every_n_steps_policy:
                    evaler_cfg.eval_policy.n = 2
                evaler_cfg.vlog = max(evaler_cfg.vlog or 0, 3)
            if getattr(cfg.checkpointer.save_policy, "fn", None) is every_n_steps_policy:
                cfg.checkpointer.save_policy.n = 2
            logging.info("_test_with_trainer_config: %s", trainer_config)
            trainer: SpmdTrainer = cfg.instantiate(parent=None, devices=topology_devices)
            # trainer.init(jax.random.PRNGKey(1))
            input_batch_spec = self.input.dataset().element_spec
            compiled_train_step = trainer._jit_train_step.lower(
                trainer.trainer_state, input_batch_spec
            ).compile()
            print(compiled_train_step)

    def test_gpt_c4(self):
        self._test_aot(
            c4_trainer.named_trainer_configs()["fuji-test"](),
            compile_topology="v4-8",
        )


if __name__ == "__main__":
    absltest.main()
