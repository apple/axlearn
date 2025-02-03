# Copyright © 2024 Apple Inc.

"""Tests orbax emergency checkpointer."""

# pylint: disable=protected-access

import multiprocessing as mp
import os
import socket
import tempfile
from contextlib import ExitStack, closing
from typing import Optional

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.common import utils_spmd
from axlearn.common.checkpointer_orbax_emergency import (
    OrbaxEmergencyCheckpointer,
    _dump_process_info,
    _get_previous_process_info,
    _init_consistent_proc_ids,
    _logger_init,
    _ProcessInfo,
    config_for_function,
    every_n_steps_policy,
    get_consistent_proc_info,
)


def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _test_orbax_main(process_id: int, port: int, persist_dir: str, local_dir: str):
    _logger_init()
    # pylint: disable=import-outside-toplevel
    from orbax.checkpoint._src.multihost import multislice
    from orbax.checkpoint.experimental.emergency import checkpoint_manager

    # Patch for GPU use. We don't need to use mock.patch because we're running in a subprocess.
    multislice.get_device_memory = lambda: int(80e9)

    def slice_devices(
        global_mesh: jax.sharding.Mesh,
        *,
        replica_id: int = 0,
        replica_axis_index: int = 0,
    ) -> np.ndarray:
        return np.take(
            global_mesh.devices,
            replica_id,
            axis=replica_axis_index,
        )

    def _all_devices_excepting_slice(
        devices: np.ndarray,
        *,
        replica_id: int = 0,
        replica_axis_index: int = 0,
    ) -> np.ndarray:
        return np.delete(devices, replica_id, axis=replica_axis_index)

    # We're not running in a true multi-slice environment. Patch the following two functions to
    # mock multi-slice discovery.
    multislice.slice_devices = slice_devices
    checkpoint_manager._all_devices_excepting_slice = _all_devices_excepting_slice

    proc_info = get_consistent_proc_info(
        distributed_coordinator=f"127.0.0.1:{port}",
        local_address=f"127.0.0.1:{port}",
        num_processes=4,
        process_id=process_id,
        trainer_dir=persist_dir,
        local_ckpt_dir=local_dir,
        jax_backend="gpu",
    )

    jax.distributed.initialize(
        coordinator_address=proc_info.address,
        num_processes=4,
        process_id=proc_info.inv_proc_id,
        local_device_ids=[process_id],
    )

    cfg: OrbaxEmergencyCheckpointer.Config = OrbaxEmergencyCheckpointer.default_config()
    cfg.name = "emergency"
    cfg.save_policy = config_for_function(every_n_steps_policy).set(n=25)
    cfg.local_save_policy = config_for_function(every_n_steps_policy).set(n=5)
    # Local checkpoint path suffix must be the same for orbax synchronization to work.
    cfg.local_dir = local_dir
    cfg.trainer_dir = persist_dir
    cfg.dir = persist_dir
    cfg.keep_last_n = 2
    cfg.local_keep_last_n = 2
    cfg.replica_axis_index = 0
    cfg.async_timeout_secs = 5
    cfg.non_tensor_async_timeout_secs = 5
    checkpointer: OrbaxEmergencyCheckpointer = cfg.instantiate(parent=None)
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2, 2), ["data", "model"])
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
    x = jax.make_array_from_process_local_data(sharding, np.zeros((8, 8), dtype=np.float32), (8, 8))

    input_iter = iter(tf.data.Dataset.counter())
    state = {"x": x, "y": None, "z": input_iter}
    with mesh:
        step, state = checkpointer.restore(step=None, state=state)
        first_run = step is None
        if step is None:
            step = 0
        else:
            # This is the step of the last persistent ckpt.
            assert checkpointer.latest_checkpoint_step(checkpointer.config.dir) == 25
            # This step should already be garbage collected.
            assert 20 not in checkpointer._tensor_manager.all_steps()
            # Step 40 is incomplete (it has the index file removed manually).
            assert 40 not in checkpointer._checkpoint_steps_include_local()
            # Although the persistent save interval is 25 steps, local save interval is 5
            # steps, so we should be able to restore step 40.
            assert step == 35, step
            # Since we save after step X is finished, after restore the first step is X + 1.
            step += 1
        for i in range(step, 100):
            assert jnp.all(state["x"] == i).item()
            assert i == next(input_iter)  # input_iter's state is modified inplace.
            state = {"x": state["x"] + 1, "y": state["y"], "z": state["z"]}
            checkpointer.save(step=i, state=state)
            if process_id == 0:
                logging.info("step %d", i)
            if i == 44 and first_run:
                break

        checkpointer.wait_until_finished()
    jax.distributed.shutdown()


def _test_init_proc_id_main(
    *,
    distributed_coordinator: str,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
    trainer_dir: str,
    local_ckpt_dir: str,
    proc_per_slice: int,
    new_idx_map: dict[int, int],
):
    _logger_init()
    # Fake some envs.
    os.environ["MEGASCALE_NUM_SLICES"] = str(num_processes // proc_per_slice)
    os.environ["MEGASCALE_SLICE_ID"] = f"{process_id // proc_per_slice}"
    os.environ["TPU_WORKER_ID"] = str(process_id % proc_per_slice)
    os.environ["TPU_WORKER_HOSTNAMES"] = ",".join(
        [distributed_coordinator.split(":")[0]] * proc_per_slice
    )

    if new_idx_map[process_id] != -1:
        _dump_process_info(
            local_ckpt_dir,
            trainer_dir=trainer_dir,
            proc_info=_ProcessInfo(distributed_coordinator, new_idx_map[process_id], process_id),
        )

    prev_setup = utils_spmd.setup

    def patch_setup(**kwargs):
        kwargs["jax_backend"] = "gpu"
        return prev_setup(**kwargs)

    # Patch setup so that we don't get an error for passing process id.
    utils_spmd.setup = patch_setup

    _init_consistent_proc_ids(
        jax_backend="tpu",
        distributed_coordinator=distributed_coordinator,
        num_processes=num_processes,
        process_id=process_id,
        trainer_dir=trainer_dir,
        local_ckpt_dir=local_ckpt_dir,
        barrier_timeout_seconds=30,
    )


class OrbaxCheckpointerTest(parameterized.TestCase):
    def test_init_proc_id_tpu(self):
        # Tests the process id restore logic for TPU. Note that the logic for GPU is tested in
        # `test_emergency_ckpt`.
        new_idx_map = {
            # First two slices are healthy, but have different slice id during restart.
            0: 2,
            1: 3,
            2: 6,
            3: 7,
            4: -1,  # This failed slice has one node swapped out.
            5: 1,
            6: -1,  # This failed slice has two nodes swapped out.
            7: -1,
        }
        with ExitStack() as stack:
            num_processes = 8
            local_tempdirs = [
                stack.enter_context(tempfile.TemporaryDirectory()) for _ in range(num_processes)
            ]
            processes = []
            # Use spawn to not inherit the already-initialized jax backend in the subprocess.
            context = mp.get_context("spawn")
            for i in range(num_processes):
                proc = context.Process(
                    target=_test_init_proc_id_main,
                    kwargs=dict(
                        distributed_coordinator="127.0.0.1:8476",
                        num_processes=num_processes,
                        process_id=i,
                        trainer_dir="any",
                        local_ckpt_dir=local_tempdirs[i],
                        proc_per_slice=2,
                        new_idx_map=new_idx_map,
                    ),
                )
                proc.start()
                processes.append(proc)

            for p in processes:
                p.join()
                self.assertEqual(p.exitcode, 0)

            infos = [
                _get_previous_process_info(local_dir, trainer_dir="any")
                for local_dir in local_tempdirs
            ]
            for info in infos:
                assert info.address == "127.0.0.1:8476"
            new_proc_ids = [info.inv_proc_id for info in infos]
            for i in range(4):
                self.assertEqual(new_proc_ids[i], new_idx_map[i])

            if new_proc_ids[4] == 0:
                self.assertEqual(new_proc_ids[5], 1)
                self.assertEqual(new_proc_ids[6], 4)
                self.assertEqual(new_proc_ids[7], 5)
            elif new_proc_ids[4] == 4:
                self.assertEqual(new_proc_ids[5], 5)
                self.assertEqual(new_proc_ids[6], 0)
                self.assertEqual(new_proc_ids[7], 1)
            else:
                self.fail("New proc id of proc 4 should be either 0 or 4!.")

    def test_emergency_ckpt(self):
        if jax.device_count() < 4:
            self.skipTest("Need at least 4 devices for this test.")
        with ExitStack() as stack:
            num_processes = 4
            local_tempdirs = [
                stack.enter_context(tempfile.TemporaryDirectory()) for _ in range(num_processes)
            ]
            persistent_tempdir = stack.enter_context(tempfile.TemporaryDirectory())
            # Use spawn to not inherit the already-initialized jax backend in the subprocess.
            context = mp.get_context("spawn")

            def start_processes(reverse_process_id: bool = False):
                free_port = _find_free_port()
                processes = []
                for i in range(num_processes):
                    p = context.Process(
                        target=_test_orbax_main,
                        args=(
                            i if not reverse_process_id else num_processes - i - 1,
                            free_port,
                            persistent_tempdir,
                            local_tempdirs[i],
                        ),
                    )
                    processes.append(p)
                    p.start()
                return processes

            processes = start_processes()
            for p in processes:
                p.join()
                self.assertEqual(p.exitcode, 0)

            # Remove step 40 index file to simulate incomplete checkpoints.
            os.remove(os.path.join(persistent_tempdir, "non-tensors", "step_00000040", "index"))

            # Shuffle the process ids to verify that we are able to restore the process id.
            processes = start_processes(reverse_process_id=True)

            try:
                for p in processes:
                    p.join()
                for p in processes:
                    self.assertEqual(p.exitcode, 0)
            finally:
                for p in processes:
                    p.kill()
