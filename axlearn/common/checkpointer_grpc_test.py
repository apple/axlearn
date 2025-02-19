# Copyright © 2025 Apple Inc.

"""Tests for gRPC server support for array serialization."""

import multiprocessing as mp
import os
from unittest import mock

import jax
import jax.distributed
import jax.numpy as jnp
import numpy as np
from jax._src.distributed import global_state

from axlearn.common.checkpointer_grpc import (
    GRPCTensorStoreStateStorage,
    TensorstoreGrpcClient,
    _check_kvstore_server_binary,
    build_step_dir,
    default_kvstore_server,
    multihost_utils,
)
from axlearn.common.test_utils import TestCase


def worker(
    array_specs: dict[str, tuple[int, ...]],
    coor_addr,
    process_id,
    num_processes,
    primary_server_addr,
    local_server_addr,
    signal_q,
):
    jax.distributed.initialize(
        coordinator_address=coor_addr, process_id=process_id, num_processes=num_processes
    )
    mesh = jax.make_mesh((num_processes,), axis_names=("model",), devices=jax.devices())
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model"))

    for key, shape in list(array_specs.items()):
        array_specs[key] = jax.make_array_from_process_local_data(
            sharding,
            np.ones(sharding.shard_shape(shape)),
            shape,
        )

    # Patch the sync function to use the jax coordinator, because sync through ICI is not supported
    # on CPU.
    def patch_sync_fn(key):
        global_state.client.wait_at_barrier(key, 30 * 1000)

    with mesh, mock.patch.object(multihost_utils, "sync_global_devices", patch_sync_fn):
        ckpt = (
            GRPCTensorStoreStateStorage.default_config()
            .set(
                local_server_addr=local_server_addr,
                local_server_builder=default_kvstore_server,
            )
            .instantiate()
        )
        ckpt.save_to_dir(
            step=0,
            state=array_specs,
            ckpt_dir=f"grpc://{primary_server_addr}/{build_step_dir('', step=0)}",
        )
        ckpt.wait_until_finished()

    signal_q.get()
    ckpt.stop()


class GRPCTest(TestCase):
    def test_multihost(self):
        try:
            _check_kvstore_server_binary()
        except ValueError:
            self.skipTest("Requires kvstore_server_main in PATH for this test.")
        num_processes = 4
        context = mp.get_context("spawn")
        procs = []
        signal_q = mp.Queue(1)

        array_specs = {
            "weight/x1": (16, 128),
            "weight/x2": (128, 16),
        }
        primary_server_addr = "localhost:9833"

        for i in range(num_processes):
            local_server_addr = f"localhost:{9833 + i}"
            p = context.Process(
                target=worker,
                kwargs=dict(
                    array_specs=array_specs,
                    coor_addr="127.0.0.1:1023",
                    process_id=i,
                    num_processes=num_processes,
                    primary_server_addr=primary_server_addr,
                    local_server_addr=local_server_addr,
                    signal_q=signal_q,
                ),
                daemon=True,
            )
            procs.append(p)
        try:
            for p in procs:
                p.start()

            client = TensorstoreGrpcClient()

            # Tests resharding with storage.
            mesh = jax.make_mesh((1,), axis_names=("model",), devices=jax.devices())
            with mesh:
                store = GRPCTensorStoreStateStorage.default_config().instantiate()

                arrays = {}
                for key, shape in list(array_specs.items()):
                    arrays[key] = jnp.zeros(shape)

                ground_truth = {}
                for key, shape in list(array_specs.items()):
                    ground_truth[key] = jnp.ones(shape)

                step_dir = build_step_dir("", step=0)
                ckpt_dir = f"grpc://{primary_server_addr}/{step_dir}"
                # pylint: disable=protected-access
                client.wait_for_exists(path=os.path.join(ckpt_dir, "index"))
                self.assertIn(
                    step_dir,
                    client.list_checkpoints(
                        f"grpc://{primary_server_addr}/", include_not_committed=False
                    ),
                )

                arr = store.restore_from_dir(0, arrays, ckpt_dir=ckpt_dir)
                self.assertNestedEqual(arr, ground_truth)

        finally:
            # Cleanup.
            for i in range(num_processes):
                signal_q.put(None)
            for p in procs:
                p.join()
            signal_q.close()
            signal_q.join_thread()
