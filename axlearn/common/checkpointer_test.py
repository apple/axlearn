# Copyright © 2023 Apple Inc.

"""Tests checkpointing utilities.

Some tests are intended to be run on TPU.
"""

import asyncio
import os
import queue
import re
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Optional, Type, cast
from unittest import mock

import jax
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.array_serialization import serialization as array_serialization

from axlearn.common import array_serialization as axlearn_serialization
from axlearn.common import file_system as fs
from axlearn.common import serialization, test_utils, utils
from axlearn.common.array_serialization import BoundedDataShardedAsyncCheckpointManager
from axlearn.common.checkpointer import (
    BaseCheckpointer,
    BestMetricPolicy,
    Checkpointer,
    CheckpointValidationType,
    EvalMetric,
    PythonSavable,
    TensorStoreStateStorage,
    async_save_tf_savables,
    check_state_structure,
    every_n_steps_and_last_policy,
    every_n_steps_policy,
    read_state_spec,
    restore_tf_savables,
)
from axlearn.common.checkpointer_orbax import _GRAIN_INSTALLED, OrbaxCheckpointer
from axlearn.common.file_system import listdir
from axlearn.common.metrics import WeightedScalar
from axlearn.common.summary_writer import SummaryWriter
from axlearn.common.utils import VDict

# Conditionally import to allow running tests on Apple silicon where grain is not supported.
if _GRAIN_INSTALLED:
    from axlearn.common.input_grain_test import range_dataset


def _mesh(mesh_shape: Sequence[int]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ("data", "model"))


def _checkpointer_config(
    checkpointer_cls: Type[BaseCheckpointer] = Checkpointer,
) -> BaseCheckpointer.Config:
    # TODO(markblee): Use context manager instead of mkdtemp.
    return checkpointer_cls.default_config().set(
        name="test",
        dir=tempfile.mkdtemp(),
        keep_last_n=1,
    )


# pylint: disable=protected-access


class CheckpointerTest(test_utils.TestCase):
    @parameterized.parameters(Checkpointer, OrbaxCheckpointer)
    def test_save_and_restore(self, checkpointer_cls: Type[BaseCheckpointer]):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            cfg.save_policy.min_step = 0
            ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32))
            state1 = dict(x=jnp.ones([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32) + 1)

            # Restoring from an empty dir returns the input state if step=None.
            self.assertNestedEqual((None, state0), ckpt.restore(step=None, state=state0))
            self.assertNestedEqual((None, state1), ckpt.restore(step=None, state=state1))
            # With an explicit step, ValueError will be raised.
            with self.assertRaises(ValueError):
                ckpt.restore(step=0, state=state0)

            ckpt.save(step=0, state=state0)
            ckpt.wait_until_finished()
            self.assertNestedEqual((0, state0), ckpt.restore(step=0, state=state1))
            # step=None restores from the latest ckpt.
            self.assertNestedEqual((0, state0), ckpt.restore(step=None, state=state1))

            ckpt.save(step=1, state=state1)
            ckpt.wait_until_finished()
            self.assertNestedEqual((1, state1), ckpt.restore(step=1, state=state0))
            # step=None restores from the latest ckpt.
            self.assertNestedEqual((1, state1), ckpt.restore(step=None, state=state0))

            # When the given state has a different dict key: 'z' instead of 'y'.
            with self.assertRaisesRegex((ValueError, KeyError), "z"):
                ckpt.restore(
                    step=None,
                    state=dict(
                        x=jnp.zeros([], dtype=jnp.int32), z=jnp.ones([2], dtype=jnp.float32)
                    ),
                )

            # When the given state has a different array shape: [3] instead of [2] for y.
            with self.assertRaisesRegex(
                ValueError, "(checkpoint tree dtypes or shapes|not compatible)"
            ):
                ckpt.restore(
                    step=None,
                    state=dict(
                        x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([3], dtype=jnp.float32)
                    ),
                )
            # TODO(matthew_e_hopkins): revert it once upgrade jax version.
            if checkpointer_cls is Checkpointer:
                # When the given state has a different dict shape: [1] instead of [] for x.
                # Orbax throws AssertionError in this case.
                with self.assertRaisesRegex(
                    (AssertionError, ValueError),
                    "(checkpoint tree dtypes or shapes|not compatible)",
                ):
                    ckpt.restore(
                        step=None,
                        state=dict(
                            x=jnp.zeros([1], dtype=jnp.int32),
                            y=jnp.ones([2], dtype=jnp.float32),
                        ),
                    )
            else:
                with self.assertRaisesRegex(
                    (AssertionError, ValueError),
                    (
                        "Cannot intersect index domain|not compatible|"
                        "Requested shape.*is not compatible with the stored shape"
                    ),
                ):
                    ckpt.restore(
                        step=None,
                        state=dict(
                            x=jnp.zeros([1], dtype=jnp.int32),
                            y=jnp.ones([2], dtype=jnp.float32),
                        ),
                    )

            # When the given state has a different dtype: float32 instead of int32 for x.
            with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
                ckpt.restore(
                    step=None,
                    state=dict(
                        x=jnp.zeros([], dtype=jnp.float32),
                        y=jnp.ones([2], dtype=jnp.float32),
                    ),
                )
            ckpt.stop()

    @parameterized.parameters(Checkpointer, OrbaxCheckpointer)
    def test_save_and_restore_mesh(self, checkpointer_cls: Type[BaseCheckpointer]):
        """Tests that we can save with one sharding and restore with a different sharding."""
        mesh_shape = (4, 2)
        restore_mesh_shape = (2, 4)
        if not test_utils.is_supported_mesh_shape(
            mesh_shape
        ) or not test_utils.is_supported_mesh_shape(restore_mesh_shape):
            return

        cfg = _checkpointer_config(checkpointer_cls)
        ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
        state = dict(
            x=jax.random.uniform(jax.random.PRNGKey(123), shape=[8, 4], dtype=jnp.float32),
        )
        step = 1

        def state_specs(state, partition_spec):
            return jax.tree.map(
                lambda x: utils.TensorSpec(shape=x.shape, dtype=x.dtype, mesh_axes=partition_spec),
                state,
            )

        with _mesh(mesh_shape) as mesh:
            # Shard across both axes.
            sharding = jax.sharding.NamedSharding(
                mesh, spec=jax.sharding.PartitionSpec("data", "model")
            )
            state = jax.tree.map(lambda x: jax.device_put(x, device=sharding), state)
            ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            # Shard only across axis=0.
            partition_spec = jax.sharding.PartitionSpec(("data", "model"), None)
            restored_step, restored_state = ckpt.restore(
                step=step, state=state_specs(state, partition_spec)
            )
            self.assertEqual(step, restored_step)
            self.assertNestedEqual(state, restored_state)

        # Restore under a different mesh.
        with _mesh(restore_mesh_shape) as mesh:
            for partition_spec in [
                jax.sharding.PartitionSpec("data", "model"),
                jax.sharding.PartitionSpec(("data", "model"), None),
            ]:
                restored_step, restored_state = ckpt.restore(
                    step=step, state=state_specs(state, partition_spec)
                )
                self.assertEqual(step, restored_step)
                self.assertNestedEqual(state, restored_state)

    @parameterized.parameters(
        # Number of files minus index and .zarray metadata.
        dict(
            max_data_shard_degree=None,
            shard_threshold_bytes=None,
            num_files=4,  # 2 ararys * 2 shards (2 model) per array.
        ),
        dict(
            max_data_shard_degree=-1,
            shard_threshold_bytes=None,
            num_files=16,  # 2 ararys * 8 shards (2 model, 4 data) per array.
        ),
        dict(
            max_data_shard_degree=2,
            shard_threshold_bytes=None,
            num_files=8,  # 2 ararys * 4 shards (2 model, 2 data) per array.
        ),
        dict(
            max_data_shard_degree=2,
            shard_threshold_bytes=1024,
            num_files=6,  # 1 array 4 shards (2 model, 2 data) + 1 array 2 shards (small array).
        ),
    )
    def test_save_restore_files_count(
        self, max_data_shard_degree: int, shard_threshold_bytes: int, num_files: int
    ):
        # Tests the effect of max_data_shard_degree and shard_threshold_bytes on number of files.
        mesh_shape = (4, 2)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        cfg: Checkpointer.Config = _checkpointer_config(Checkpointer)
        cfg.storage.max_data_shard_degree = max_data_shard_degree
        cfg.storage.shard_threshold_bytes = shard_threshold_bytes
        ckpt: Checkpointer = cfg.instantiate(parent=None)
        state = dict(
            x=jnp.zeros((1024, 1024), dtype=jnp.float32),
            small_x=jnp.zeros((16, 16), dtype=jnp.float32),
        )
        step = 1

        def count_files(directory):
            file_count = 0
            for _, _, files in os.walk(directory):
                file_count += len(files)
            return file_count

        def state_specs(state):
            return jax.tree.map(
                lambda x: utils.TensorSpec(
                    shape=x.shape,
                    dtype=x.dtype,
                    mesh_axes=jax.sharding.PartitionSpec(None, "model"),
                ),
                state,
            )

        with _mesh(mesh_shape) as mesh:
            sharding = jax.sharding.NamedSharding(
                mesh, spec=jax.sharding.PartitionSpec(None, "model")
            )
            state = jax.tree.map(lambda x: jax.device_put(x, device=sharding), state)
            ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            restored_step, restored_state = ckpt.restore(step=step, state=state_specs(state))
            self.assertEqual(step, restored_step)
            self.assertNestedEqual(state, restored_state)

            self.assertEqual(count_files(ckpt.ckpt_dir(step)), num_files + 3)

    @parameterized.parameters(Checkpointer, OrbaxCheckpointer)
    def test_save_and_restore_latest_valid(self, checkpointer_cls: Type[BaseCheckpointer]):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32))

            # Restoring from an empty dir returns the input state if step=None.
            self.assertNestedEqual((None, state0), ckpt.restore(state=state0))

            def create_corrupt_ckpt(step):
                if checkpointer_cls is Checkpointer:
                    ckpt_dir = cast(Checkpointer, ckpt).ckpt_dir(step)
                elif checkpointer_cls is OrbaxCheckpointer:
                    ckpt_dir = os.path.join(
                        cast(OrbaxCheckpointer, ckpt)._manager.directory,
                        f"{ocp.utils.TMP_DIR_SUFFIX}{step}",
                    )
                else:
                    raise NotImplementedError(checkpointer_cls)
                fs.makedirs(ckpt_dir)

                if checkpointer_cls is OrbaxCheckpointer:
                    assert not ocp.utils.is_checkpoint_finalized(ckpt_dir)

            # Test that we return the same state if no checkpoints valid.
            create_corrupt_ckpt(step=0)
            self.assertNestedEqual(
                (None, state0),
                ckpt.restore(step=None, state=state0),
            )

            # Test that we restore from an earlier valid state if latest state load fails.
            ckpt.save(step=2, state=state0)
            ckpt.wait_until_finished()
            create_corrupt_ckpt(step=3)
            self.assertNestedEqual((2, state0), ckpt.restore(step=None, state=state0))

            # Attempting to save at the corrupted step 3 should succeed.
            ckpt.save(step=3, state=state0)
            ckpt.wait_until_finished()
            self.assertNestedEqual((3, state0), ckpt.restore(step=None, state=state0))

    @parameterized.parameters(Checkpointer, OrbaxCheckpointer)
    def test_save_can_override_on_gcs(self, checkpointer_cls: Type[BaseCheckpointer]):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        # Patch is_gcs_path for orbax, since it commits differently on gcs vs local.
        with _mesh(mesh_shape), mock.patch(f"{ocp.utils.__name__}.is_gcs_path", return_value=True):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32))

            # Save a checkpoint.
            ckpt.save(step=1, state=state0)
            ckpt.wait_until_finished()
            self.assertNestedEqual((1, state0), ckpt.restore(step=None, state=state0))

            if isinstance(ckpt, (Checkpointer, OrbaxCheckpointer)):
                ckpt_dir = ckpt.ckpt_dir(step=1)
            else:
                raise NotImplementedError(type(ckpt))

            # Corrupt the checkpoint by removing some files, while ensuring it is non-empty.
            if isinstance(ckpt, Checkpointer):
                commit_file = "index"
                fs.rmtree(os.path.join(ckpt_dir, commit_file))
                self.assertGreater(len(fs.listdir(ckpt_dir)), 0)
            else:
                tmp_dir = f"{ckpt_dir}{ocp.utils.TMP_DIR_SUFFIX}"
                os.rename(ckpt_dir, tmp_dir)
                self.assertTrue(fs.exists(tmp_dir))

            if isinstance(ckpt, OrbaxCheckpointer):
                ckpt._manager.reload()  # Orbax caches complete checkpoints.

            self.assertEqual(0, len(ckpt.checkpoint_paths(ckpt.config.dir)))

            # Test that save() should be able to override non-empty ckpt dir.
            state1 = dict(x=jnp.ones([], dtype=jnp.int32), y=jnp.zeros([2], dtype=jnp.float32))
            ckpt.save(step=1, state=state1)
            ckpt.wait_until_finished()

            # Should match the new state.
            self.assertNestedEqual((1, state1), ckpt.restore(step=None, state=state1))

    @parameterized.product(
        checkpointer_cls=[Checkpointer, OrbaxCheckpointer],
        mesh_shape=[(1, 1), (2, 2), (4, 2)],
    )
    def test_gda(self, checkpointer_cls, mesh_shape):
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state = dict(x=jnp.arange(16).reshape((4, 4)))
            ckpt.save(step=10, state=state)
            ckpt.wait_until_finished()

            state0 = dict(x=jnp.zeros(shape=(4, 4), dtype=jnp.int32))
            step, restored_state = ckpt.restore(
                step=None,
                state=state0,
            )
            self.assertEqual(10, step)
            self.assertNestedEqual(restored_state, state)

            # dtype mismatch.
            with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
                ckpt.restore(
                    step=None,
                    state=dict(x=jnp.zeros(shape=(4, 4), dtype=jnp.float32)),
                )
            ckpt.stop()

    @parameterized.product(
        checkpointer_cls=[Checkpointer, OrbaxCheckpointer],
        custom_dict_type=(utils.VDict,),
    )
    def test_custom_dict(self, checkpointer_cls, custom_dict_type):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state0 = custom_dict_type(
                x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32)
            )

            ckpt.save(step=100, state=state0)
            ckpt.wait_until_finished()

            # Restore with state structure hints will preserve VDict.
            step, restored_state = ckpt.restore(step=None, state=state0)
            self.assertEqual(100, step)
            self.assertEqual(type(restored_state), custom_dict_type)
            self.assertIn(
                custom_dict_type.__name__, str(jax.tree_util.tree_structure(restored_state))
            )
            self.assertNestedEqual(state0, restored_state)
            ckpt.stop()

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_input_iterator(self, checkpointer_cls):
        self.skipTest("TODO(mark-b-lee): figure out why it fails on CI.")
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            input_iter = iter(tf.data.Dataset.from_tensor_slices([1, 2, 3]))
            # Move the input_iter.
            self.assertEqual(next(input_iter), 1)
            state0 = dict(
                x=jnp.zeros([], dtype=jnp.int32),
                input_iter=input_iter,
            )

            self.assertEqual([], os.listdir(cfg.dir))

            ckpt.save(step=100, state=state0)
            ckpt.wait_until_finished()

            # Check that input iterators are saved under a per-worker path.
            # E.g., /path/to/<step>/[state/]tf_0/input_iter.index.
            state_dir = ckpt.ckpt_dir(100)
            if "state" in os.listdir(state_dir):
                state_dir = os.path.join(state_dir, "state")
            self.assertIn("tf_0", os.listdir(state_dir))

            state0_specs = dict(
                x=utils.TensorSpec(shape=[], dtype=jnp.int32),
                # The same iterator, but with the position at 0.
                input_iter=iter(tf.data.Dataset.from_tensor_slices([1, 2, 3])),
            )

            def tensors_only(tree):
                return (
                    utils.prune_tree(
                        tree, should_prune=lambda _, v: not isinstance(v, utils.Tensor)
                    ),
                )

            step, restored_state = ckpt.restore(step=None, state=state0_specs)
            self.assertEqual(100, step)
            # The iterators will be different (despite pointing to the same values).
            self.assertNestedEqual(tensors_only(state0), tensors_only(restored_state))
            # The restored_state contains the input_iter pointing to the next value.
            self.assertEqual(next(restored_state["input_iter"]), 2)
            self.assertEqual(next(restored_state["input_iter"]), 3)
            ckpt.stop()

    def test_python_savable(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        class _DummySavable:
            """A dummy class implementing PythonSavable."""

            def __init__(self, max_step: int):
                self._max_step = max_step
                self._state = dict(step=0, values=b"123")

            def get_state(self):
                return self._state

            def set_state(self, state):
                self._state = state

            def __iter__(self):
                for i in range(self._state["step"], self._max_step):
                    self._state["step"] += 1
                    yield i

        with _mesh(mesh_shape):
            cfg = _checkpointer_config(Checkpointer)
            ckpt: Checkpointer = cfg.instantiate(parent=None)

            x = _DummySavable(max_step=3)
            # Check that runtime_checks is enabled.
            self.assertIsInstance(x, PythonSavable)

            # Iterate once and save.
            self.assertEqual(next(iter(x)), 0)
            state0 = dict(x=x)

            self.assertEqual([], os.listdir(cfg.dir))
            ckpt.save(step=1, state=state0)
            ckpt.wait_until_finished()

            # Check that input iterators are saved under a per-worker path.
            # E.g., /path/to/<step>/[state/]python_0/x.
            state_dir = ckpt.ckpt_dir(1)
            if "state" in os.listdir(state_dir):
                state_dir = os.path.join(state_dir, "state")
            self.assertIn("python_0", os.listdir(state_dir))
            self.assertIn("x", os.listdir(os.path.join(state_dir, "python_0")))

            # Construct a new savable object to restore.
            state1 = dict(x=_DummySavable(max_step=3))
            step, restored_state = ckpt.restore(step=None, state=state1)
            self.assertEqual(1, step)
            restored_x: _DummySavable = restored_state["x"]
            # The restored_state contains the iter pointing to the next value.
            self.assertEqual(list(range(1, 3)), list(iter(restored_x)))
            self.assertEqual(b"123", restored_x.get_state()["values"])
            ckpt.stop()

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_grain(self, checkpointer_cls):
        self.skipTest("TODO(mark-b-lee): figure out why it fails on CI.")
        if not _GRAIN_INSTALLED:
            self.skipTest("Cannot run when grain is not installed.")
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            ds = iter(range_dataset(start=1, stop=4))
            # Move the input_iter.
            self.assertEqual(next(ds), 1)
            state0 = dict(x=jnp.ones([3, 2]), y=ds)

            self.assertEqual([], os.listdir(cfg.dir))

            ckpt.save(step=100, state=state0)
            ckpt.wait_until_finished()

            # Check that input iterators are saved under a per-worker path.
            # E.g., /path/to/<step>/[state/]python_0/input_iter.index.
            state_dir = ckpt.ckpt_dir(100)
            if "state" in os.listdir(state_dir):
                state_dir = os.path.join(state_dir, "state")
            self.assertIn("python_0", os.listdir(state_dir))

            state0_specs = dict(
                x=utils.TensorSpec(shape=[3, 2], dtype=jnp.float32),
                # The same iterator, but with the position at 0.
                y=iter(range_dataset(start=0, stop=4)),
            )

            def tensors_only(tree):
                return (
                    utils.prune_tree(
                        tree, should_prune=lambda _, v: not isinstance(v, utils.Tensor)
                    ),
                )

            step, restored_state = ckpt.restore(step=None, state=state0_specs)
            self.assertEqual(100, step)
            # The iterators will be different (despite pointing to the same values).
            self.assertNestedEqual(tensors_only(state0), tensors_only(restored_state))
            # The restored_state contains the input_iter pointing to the next value.
            self.assertEqual(list(range(1, 4)), list(restored_state["y"]))
            ckpt.stop()

    def test_cleanup_checkpoint(self):
        # Mock the rmtree s.t. it does nothing.
        with (
            mock.patch("axlearn.common.file_system.rmtree", side_effect=None),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            # Create a few mock checkpoints.
            ckpt_paths = []
            for step in [1, 2]:
                ckpt_paths.append(os.path.join(temp_dir, f"step_{step:08d}"))
                os.makedirs(ckpt_paths[-1])
                for file in ["test", "index"]:
                    with open(os.path.join(ckpt_paths[-1], file), "w", encoding="utf-8") as f:
                        f.write(str(step))
            self.assertEqual(Checkpointer.latest_checkpoint_path(temp_dir), ckpt_paths[-1])
            # Simulate a corrupted cleanup on the last ckpt.
            Checkpointer.cleanup_checkpoint(ckpt_paths[-1], sync=False)
            # Ensure that the last ckpt still has the "test" file.
            with open(os.path.join(ckpt_paths[-1], "test"), encoding="utf-8") as f:
                self.assertEqual("2", f.read())
            # Ensure that the last ckpt is considered invalid.
            self.assertEqual(Checkpointer.latest_checkpoint_path(temp_dir), ckpt_paths[0])

    @parameterized.product(
        # By default, we restore from the latest ckpt, keep the last 3 steps, and every 2 after.
        (
            dict(
                ckpt_paths=None,
                expect_restore_step=9,
                expect_saved_steps=[0, 2, 4, 6, 7, 8, 9],
            ),
            # If we pretend that the first 2 ckpt paths failed, they should be retained.
            # We then keep the next 3 steps and every 2 afterwards.
            dict(
                ckpt_paths=range(8),
                expect_restore_step=7,
                expect_saved_steps=[0, 2, 4, 5, 6, 7, 8, 9],
            ),
            # Test a case where committed dirs are not consecutive.
            # In this case, we keep 9 due to possible in-progress write;
            # we keep 8, 6, 3, which are the last 3 checkpoints;
            # And we keep 0 (but not 1, since it doesn't respect keep_every_n=2).
            dict(
                ckpt_paths=[0, 1, 3, 6, 8],
                expect_restore_step=8,
                expect_saved_steps=[0, 3, 6, 8, 9],
            ),
        ),
        listdir_add_trailing_slash=[True, False],
    )
    def test_garbage_collection(
        self,
        ckpt_paths: Optional[Sequence[str]],
        expect_restore_step: int,
        expect_saved_steps: Sequence[int],
        listdir_add_trailing_slash: bool,
    ):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        orig_listdir = listdir

        def patch_tf_io_behavior(*args):
            out = orig_listdir(*args)
            return [x + "/" for x in out if not x.endswith("/")]

        # pylint: disable=line-too-long
        with (
            _mesh(mesh_shape),
            mock.patch("axlearn.common.file_system.listdir", patch_tf_io_behavior)
            if listdir_add_trailing_slash
            else nullcontext(),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            cfg = Checkpointer.default_config().set(
                name="test",
                dir=temp_dir,
                keep_last_n=3,
                keep_every_n_steps=2,
                gc_loop_interval_seconds=1,
            )
            cfg.save_policy.min_step = 0

            # Running gc for non-existent dir shouldn't fail.
            ckpt_fake = cfg.clone(dir=os.path.join(temp_dir, "fake_dir")).instantiate(parent=None)
            ckpt_fake._run_garbage_collection()

            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state = dict(x=jnp.zeros([], dtype=jnp.int32))

            for step in range(10):
                ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            # Mock out the checkpoints that are committed.
            if ckpt_paths:
                ckpt_paths = [os.path.join(temp_dir, f"step_{i:08d}") for i in ckpt_paths]
            else:
                ckpt_paths = Checkpointer.checkpoint_paths(cfg.dir)

            with mock.patch.object(Checkpointer, "checkpoint_paths", return_value=ckpt_paths):
                ckpt._run_garbage_collection()

                # step=None restores from the latest ckpt.
                self.assertNestedEqual(
                    expect_restore_step,
                    ckpt.restore(step=None, state=state)[0],
                )

                saved = []
                for step in range(10):
                    try:
                        restored_step, _ = ckpt.restore(step=step, state=state)
                        saved.append(restored_step)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.info("%s", e)
                self.assertSequenceEqual(expect_saved_steps, saved)
                ckpt.stop()

            # Check that the directories not in expect_saved_steps are indeed removed.
            expect_removed = set(range(10)) - set(expect_saved_steps)
            for path in [os.path.join(temp_dir, f"step_{i:08d}") for i in expect_removed]:
                self.assertFalse(os.path.exists(path))

    def test_check_state_structure_exact(self):
        actual = []
        key, value = "##", "test"
        target = [(key, value)]
        check_state_structure(actual, actual)
        check_state_structure(target, target)
        # Make sure the error message correctly shows the diff.
        with self.assertRaisesRegex(ValueError, f"{key}={value}"):
            check_state_structure(actual, target)

    def test_check_state_structure_exact_up_to_dtype(self):
        actual = [
            ("step", 1000),
            ("prng_key", {"dtype": "uint32", "shape": "(4,)"}),
            ("model/linear/bias", {"dtype": "bfloat16", "shape": "(8,)"}),
        ]
        target = [
            ("step", 1000),
            ("prng_key", {"dtype": "uint16", "shape": "(4,)"}),
            ("model/linear/bias", {"dtype": "float32", "shape": "(8,)"}),
        ]
        check_state_structure(actual, target, validation=CheckpointValidationType.EXACT_UP_TO_DTYPE)
        not_in_target = ("learner/ema/bias", "##")
        actual.append(not_in_target)
        # Make sure the error message shows the correct .
        with self.assertRaisesRegex(ValueError, "=".join(not_in_target)):
            check_state_structure(
                actual, target, validation=CheckpointValidationType.EXACT_UP_TO_DTYPE
            )

    def test_check_state_structure_contains_state(self):
        actual = [(f"#{i}#", f"{i}{i}") for i in range(1, 5)]
        target = [(f"#{i}#", f"{i}{i}") for i in range(2, 4)]
        check_state_structure(actual, target, validation=CheckpointValidationType.CONTAINS_STATE)
        not_contained = ("#7#", "77")
        target.append(not_contained)
        # Make sure the error message correctly shows the diff.
        with self.assertRaisesRegex(ValueError, "=".join(not_contained)):
            check_state_structure(
                actual, target, validation=CheckpointValidationType.CONTAINS_STATE
            )

    def test_check_state_structure_contains_state_up_to_dtype(self):
        actual = [
            ("step", 1000),
            ("prng_key", {"dtype": "uint32", "shape": "(4,)"}),
            ("model/linear/bias", {"dtype": "bfloat16", "shape": "(8,)"}),
            ("model/linear/weight", {"dtype": "bfloat16", "shape": "(4, 8)"}),
        ]
        target = [
            ("step", 1000),
            ("prng_key", {"dtype": "uint16", "shape": "(4,)"}),
            ("model/linear/bias", {"dtype": "float32", "shape": "(8,)"}),
        ]
        check_state_structure(
            actual, target, validation=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE
        )
        not_contained = ("learner/ema/bias", "##")
        target.append(not_contained)
        # Make sure the error message shows the correct .
        with self.assertRaisesRegex(ValueError, "=".join(not_contained)):
            check_state_structure(
                actual, target, validation=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE
            )

    def test_stop(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        cfg = _checkpointer_config()
        ckpt: Checkpointer = cfg.instantiate(parent=None)
        # GC thread is not started until the start_gc_thread() call.
        self.assertIsNone(ckpt._gc_thread)

        ckpt._start_gc_thread()
        self.assertIsNotNone(ckpt._gc_thread)
        ckpt.stop()
        # GC thread is terminated after stop() returns.
        self.assertIsNone(ckpt._gc_thread)

        # We can start gc and stop again.
        ckpt._start_gc_thread()
        self.assertIsNotNone(ckpt._gc_thread)
        ckpt.stop()
        self.assertIsNone(ckpt._gc_thread)

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_context(self, checkpointer_cls):
        ckpt = _checkpointer_config(checkpointer_cls).instantiate(parent=None)

        if checkpointer_cls is Checkpointer:
            with ckpt:
                self.assertIsNotNone(ckpt._gc_thread)
            self.assertIsNone(ckpt._gc_thread)

        # Nested contexts are not supported.
        with ckpt:
            with self.assertRaisesRegex(ValueError, "Already in a context"):
                with ckpt:
                    pass

        if checkpointer_cls is Checkpointer:
            self.assertIsNone(ckpt._gc_thread)

    def test_stop_on_exception(self):
        # Ensure that checkpointer gc thread terminates if there's an exception.
        ckpt = _checkpointer_config().instantiate(parent=None)

        def run():
            ckpt._start_gc_thread()
            raise ValueError("expected error")

        # By default, an exception in the main thread does not terminate child threads.
        run_thread = threading.Thread(target=run)
        run_thread.start()
        run_thread.join()
        self.assertFalse(ckpt._gc_stopping.is_set())
        ckpt.stop()  # Stop it explicitly, otherwise test will run forever.

        def run_in_context():
            with ckpt:
                raise ValueError("expected error")

        # With a context manager, we stop when context is exited.
        run_thread = threading.Thread(target=run_in_context)
        run_thread.start()
        run_thread.join()
        self.assertTrue(ckpt._gc_stopping.is_set())

    def test_summary_writer_checkpoint(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
            cfg.summary_writer = SummaryWriter.default_config()
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            self.assertIsNotNone(ckpt.summary_writer)

            ckpt.summary_writer.log_checkpoint = mock.Mock()

            state = dict(x=jnp.zeros([], dtype=jnp.int32))
            ckpt.save(step=1, state=state)
            ckpt.wait_until_finished()

            ckpt.summary_writer.log_checkpoint.assert_called_once()
            ckpt.stop()

    @parameterized.product(
        checkpointer_cls=[Checkpointer, OrbaxCheckpointer],
        mode=("max", "min"),
        metric_type=("array", "weighted_scalar"),
    )
    def test_best_metric_policy(self, checkpointer_cls, mode, metric_type):
        def _create_metric(value):
            if metric_type == "array":
                return jnp.asarray(value)
            elif metric_type == "weighted_scalar":
                return WeightedScalar(mean=jnp.asarray(value), weight=jnp.asarray(1.0))
            else:
                raise ValueError("Unsupported metric type!")

        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls).set(
                save_policy=BestMetricPolicy.default_config().set(
                    metric=EvalMetric(evaler_name="evaler", metric_name="metric"), mode=mode
                )
            )
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32))
            state2 = dict(x=jnp.ones([], dtype=jnp.int32) * 2)
            state4 = dict(x=jnp.ones([], dtype=jnp.int32) * 4)

            # Validate that first save succeeds.
            ckpt.save(
                step=0, state=state0, evaler_summaries={"evaler": {"metric": _create_metric(10)}}
            )
            ckpt.wait_until_finished()
            self.assertNestedEqual((0, state0), ckpt.restore(step=None, state=state0))

            ckpt.save(
                step=2, state=state2, evaler_summaries={"evaler": {"metric": _create_metric(5)}}
            )
            ckpt.wait_until_finished()
            if mode == "max":
                self.assertNestedEqual((0, state0), ckpt.restore(step=None, state=state0))
            else:
                self.assertNestedEqual((2, state2), ckpt.restore(step=None, state=state0))

            ckpt.save(
                step=4, state=state4, evaler_summaries={"evaler": {"metric": _create_metric(11)}}
            )
            ckpt.wait_until_finished()
            if mode == "max":
                self.assertNestedEqual((4, state4), ckpt.restore(step=None, state=state0))
            else:
                self.assertNestedEqual((2, state2), ckpt.restore(step=None, state=state0))
            ckpt.stop()

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_best_metric_policy_value_error(self, checkpointer_cls):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls).set(
                save_policy=BestMetricPolicy.default_config().set(
                    metric=EvalMetric(evaler_name="evaler", metric_name="metric"), mode="max"
                )
            )
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32))

            with self.assertRaisesRegex(ValueError, re.escape("evaler_summaries is empty")):
                ckpt.save(step=0, state=state0, evaler_summaries={})
                ckpt.wait_until_finished()

            with self.assertRaisesRegex(ValueError, re.escape("not found in evaler_summaries")):
                ckpt.save(
                    step=0, state=state0, evaler_summaries={"evaler2": {"metric": jnp.asarray(10)}}
                )
                ckpt.wait_until_finished()

            with self.assertRaisesRegex(ValueError, re.escape("not in evaler_summaries")):
                ckpt.save(
                    step=0, state=state0, evaler_summaries={"evaler": {"metric2": jnp.asarray(10)}}
                )
                ckpt.wait_until_finished()

            with self.assertRaisesRegex(ValueError, re.escape("is None")):
                ckpt.save(step=0, state=state0, evaler_summaries={"evaler": {"metric": None}})
                ckpt.wait_until_finished()

            with self.assertRaisesRegex(ValueError, re.escape("scalar")):
                ckpt.save(
                    step=0,
                    state=state0,
                    evaler_summaries={"evaler": {"metric": jnp.asarray([1, 2])}},
                )
                ckpt.wait_until_finished()
            ckpt.stop()

    def test_every_n_steps_policy(self):
        policy = every_n_steps_policy(n=3)
        self.assertFalse(policy(step=0, evaler_summaries={}))
        self.assertFalse(policy(step=1, evaler_summaries={}))
        self.assertFalse(policy(step=2, evaler_summaries={}))
        self.assertTrue(policy(step=3, evaler_summaries={}))
        self.assertFalse(policy(step=4, evaler_summaries={}))

    def test_every_n_steps_policy_min_step(self):
        policy = every_n_steps_policy(n=3, min_step=0)
        self.assertTrue(policy(step=0, evaler_summaries={}))
        self.assertFalse(policy(step=1, evaler_summaries={}))
        self.assertFalse(policy(step=2, evaler_summaries={}))
        self.assertTrue(policy(step=3, evaler_summaries={}))
        self.assertFalse(policy(step=4, evaler_summaries={}))

    def test_every_n_steps_and_last_policy(self):
        policy = every_n_steps_and_last_policy(n=5, max_step=13)
        self.assertTrue(policy(step=5, evaler_summaries={}))
        self.assertFalse(policy(step=9, evaler_summaries={}))
        self.assertTrue(policy(step=10, evaler_summaries={}))
        self.assertFalse(policy(step=11, evaler_summaries={}))
        self.assertFalse(policy(step=12, evaler_summaries={}))
        self.assertTrue(policy(step=13, evaler_summaries={}))

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_latest_checkpoint_path_and_step(self, checkpointer_cls: Type[BaseCheckpointer]):
        with tempfile.TemporaryDirectory() as td:
            # Test that the most recent checkpoint is returned.
            ckpt_paths = {}
            for step in [1, 2, 10, 11]:
                if checkpointer_cls is Checkpointer:
                    ckpt_paths[step] = os.path.join(td, f"step_{step:08d}")
                    os.makedirs(ckpt_paths[step])
                    if step <= 10:
                        with open(
                            os.path.join(ckpt_paths[step], "index"), "w", encoding="utf-8"
                        ) as f:
                            f.write("dummy")
                elif checkpointer_cls is OrbaxCheckpointer:
                    if step <= 10:
                        ckpt_paths[step] = os.path.join(td, f"step_{step:08d}")
                        os.makedirs(ckpt_paths[step])
                else:
                    raise NotImplementedError(checkpointer_cls)
            final_ckpt_path = ckpt_paths[10]
            # Note: step 11 is not complete, so the latest path returns step 10.
            self.assertEqual(checkpointer_cls.latest_checkpoint_path(td), final_ckpt_path)
            self.assertEqual(checkpointer_cls.latest_checkpoint_step(td), 10)

    @parameterized.parameters([Checkpointer, OrbaxCheckpointer])
    def test_read_state_spec(self, checkpointer_cls: Type[BaseCheckpointer]):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config(checkpointer_cls)
            cfg.save_policy.min_step = 0
            ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
            state0 = dict(
                **{
                    f"v_{str(dtype.dtype)}": jnp.zeros([], dtype=dtype)
                    for dtype in (jnp.uint32, jnp.int32, jnp.int64)
                },
                **{
                    f"v_{str(dtype.dtype)}": jnp.zeros([4], dtype=dtype)
                    for dtype in (jnp.float16, jnp.float32, jnp.float64)
                },
                **{
                    f"v_{str(dtype.dtype)}": jnp.zeros([4, 2], dtype=dtype)
                    for dtype in (jnp.bfloat16, jnp.bool_)
                },
            )
            ckpt.save(step=0, state=state0)
            ckpt.wait_until_finished()
            # Tests `read_state_spec`.
            state_spec = read_state_spec(checkpointer_cls.latest_checkpoint_path(cfg.dir))
            self.assertNestedEqual(
                state_spec,
                jax.tree.map(lambda t: utils.TensorSpec(shape=t.shape, dtype=t.dtype), state0),
            )
            step, state1 = ckpt.restore(state=state_spec)
            self.assertNestedEqual(0, step)
            self.assertNestedEqual(state0, state1)

    def test_vdict_order_compatibility(self):
        """Tests that changing VDict to correctly have the same pytree flattenning behavior as dict
        can still restore old checkpoints created using the old VDict version.
        """
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        # Subclass VDict so that VDict-specific code works the same with this class.
        # The pytree flattening / serialization logic is defined explicitly for this subclass
        # and is not inherited.
        @jax.tree_util.register_pytree_node_class
        class OldVDict(VDict):
            """The original implementation of VDict."""

            def __repr__(self):
                return f"VDict({super().__repr__()})"

            def tree_flatten_with_keys(self):
                raise NotImplementedError

            def tree_flatten(self):
                # Convert dict_values and_keys to lists to avoid holding reference to the VDict.
                return (list(self.values()), list(self.keys()))

            @classmethod
            def tree_unflatten(cls, keys, values):
                return cls(zip(keys, values))

        with _mesh(mesh_shape):
            with unittest.mock.patch.dict(globals(), {"SWITCHABLE_VDICT_IMPL": OldVDict}):
                cfg = _checkpointer_config()
                cfg.save_policy.min_step = 0
                ckpt: Checkpointer = cfg.instantiate(parent=None)
                # VDict with out of order keys.
                state0 = dict(a=3, b=SwitchableVDict(d=6, b=5))
                state0 = jax.tree.map(jnp.asarray, state0)
                self.assertEqual(list(state0["b"].keys()), ["d", "b"])
                ckpt.save(step=0, state=state0)
                ckpt.wait_until_finished()

                _, result = ckpt.restore(step=0, state=state0)
                self.assertNestedEqual(state0, result)
                self.assertEqual(list(result["b"].keys()), ["d", "b"])
            with unittest.mock.patch.dict(globals(), {"SWITCHABLE_VDICT_IMPL": VDict}):
                _, result = ckpt.restore(step=0, state=state0)
                self.assertNestedEqual(state0, result)
                self.assertEqual(list(result["b"].keys()), ["b", "d"])

                after_tree_map = jax.tree.map(lambda x: x, result)
                self.assertEqual(list(after_tree_map["b"].keys()), ["b", "d"])

                _, result = ckpt.restore(step=0, state=after_tree_map)
                self.assertNestedEqual(state0, result)
                self.assertEqual(list(result["b"].keys()), ["b", "d"])

    def test_save_with_exception(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        cfg = _checkpointer_config(Checkpointer)
        cfg.save_policy.min_step = 0
        ckpt: BaseCheckpointer = cfg.instantiate(parent=None)
        old_stop = ckpt.stop
        has_exc = None

        def new_stop(has_exception):
            nonlocal has_exc
            old_stop(has_exception=has_exception)
            has_exc = has_exception

        with mock.patch.object(ckpt, "stop", new_stop):
            with self.assertRaises(TypeError):
                with _mesh(mesh_shape), ckpt:
                    # Pass a non serializable object to trigger an exception.
                    state0 = dict(
                        x=jnp.zeros([], dtype=jnp.int32),
                        y=jnp.ones([2], dtype=jnp.float32),
                        z=lambda: 1,
                    )
                    ckpt.save(step=0, state=state0)
                    ckpt.wait_until_finished()
        self.assertEqual(has_exc, True)


class TensorStoreStateStorageTest(test_utils.TestCase):
    @parameterized.product(
        max_concurrent_gb=[None, 1],
        max_data_shard_degree=[None, 1, -1],
        shard_threshold_bytes=[None, 0, int(1024**3)],
    )
    def test_checkpointer_configs(
        self,
        max_concurrent_gb: Optional[int],
        max_data_shard_degree: int,
        shard_threshold_bytes: int,
    ):
        cfg = TensorStoreStateStorage.default_config().set(
            max_concurrent_gb=max_concurrent_gb,
            max_data_shard_degree=max_data_shard_degree,
            shard_threshold_bytes=shard_threshold_bytes,
        )
        if (
            not max_concurrent_gb
            and not max_data_shard_degree
            and shard_threshold_bytes is not None
        ):
            with self.assertRaises(ValueError):
                storage = cfg.instantiate()
            return
        storage = cfg.instantiate()
        if max_concurrent_gb is not None or max_data_shard_degree:
            self.assertIsInstance(storage._manager, BoundedDataShardedAsyncCheckpointManager)
        else:
            self.assertIsInstance(
                storage._manager, array_serialization.GlobalAsyncCheckpointManager
            )

    def test_max_concurrent_restore_gb_setting(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            TensorStoreStateStorage.default_config().set(max_concurrent_restore_gb=-2).instantiate()
        t = TensorStoreStateStorage.default_config().instantiate()
        # Test default value.
        self.assertEqual(t._max_concurrent_restore_gb, 32)

    def test_stop(self):
        storage = TensorStoreStateStorage.default_config().instantiate()
        worker_result = None

        def worker():
            nonlocal worker_result
            time.sleep(1)
            worker_result = True

        storage._executor.submit(worker)
        storage.stop()
        self.assertTrue(worker_result, "storage.stop() should wait for executor to finish.")

        with self.assertRaisesRegex(RuntimeError, "cannot schedule new futures after shutdown"):
            storage._executor.submit(worker)

    @parameterized.product(
        restore_floats_as=[jnp.float32, jnp.bfloat16, jnp.int32, jnp.int16],
        premaped_buffer_size=[1024 * 1024 * 1024, 1],
    )
    def test_save_and_restore_from_dir(
        self, restore_floats_as: jnp.dtype, premaped_buffer_size: int
    ):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        def make_state(float_dtype):
            return dict(x=jnp.zeros([1024], dtype=jnp.int32), y=jnp.ones([2], dtype=float_dtype))

        with (
            _mesh(mesh_shape),
            mock.patch.object(
                axlearn_serialization, "_get_premapped_buffer_size", lambda: premaped_buffer_size
            ),
        ):
            state = make_state(float_dtype=jnp.float32)
            storage = TensorStoreStateStorage.default_config().instantiate()
            with tempfile.TemporaryDirectory() as root_dir:
                step = 1000
                # Save ckpt.
                final_dir = os.path.join(root_dir, f"step_{step:08d}")
                storage.save_to_dir(step=step, state=state, ckpt_dir=final_dir)
                storage.wait_until_finished()

                # Successfully restores with different dtypes.
                restored_state = storage.restore_from_dir(
                    step,
                    state=make_state(float_dtype=restore_floats_as),
                    ckpt_dir=final_dir,
                    validation=CheckpointValidationType.EXACT_UP_TO_DTYPE,
                )
                self.assertNestedEqual(
                    restored_state,
                    (
                        state
                        if restore_floats_as is None
                        else make_state(float_dtype=restore_floats_as)
                    ),
                )

    @parameterized.product(
        restore_floats_as=[jnp.float32, jnp.bfloat16, jnp.int32, jnp.int16],
        max_concurrent_gb=[None, 1],
    )
    def test_save_and_restore_from_dir_async(
        self, restore_floats_as: jnp.dtype, max_concurrent_gb: Optional[int]
    ):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        def make_state(float_dtype):
            return dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=float_dtype))

        with _mesh(mesh_shape):
            state = make_state(float_dtype=jnp.float32)
            storage = (
                TensorStoreStateStorage.default_config()
                .set(max_concurrent_gb=max_concurrent_gb)
                .instantiate()
            )
            with tempfile.TemporaryDirectory() as root_dir:
                step = 1000
                # Save ckpt.
                final_dir = os.path.join(root_dir, f"step_{step:08d}")

                async def save():
                    storage.save_to_dir(step=step, state=state, ckpt_dir=final_dir)
                    storage.wait_until_finished()

                asyncio.run(save())

                async def restore():
                    # Successfully restores with different dtypes.
                    restored_state = storage.restore_from_dir(
                        step,
                        state=make_state(float_dtype=restore_floats_as),
                        ckpt_dir=final_dir,
                        validation=CheckpointValidationType.EXACT_UP_TO_DTYPE,
                    )
                    return restored_state

                restored_state = asyncio.run(restore())
                self.assertNestedEqual(
                    restored_state,
                    (
                        state
                        if restore_floats_as is None
                        else make_state(float_dtype=restore_floats_as)
                    ),
                )

    def test_save_to_dir_async(self):
        """Tests that serialization happens async."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            storage = TensorStoreStateStorage.default_config().instantiate()
            with tempfile.TemporaryDirectory() as temp_dir:
                # We do a blocking set on the main thread and a blocking get on commit.
                q = queue.Queue()
                committed_value = None

                def on_commit_callback(**kwargs):
                    del kwargs
                    nonlocal committed_value
                    committed_value = q.get(block=True)

                storage.save_to_dir(
                    step=1,
                    state=dict(x=jnp.zeros([], dtype=jnp.int32)),
                    ckpt_dir=temp_dir,
                    on_commit_callback=on_commit_callback,
                )
                q.put("test", block=True)
                storage.wait_until_finished()
                self.assertEqual("test", committed_value)

    def test_check_state_structure_compat(self):
        """Tests that empty dicts in state structure do not trigger validation errors."""

        check_state_structure(
            list(utils.flatten_items({"a": {"x": jnp.array(123)}})),
            list(utils.flatten_items({"a": {"x": jnp.array(123)}, "b": {}})),
        )


def _write_shards(lines: Iterable[str], *, path_prefix, num_shards) -> list[str]:
    filenames = [
        f"{path_prefix}-{shard_id:05d}-of-{num_shards:05d}" for shard_id in range(num_shards)
    ]
    files = [fs.open(filename, "w") for filename in filenames]
    for i, line in enumerate(lines):
        files[i % num_shards].write(line + "\n")
    return filenames


class TfIteratorTest(test_utils.TestCase):
    def test_restored_iterator_resumes(self):
        executor = ThreadPoolExecutor(1)
        num_examples = 30
        tempdir = tempfile.mkdtemp()
        lines = [str(id) for id in range(num_examples)]
        filenames = _write_shards(lines, path_prefix=os.path.join(tempdir, "records"), num_shards=4)
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.interleave(
            # pylint: disable-next=unnecessary-lambda
            lambda filename: tf.data.TextLineDataset(filename),
            num_parallel_calls=2,
        )
        ds = ds.shuffle(5)
        it = iter(ds)
        seen = []
        for seq_num in range(num_examples):
            if seq_num % 3 == 0:
                # Save and restore the iterator.
                ckpt_path = os.path.join(tempdir, "ckpt")
                prev_it = it
                # Manually increase the delay of executor to test `it` mutation after
                # call to async_save_tf_savables doesn't affect saving.
                blocker = executor.submit(lambda: time.sleep(1))
                f = async_save_tf_savables({"it": it}, executor=executor, dir=ckpt_path)
                next(it)  # modify it in place
                it = iter(ds)  # reset `it`.
                self.assertIsNot(it, prev_it)
                blocker.result()
                f.result()  # wait for async save
                restore_tf_savables({"it": it}, dir=ckpt_path)
            line = next(it).numpy()
            seen.append(int(line))
        # Every input example should be seen exactly once, since the saved and restored iterator
        # should continue from the interruption.
        self.assertSetEqual(set(seen), set(range(num_examples)))

    def test_no_save_input_iterator(self):
        executor = ThreadPoolExecutor(1)
        tmpdir = tempfile.mkdtemp()
        ckpt_dir = os.path.join(tmpdir, "tf_ckpt")
        self.assertEqual(0, len(fs.listdir(tmpdir)))
        # Test that when we don't save input iterator, tf dirs are not created.
        fut = async_save_tf_savables({}, executor=executor, dir=ckpt_dir)
        fut.result()
        self.assertEqual([], fs.listdir(tmpdir))
        # Test that dirs are created if we save.
        ds = tf.data.Dataset.from_tensor_slices([])
        fut = async_save_tf_savables({"it": iter(ds)}, executor=executor, dir=ckpt_dir)
        fut.result()
        self.assertEqual(["tf_ckpt"], fs.listdir(tmpdir))


SWITCHABLE_VDICT_IMPL: Optional[type[VDict]] = None


# Subclass VDict so that VDict-specific code works the same with this class.
# The pytree flattening logic is defined explicitly for this subclass
# and is not inherited.
@jax.tree_util.register_pytree_with_keys_class
class SwitchableVDict(VDict):
    """A VDict that can switch its implementation between different implementations.

    For testing backwards compatibility.
    """

    def __repr__(self):
        return SWITCHABLE_VDICT_IMPL.__repr__(self)

    def tree_flatten_with_keys(self):
        try:
            return SWITCHABLE_VDICT_IMPL.tree_flatten_with_keys(self)
        except NotImplementedError:
            # tree_paths() no longer generates named keys for pytrees that don't register
            # their child keys, so we simulate the child keys from the old implementation
            # of tree_paths().
            values, keys = SWITCHABLE_VDICT_IMPL.tree_flatten(self)
            key_value = [(jax.tree_util.DictKey(k), v) for k, v in zip(keys, values)]
            return key_value, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(SWITCHABLE_VDICT_IMPL.tree_unflatten(keys, values))


serialization.register_serialization_state(
    SwitchableVDict,
    # pylint: disable-next=protected-access
    ty_to_state_dict=serialization._dict_state_dict,
    # pylint: disable-next=protected-access
    ty_from_state_dict=serialization._restore_dict,
)

if __name__ == "__main__":
    absltest.main()
