# Copyright © 2023 Apple Inc.

"""Tests checkpointer.py.

Some tests are intended to be run on TPU.
"""
# pylint: disable=no-self-use,protected-access
import os
import re
import tempfile
import threading
from typing import Iterable, List, Sequence

import flax.core
import jax
import pytest
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental import mesh_utils

from axlearn.common import test_utils, utils
from axlearn.common.checkpointer import (
    BestMetricPolicy,
    Checkpointer,
    CheckpointValidationType,
    EvalMetric,
    TensorStoreStateStorage,
    check_state_structure,
    every_n_steps_policy,
    latest_checkpoint_path,
    restore_tf_savables,
    save_tf_savables,
)
from axlearn.common.metrics import WeightedScalar


def _mesh(mesh_shape: Sequence[int]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ("data", "model"))


def _checkpointer_config():
    return Checkpointer.default_config().set(name="test", dir=tempfile.mkdtemp())


class CheckpointerTest(test_utils.TestCase):
    def test_save_and_restore(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
            cfg.save_policy.min_step = 0
            ckpt: Checkpointer = cfg.instantiate(parent=None)
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
            with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
                ckpt.restore(
                    step=None,
                    state=dict(
                        x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([3], dtype=jnp.float32)
                    ),
                )

            # When the given state has a different dict shape: [1] instead of [] for x.
            with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
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

    def test_save_and_restore_latest_valid(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32))

            # Restoring from an empty dir returns the input state if step=None.
            self.assertNestedEqual((None, state0), ckpt.restore(state=state0))

            def create_corrupt_ckpt(step):
                final_dir = ckpt.ckpt_dir(step)
                tf.io.gfile.makedirs(final_dir)

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

    @parameterized.parameters((1, 1), (2, 2), (4, 2))
    def test_gda(self, *mesh_shape):
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
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

    @parameterized.parameters(utils.VDict, flax.core.FrozenDict)
    def test_custom_dict(self, custom_dict_type):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
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

    def test_input_iterator(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = _checkpointer_config()
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            input_iter = iter(tf.data.Dataset.from_tensor_slices([1, 2, 3]))
            # Move the input_iter.
            self.assertEqual(next(input_iter), 1)
            state0 = dict(
                x=jnp.zeros([], dtype=jnp.int32),
                input_iter=input_iter,
            )

            ckpt.save(step=100, state=state0)
            ckpt.wait_until_finished()

            # Restore with state structure hints will preserve VDict.
            step, restored_state = ckpt.restore(step=None, state=state0)
            self.assertEqual(100, step)
            self.assertNestedEqual(state0, restored_state)
            # The restored_state contains the input_iter pointing to the next value.
            self.assertEqual(next(restored_state["input_iter"]), 2)
            ckpt.stop()

    def test_garbage_collection(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape):
            cfg = Checkpointer.default_config().set(
                name="test",
                dir=tempfile.mkdtemp(),
                keep_last_n=3,
                keep_every_n_steps=2,
                gc_loop_interval_seconds=1,
            )
            cfg.save_policy.min_step = 0
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state = dict(x=jnp.zeros([], dtype=jnp.int32))

            for step in range(10):
                ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()
            ckpt.run_garbage_collection()

            # step=None restores from the latest ckpt.
            self.assertNestedEqual(9, ckpt.restore(step=None, state=state)[0])

            saved = []
            for step in range(10):
                try:
                    restored_step, _ = ckpt.restore(step=step, state=state)
                    saved.append(restored_step)
                except Exception as e:  # pylint: disable=broad-except
                    logging.info("%s", e)
            self.assertSequenceEqual([0, 2, 4, 6, 7, 8, 9], saved)
            ckpt.stop()

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

        ckpt.start_gc_thread()
        self.assertIsNotNone(ckpt._gc_thread)
        ckpt.stop()
        # GC thread is terminated after stop() returns.
        self.assertIsNone(ckpt._gc_thread)

        # We can start gc and stop again.
        ckpt.start_gc_thread()
        self.assertIsNotNone(ckpt._gc_thread)
        ckpt.stop()
        self.assertIsNone(ckpt._gc_thread)

    def test_context(self):
        ckpt = _checkpointer_config().instantiate(parent=None)

        with ckpt:
            self.assertIsNotNone(ckpt._gc_thread)
        self.assertIsNone(ckpt._gc_thread)

        # Nested contexts are not supported.
        with ckpt:
            with self.assertRaisesRegex(ValueError, "Already in a context"):
                with ckpt:
                    pass
        self.assertIsNone(ckpt._gc_thread)

    def test_stop_on_exception(self):
        # Ensure that checkpointer gc thread terminates if there's an exception.
        ckpt = _checkpointer_config().instantiate(parent=None)

        def run():
            ckpt.start_gc_thread()
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

    @parameterized.product(
        mode=("max", "min"),
        metric_type=("array", "weighted_scalar"),
    )
    def test_best_metric_policy(self, mode, metric_type):
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
            cfg = _checkpointer_config().set(
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

    def test_best_metric_policy_value_error(self):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        with _mesh(mesh_shape):
            cfg = _checkpointer_config().set(
                save_policy=BestMetricPolicy.default_config().set(
                    metric=EvalMetric(evaler_name="evaler", metric_name="metric"), mode="max"
                )
            )
            ckpt: Checkpointer = cfg.instantiate(parent=None)
            state0 = dict(x=jnp.zeros([], dtype=jnp.int32))

            with pytest.raises(ValueError, match=re.escape("evaler_summaries is empty")):
                ckpt.save(step=0, state=state0, evaler_summaries={})
                ckpt.wait_until_finished()

            with pytest.raises(ValueError, match=re.escape("not found in evaler_summaries")):
                ckpt.save(
                    step=0, state=state0, evaler_summaries={"evaler2": {"metric": jnp.asarray(10)}}
                )
                ckpt.wait_until_finished()

            with pytest.raises(ValueError, match=re.escape("not in evaler_summaries")):
                ckpt.save(
                    step=0, state=state0, evaler_summaries={"evaler": {"metric2": jnp.asarray(10)}}
                )
                ckpt.wait_until_finished()

            with pytest.raises(ValueError, match=re.escape("is None")):
                ckpt.save(step=0, state=state0, evaler_summaries={"evaler": {"metric": None}})
                ckpt.wait_until_finished()

            with pytest.raises(ValueError, match=re.escape("scalar")):
                ckpt.save(
                    step=0,
                    state=state0,
                    evaler_summaries={"evaler": {"metric": jnp.asarray([1, 2])}},
                )
                ckpt.wait_until_finished()

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

    def test_latest_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as td:
            # Test that the most recent checkpoint is returned.
            ckpt_paths = {}
            for step in [1, 2, 10, 11]:
                ckpt_paths[step] = os.path.join(td, f"step_{step:08d}")
                os.makedirs(ckpt_paths[step])
                if step <= 10:
                    with open(os.path.join(ckpt_paths[step], "index"), "w", encoding="utf-8") as f:
                        f.write("dummy")
            final_ckpt_path = ckpt_paths[10]
            # Note: step 11 is not complete, so the latest path returns step 10.
            self.assertEqual(latest_checkpoint_path(td), final_ckpt_path)


class TensorStoreStateStorageTest(test_utils.TestCase):
    @parameterized.parameters(jnp.float32, jnp.bfloat16, jnp.int32, jnp.int16)
    def test_save_and_restore_from_dir(self, restore_floats_as: jnp.dtype):
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        def make_state(float_dtype):
            return dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=float_dtype))

        with _mesh(mesh_shape):
            state = make_state(float_dtype=jnp.float32)
            storage = TensorStoreStateStorage.default_config().instantiate()
            with tempfile.TemporaryDirectory() as root_dir:
                step = 1000
                # Save ckpt.
                final_dir = os.path.join(root_dir, f"step_{step:08d}")
                storage.save_to_dir(step=step, state=state, ckpt_dir=final_dir)
                storage.wait_until_finished()

                # Restore.
                def restore_state():
                    return storage.restore_from_dir(
                        step,
                        state=make_state(float_dtype=restore_floats_as),
                        ckpt_dir=final_dir,
                        validation=CheckpointValidationType.EXACT_UP_TO_DTYPE,
                    )

                # Succesfully restores with different dtypes.
                restored_state = restore_state()
                self.assertNestedEqual(
                    restored_state,
                    state
                    if restore_floats_as is None
                    else make_state(float_dtype=restore_floats_as),
                )


def _write_shards(lines: Iterable[str], *, path_prefix, num_shards) -> List[str]:
    filenames = [
        f"{path_prefix}-{shard_id:05d}-of-{num_shards:05d}" for shard_id in range(num_shards)
    ]
    files = [tf.io.gfile.GFile(filename, "w") for filename in filenames]
    for i, line in enumerate(lines):
        files[i % num_shards].write(line + "\n")
    return filenames


class TfIteratorTest(test_utils.TestCase):
    def test_restored_iterator_resumes(self):
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
                save_tf_savables({"it": it}, dir=ckpt_path)
                it = iter(ds)  # reset `it`.
                self.assertIsNot(it, prev_it)
                restore_tf_savables({"it": it}, dir=ckpt_path)
            line = next(it).numpy()
            seen.append(int(line))
        # Every input example should be seen exactly once, since the saved and restored iterator
        # should continue from the interruption.
        self.assertSetEqual(set(seen), set(range(num_examples)))


if __name__ == "__main__":
    absltest.main()
