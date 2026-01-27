# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# AI-Hypercomputer/maxtext:
# Copyright 2023–2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License").

"""Checkpointing utilities using orbax.

See also checkpointer.py for other checkpointing utilities and checkpointer_test.py for tests.
"""

import asyncio
import copy
import dataclasses
import functools
import os
from concurrent import futures
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization.type_handlers import ArrayHandler
from orbax.checkpoint.checkpoint_manager import CheckpointInfo, _ShouldSaveFnPolicy
from tensorflow.python.checkpoint import async_checkpoint_helper

from axlearn.common import utils
from axlearn.common.checkpointer import (
    STEP_NUM_DIGITS,
    STEP_PREFIX,
    BaseCheckpointer,
    CheckpointValidationType,
    PythonSavable,
    check_state_structure,
    maybe_restore_python_savables,
    maybe_save_python_savables,
)
from axlearn.common.config import config_class
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec

try:
    # The import also registers the checkpoint handlers.
    import grain.python as grain

    _GrainIterator = Union[grain.DatasetIterator, grain.PyGrainDatasetIterator]
    _GRAIN_INSTALLED = True
except ImportError:
    logging.warning("grain is not installed; checkpointing grain iterators will not work.")
    _GRAIN_INSTALLED = False


class _TfIteratorHandler(ocp.type_handlers.TypeHandler):
    """Serializes tf.data.Iterator.

    Reference:
    https://orbax.readthedocs.io/en/latest/custom_handlers.html#custom-serialization-deserialization

    Note that since handlers are global instances (potentially shared by multiple checkpointer
    instances), we construct and cleanup the executor per-serialize/deserialize call.
    """

    def __init__(self):
        super().__init__()
        self._executor = futures.ThreadPoolExecutor()
        self._tf_ckpt_cache: dict[tf.data.Iterator, tf.train.Checkpoint] = {}

    # Must be a subclass of RestoreArgs for `PyTreeRestore` to recognize it.
    @dataclasses.dataclass
    class RestoreArgs(ocp.type_handlers.RestoreArgs):
        item: Optional[tf.data.Iterator] = None

    def typestr(self) -> str:
        return "TfIterator"

    def _ckpt_dir(self, info: ocp.type_handlers.ParamInfo) -> str:
        # Each worker writes its tf checkpoints under a different path.
        return os.path.join(
            os.path.dirname(info.parent_dir), "tfds", f"tf_{jax.process_index()}", info.name
        )

    def _get_or_create_tf_ckpt(self, value: tf.data.Iterator) -> tf.train.Checkpoint:
        # This is to avoid recreating `tf.train.Checkpoint` instances on each
        # checkpointing. When `enable_async` is `True` the resources will only
        # be released at the end of program.
        if value not in self._tf_ckpt_cache:
            self._tf_ckpt_cache[value] = tf.train.Checkpoint(value)
        return self._tf_ckpt_cache[value]

    def _sync_tf_ckpt_and_check_error(self, ckpt: tf.train.Checkpoint):
        # When `enable_async=True`, `ckpt.sync` will always return silently even
        # if it failed to save the checkpoint correctly. What makes it worse is
        # that Orbax Checkpoint Manager relies on the successful execution to
        # write the `commit_success.txt` file. Here we check and rethrow the
        # error if there's any.
        ckpt.sync()
        # pylint: disable=protected-access
        # pytype: disable=attribute-error
        assert isinstance(ckpt._async_checkpointer(), async_checkpoint_helper.AsyncCheckpointHelper)
        ckpt._async_checkpointer()._check_async_thread_error()
        # pytype: enable=attribute-error
        # pylint: enable=protected-access

    async def serialize(
        self,
        values: Sequence[tf.data.Iterator],
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[ocp.args.PyTreeSave]],
    ) -> List[futures.Future]:
        """Serializes `values` into corresponding `info.path`s."""
        del args  # Unused.
        futs = []
        for value, info in zip(values, infos, strict=False):
            tf_ckpt = self._get_or_create_tf_ckpt(value)
            tf_ckpt.write(self._ckpt_dir(info), tf.train.CheckpointOptions(enable_async=True))
            futs.append(
                self._executor.submit(
                    functools.partial(self._sync_tf_ckpt_and_check_error, tf_ckpt)
                )
            )
        return futs

    async def deserialize(
        self,
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[RestoreArgs]] = None,
    ) -> Sequence[tf.data.Iterator]:
        if args is None:
            raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")

        tf_ckpts = []
        for arg, info in zip(args, infos, strict=False):
            tf_ckpt = self._get_or_create_tf_ckpt(arg.item)
            tf_ckpt.read(self._ckpt_dir(info), tf.train.CheckpointOptions(enable_async=True))
            tf_ckpts.append(tf_ckpt)

        await asyncio.gather(
            *(
                asyncio.get_event_loop().run_in_executor(
                    self._executor, functools.partial(self._sync_tf_ckpt_and_check_error, ckpt)
                )
                for ckpt in tf_ckpts
            )
        )

        return [arg.item for arg in args]

    async def metadata(
        self, infos: Sequence[ocp.type_handlers.ParamInfo]
    ) -> Sequence[ocp.metadata.Metadata]:
        return [ocp.metadata.Metadata(name=info.name, directory=info.path) for info in infos]


ocp.type_handlers.register_type_handler(tf.data.Iterator, _TfIteratorHandler(), override=True)
ocp.type_handlers.register_type_handler(
    jax.Array,
    ArrayHandler(
        array_metadata_store=array_metadata_store_lib.Store(),
        use_replica_parallel=False,
        enable_write_sharding_file=False,
    ),
    override=True,
)


if _GRAIN_INSTALLED:
    # TODO(markblee): Generalize to PythonSavableHandler.
    class _GrainDatasetIteratorHandler(ocp.type_handlers.TypeHandler):
        """Serializes grain dataset iterators."""

        def __init__(self):
            super().__init__()
            self._executor = futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="GrainDatasetIteratorHandler"
            )

        @dataclasses.dataclass
        class RestoreArgs(ocp.type_handlers.RestoreArgs):
            item: Optional[_GrainIterator] = None

        def typestr(self) -> str:
            return "DatasetIterator"

        def _ckpt_dir(self, info: ocp.type_handlers.ParamInfo) -> str:
            # Each worker writes its grain checkpoints under a different path.
            return os.path.join(
                os.path.dirname(info.parent_dir),
                "python",
                f"python_{jax.process_index()}",
            )

        async def serialize(
            self,
            values: Sequence[grain.DatasetIterator],
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[ocp.args.PyTreeSave]],
        ) -> List[futures.Future]:
            """Serializes `values` into corresponding `info.path`s."""
            del args  # Unused.
            futs = []
            for value, info in zip(values, infos, strict=False):
                futs.append(
                    self._executor.submit(
                        maybe_save_python_savables,
                        {info.name: value},
                        dir=self._ckpt_dir(info),
                    )
                )

            return futs

        async def deserialize(
            self,
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[RestoreArgs]] = None,
        ) -> Sequence[_GrainIterator]:
            if args is None:
                raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")

            await asyncio.gather(
                *(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        functools.partial(
                            maybe_restore_python_savables,
                            {info.name: arg.item},
                            dir=self._ckpt_dir(info),
                        ),
                    )
                    for arg, info in zip(args, infos, strict=False)
                )
            )

            return [arg.item for arg in args]

        async def metadata(
            self, infos: Sequence[ocp.type_handlers.ParamInfo]
        ) -> Sequence[ocp.metadata.Metadata]:
            return [ocp.metadata.Metadata(name=info.name, directory=info.path) for info in infos]

    ocp.type_handlers.register_type_handler(
        grain.DatasetIterator, _GrainDatasetIteratorHandler(), override=True
    )


class _CheckpointManagerWithTrackerFile(ocp.CheckpointManager):
    # pylint: disable=line-too-long
    """
    In some extreme cases, the number of available checkpoints may be quite
    large and the time spent on listing checkpoints can take > 10 minutes. This
    implementation extends the original CheckpointManager from Orbax to support
    reading only the latest checkpoint from a tracker file. So the time spent on
    initialization is constant and small. Inspired by
    [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/98d8c56dbdc9cc91b8a473debcf400958bba4524/megatron/training/checkpointing.py#L259).

    TODO(jtian22): Avoid overriding private methods or up merge it as a configurable option.
    """

    # pylint: enable=line-too-long

    def __init__(self, *args, tracker_filename="latest_checkpointed_step.txt", **kwargs):
        self._tracker_filename = tracker_filename
        super().__init__(*args, **kwargs)

    @property
    def tracker_file_path(self) -> epath.Path:
        return self.directory / self._tracker_filename

    def _load_checkpoint_infos(self, skip_metadata_read=False) -> List[CheckpointInfo]:
        """The original version looks up in the root directory and return all
        the successfully committed steps. Here we only return the latest one
        recorded in the track file. If that one is invalid or any exception is
        caught, we fallback to the original implementation.

        The checkpoint garbage collection thread will still work during current
        restart attempt, however steps prior to the one read from the tracker
        file at the start of current restart attempt will be skipped.

        TODO(jtian22): Improve GC of unnecessary checkpoints.
        """
        try:
            latest_step = int(self.tracker_file_path.read_text())
            latest_step_directory = self._get_write_step_directory(latest_step, self.directory)

            if ocp.step.is_path_finalized(latest_step_directory):
                mtime = self.tracker_file_path.stat().mtime
                latest_ckpt_info = CheckpointInfo(latest_step, datetime.fromtimestamp(mtime), None)
                return [latest_ckpt_info]
            else:
                raise ValueError(
                    (
                        f"The checkpoint {latest_step_directory} is corrupted!"
                        "The commit success file is missing!"
                    )
                )
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            logging.warning(
                (
                    "Failed to read latest checkpoint from tracker file %s!"
                    "Fallback to reading subfolders under %s"
                    "Error message: %s"
                ),
                self.tracker_file_path,
                self.directory,
                e,
            )
            return super()._load_checkpoint_infos(skip_metadata_read)

    def _finalize_checkpoint(self, step: int):
        super()._finalize_checkpoint(step)
        # In the worst (and rare) case, the checkpoint at `step` was just
        # successfully saved but the job then restarted and the tracker file was
        # not updated. Then we lost the progress up to the second latest
        # checkpoint.
        save_directory = self._get_write_step_directory(step, self.directory)
        if ocp.utils.is_primary_host(self._multiprocessing_options.primary_host):
            if ocp.step.is_path_finalized(save_directory):
                self.tracker_file_path.write_text(str(step))


class OrbaxCheckpointer(BaseCheckpointer):
    """A checkpointer that uses orbax CheckpointManager.

    NOTE: While this class uses index files to do additional validation on checkpoint state, the
    index file is not used for committing checkpoints (i.e., it is not reliable to check for the
    presence of 'index' file). In Orbax, checkpoints are committed differently on different
    filesystems. If the filesystem supports atomic renames, un-committed checkpoints are represented
    by a "tmp" prefix. OTOH, if the filesystem does not support atomic renames (such as GCS as of
    writing), committed checkpoints are represented by a commit file with the name
    "commit_success.txt". Thus, an incomplete checkpoint can still contain an "index" file (note
    that valid checkpoints should always contain an "index" file). In general, users should instead
    use `checkpoint_paths` to identify committed checkpoints.
    """

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxCheckpointer."""

        # Keep this many past ckpts.
        keep_last_n: int = 1
        # If > 0, permanently retain checkpoints at step intervals divisible by this value.
        # This is the Orbax equivalent of `keep_every_n_steps` in the default AXLearn checkpointer.
        keep_period: Optional[int] = None
        # Checkpoint validation during restore.
        validation_type: CheckpointValidationType = CheckpointValidationType.EXACT
        # Will be passed to ocp.options.AsyncOptions(timeout_secs).
        # It is the timeout for async checkpointing operations in Orbax.
        async_timeout_secs: int = 600
        max_concurrent_save_gb: Optional[int] = None
        max_concurrent_restore_gb: Optional[int] = None
        # An AXLearn training job may use multiple TPU slices. When restoring, by default, all
        # slices read the checkpoint from GCS. If True, only the first replica loads from GCS and
        # broadcasts to other slices via the training cluster network.
        enable_single_replica_ckpt_restoring: bool = False
        # Defaults to the `data` dimension
        replica_axis_index: int = 1
        # The step to save may already exist in a incomplete state. This option
        # controls whether to skip that saving or not. The benefit of skipping
        # is to save the time on deleting an incomplete checkpoint folder.
        skip_uncommitted_checkpoint: bool = True
        # Read only the latest checkpoint recorded in the tracker file if set to `True`
        read_latest_checkpoint_from_tracker_file: bool = False

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> List[str]:
        """See `BaseCheckpointer.checkpointer_paths`."""
        return [str(path) for path in ocp.utils.checkpoint_steps_paths(base_dir)]

    @classmethod
    def checkpoint_steps(cls, base_dir) -> list[int]:
        """See `BaseCheckpointer.checkpointer_steps`."""
        return ocp.utils.checkpoint_steps(base_dir)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

        cfg: OrbaxCheckpointer.Config = self.config
        save_policy = cfg.save_policy.instantiate()

        if cfg.enable_single_replica_ckpt_restoring:
            array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
                replica_axis_index=cfg.replica_axis_index,
                primary_replica_id=0,
                array_metadata_store=array_metadata_store_lib.Store(),
                use_replica_parallel=False,
                enable_write_sharding_file=False,
            )
            ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

        # self._eval_summaries will be set in save() and used by save_fn_with_summaries() to decide
        # whether to save at the step.
        #
        # We need this because Orbax only provides (current_step, last_saved_step) as args to the
        # save policy, so we use `self._eval_summaries` to capture evaler summaries.
        #
        # While we can check the eval summaries in `save()`, some Orbax features like
        # save-on-preemption requires the user to call `ocp.CheckpointManager.save()` at every step,
        # even if the verdict from `cfg.save_policy` is negative.
        self._eval_summaries = None

        self._name_format = ocp.step.standard_name_format(
            step_prefix=STEP_PREFIX,
            step_format_fixed_length=STEP_NUM_DIGITS,
        )

        # pylint: disable-next=unused-argument
        def save_fn_with_summaries(step: int, last_saved_step: Optional[int]) -> bool:
            is_save = save_policy(step=step, evaler_summaries=self._eval_summaries)

            if (
                is_save
                and cfg.skip_uncommitted_checkpoint
                and ocp.path.step.build_step_path(cfg.dir, self._name_format, step).exists()
            ):
                logging.warning(
                    (
                        "Step %s exists and will be skipped since"
                        "`skip_uncommitted_checkpoint` is configured to be `True`"
                    ),
                    step,
                )
                return False

            return is_save

        CheckpointManager = (
            _CheckpointManagerWithTrackerFile
            if cfg.read_latest_checkpoint_from_tracker_file
            else ocp.CheckpointManager
        )
        self._manager = CheckpointManager(
            directory=cfg.dir,
            options=ocp.CheckpointManagerOptions(
                create=True,
                max_to_keep=cfg.keep_last_n,
                keep_period=cfg.keep_period,
                enable_async_checkpointing=True,
                step_name_format=self._name_format,
                should_save_fn=save_fn_with_summaries,
                enable_background_delete=True,
                async_options=ocp.options.AsyncOptions(
                    timeout_secs=cfg.async_timeout_secs,
                    create_directories_asynchronously=False,
                ),
                # Explicitly wrapped in `_ShouldSaveFnPolicy`, otherwise
                # `PreemptionCheckpointingPolicy` is auto injected
                save_decision_policy=_ShouldSaveFnPolicy(save_fn_with_summaries),
                lightweight_initialize=True,
                cleanup_tmp_directories=True,
            ),
            item_handlers={
                # NOTE: we make a relatively weak assumption that index files are JSON serialized
                # for simplicity. The test cases ensure that this is compatible with
                # `read_index_file`.
                "index": ocp.JsonCheckpointHandler(filename="index"),
                # Note that this defaults to use_ocdb=True. Note also that custom `TypeHandler`s are
                # ignored by `StandardCheckpointHandler`, so we use `PyTreeCheckpointHandler`.
                "state": ocp.PyTreeCheckpointHandler(
                    save_concurrent_gb=cfg.max_concurrent_save_gb,
                    restore_concurrent_gb=cfg.max_concurrent_restore_gb,
                ),
            },
        )

    def _get_spec(self, *, step: int, state: Nested[Any]) -> Nested[Any]:
        spec = {"index": [("step", step)]}
        for path, value in utils.flatten_items(state):
            if isinstance(value, (Tensor, TensorSpec)):
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec["index"].append(
                    (path, {"dtype": str(dtype), "shape": str(tuple(value.shape))})
                )
            elif isinstance(value, (tf.data.Iterator, PythonSavable)):
                spec["index"].append((path, str(type(value))))
            else:
                spec["index"].append((path, value))
        return spec

    # pylint: disable-next=redefined-builtin
    def ckpt_dir(self, step: int, dir: Optional[str] = None) -> str:
        """Obtains the checkpoint dir for the given step."""
        if dir is None:
            dir = self._manager.directory
        return str(ocp.step.build_step_path(dir, self._name_format, step))

    def save(
        self,
        *,
        step: int,
        state: Nested[Tensor],
        evaler_summaries: Optional[Dict[str, Any]] = None,
    ):
        """See `BaseCheckpointer.save` for details.

        Checkpoint saving is handled by `orbax` checkpoint manager.
        """
        spec = self._get_spec(step=step, state=state)
        assert self._eval_summaries is None, self._eval_summaries
        self._eval_summaries = copy.deepcopy(evaler_summaries or {})

        try:
            # Note that save() waits for prior serialization to finish.
            self._manager.save(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(spec["index"]),
                    # TODO(markblee): Investigate save_args for chunk_byte_size and
                    # ocdbt_target_data_file_size:
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#custom-chunk-sizes
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#customizing-data-file-size
                    state=ocp.args.PyTreeSave(item=state),
                ),
            )
            # Exit early after pre-emption, equivalent to sys.exit():
            # https://orbax.readthedocs.io/en/latest/preemption_checkpointing.html
            if self._manager.reached_preemption(step):
                self._manager.wait_until_finished()
                raise SystemExit(f"Exiting after saving checkpoint at {step=} due to pre-emption.")
        finally:
            self._eval_summaries = None

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """See `BaseCheckpointer.restore` for details."""

        cfg: OrbaxCheckpointer.Config = self.config

        def _restore_args(x: Any) -> ocp.RestoreArgs:
            if isinstance(x, (Tensor, TensorSpec)):
                arg = ocp.checkpoint_utils.construct_restore_args(
                    jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
                )
                if cfg.enable_single_replica_ckpt_restoring and isinstance(
                    arg, ocp.type_handlers.ArrayRestoreArgs
                ):
                    mesh = x.sharding.mesh
                    arg = ocp.type_handlers.SingleReplicaArrayRestoreArgs(
                        restore_type=arg.restore_type,
                        dtype=arg.dtype,
                        mesh=arg.mesh,
                        mesh_axes=arg.mesh_axes,
                        sharding=arg.sharding,
                        global_shape=arg.global_shape,
                        shape=arg.shape,
                        strict=arg.strict,
                        single_replica_sharding=jax.sharding.NamedSharding(
                            jax.sharding.Mesh(
                                _replica_devices(mesh.devices, cfg.replica_axis_index),
                                mesh.axis_names,
                            ),
                            x.sharding.spec,
                        ),
                    )
                return arg
            elif isinstance(x, tf.data.Iterator):
                return _TfIteratorHandler.RestoreArgs(item=x)
            elif _GRAIN_INSTALLED and isinstance(x, _GrainIterator):
                return _GrainDatasetIteratorHandler.RestoreArgs(item=x)
            else:
                return None

        restore_args = jax.tree.map(_restore_args, state)

        try:
            composite_state = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    index=ocp.args.JsonRestore(None),
                    state=ocp.args.PyTreeRestore(item=state, restore_args=restore_args),
                ),
            )
        except FileNotFoundError as e:
            # Orbax raises FileNotFoundError if there are no checkpoints.
            if step is not None:
                raise ValueError(f"Failed to restore at step {step}.") from e
            logging.info("Could not find any completed checkpoints under %s: %s", cfg.dir, e)
            return None, state  # Return the input state.

        restored_index = composite_state["index"]
        restored_state = composite_state["state"]

        # If we successfully restored from step=None, use the restored step.
        if step is None:
            for k, v in restored_index:
                if k == "step":
                    step = v
                    break

        # Validate ckpt structure.
        check_state_structure(
            restored_index,
            target_structure=self._get_spec(step=step, state=state)["index"],
            validation=cfg.validation_type,
        )
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._manager.wait_until_finished()

    def stop(self, *, has_exception: bool = False):
        """See `BaseCheckpointer.stop` for details."""
        self._manager.close()


# Below are adapted from:
# https://github.com/AI-Hypercomputer/maxtext/blob/3d9378d77759a7756e20ae2940ce71dcaa17ef13/src/MaxText/checkpointing.py#L333-L356


def _find_idx(array: np.ndarray, replica_axis_idx: int) -> int:
    """Returns the index along given dimension that the current host belongs to."""
    idx = None
    for idx, val in np.ndenumerate(array):
        if val.process_index == jax.process_index():
            break
    return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int) -> np.ndarray:
    """Returns the devices from the replica that current host belongs to.

    Replicas are assumed to be restricted to the first axis.

    Args:
      device_array: devices of the mesh that can be obtained by mesh.devices()
      replica_axis_idx: axis dimension along which replica is taken

    Returns:
      devices inside the replica that current host is in
    """
    idx = _find_idx(device_array, replica_axis_idx)
    replica_result = np.take(device_array, idx, axis=replica_axis_idx)
    return np.expand_dims(replica_result, axis=replica_axis_idx)
