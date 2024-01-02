# Copyright © 2023 Apple Inc.

"""A simple checkpointer.

Checkpointer uses jax.experimental.array_serialization as the storage layer and provides:
(1) additional guards on top of the storage layers to verify that dtypes and shapes match those of
    the model parameters.
(2) checkpoint garbage collection.
(3) global synchronization across processes to ensure that a checkpoint directory is visible only
    after all processes have completed saving the checkpoint.
"""
import dataclasses
import difflib
import enum
import json
import os.path
import threading
import time
from concurrent import futures
from types import TracebackType
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple, Type, Union

import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import logging
from jax.experimental import maps, multihost_utils
from jax.experimental.array_serialization import serialization as array_serialization

from axlearn.common import utils
from axlearn.common.config import (
    REQUIRED,
    Configurable,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import (
    InvocationContext,
    Module,
    clone_context_stack,
    install_context_stack,
)
from axlearn.common.utils import NestedTensor, NestedTensorSpec, Tensor, TensorSpec, set_recursively


class CheckpointValidationType(str, enum.Enum):
    """Represents a type of checkpoint validation.

    Values:
    - EXACT: Entire checkpoint should match the entire state exactly.
    - EXACT_UP_TO_DTYPE: Entire checkpoint should match entire state exactly,
        except for any dtypes declared.
    - CONTAINS_STATE: Checkpoint exactly matches at least a subset of the state.
    - CONTAINS_STATE_UP_TO_DTYPE: Checkpoint matches at least a subset of the state,
        except for any dtypes declared.
    """

    # TODO(tom_gunter): Factorize validation specs into (at a minimum)
    #   {exact, subset} x {with-dtype, without-dtype}.

    EXACT = "EXACT"
    EXACT_UP_TO_DTYPE = "EXACT_UP_TO_DTYPE"
    CONTAINS_STATE = "CONTAINS_STATE"
    CONTAINS_STATE_UP_TO_DTYPE = "CONTAINS_STATE_UP_TO_DTYPE"


def parse_step_from_dir(step_dir: str) -> int:
    return int(step_dir[-8:])


def checkpoint_paths(base_dir: str) -> List[str]:
    """Returns complete checkpoint paths under base dir."""
    index_paths = tf.io.gfile.glob(os.path.join(base_dir, "step_*", "index"))  # type: ignore
    return [os.path.dirname(path) for path in index_paths]


def latest_checkpoint_path(base_dir: str) -> str:
    """Returns the most recent (highest step count) complete checkpoint under base dir.

    Args:
        base_dir: Path to checkpoints dir.

    Returns:
        The path to the checkpoint directory under base_dir with the highest step count.
        The checkpoint is guaranteed to be complete.
    """
    # Note: checkpoint_paths already filters incomplete checkpoints.
    return sorted(checkpoint_paths(base_dir)).pop()


def check_state_structure(
    ckpt_structure: List[Tuple[str, Any]],
    target_structure: List[Tuple[str, Any]],
    validation: CheckpointValidationType = CheckpointValidationType.EXACT,
):
    # Maybe filter structure before comparison.
    def filter_for_validation(structure):
        filtered_structure = []
        for key, value in structure:
            if validation in [
                CheckpointValidationType.EXACT_UP_TO_DTYPE,
                CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE,
            ] and isinstance(value, dict):
                # Drop dtype if it's in the value.
                value = {k: v for k, v in value.items() if k != "dtype"}
            filtered_structure.append((key, value))
        return filtered_structure

    filtered_ckpt_structure = sorted(
        [f"{key}={value}" for key, value in filter_for_validation(ckpt_structure)]
    )
    filtered_target_structure = sorted(
        [f"{key}={value}" for key, value in filter_for_validation(target_structure)]
    )

    msg = ""
    if validation in [CheckpointValidationType.EXACT, CheckpointValidationType.EXACT_UP_TO_DTYPE]:
        is_compatible = filtered_ckpt_structure == filtered_target_structure
    elif validation in [
        CheckpointValidationType.CONTAINS_STATE,
        CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE,
    ]:
        # Allow checkpoint to contain additional information.
        is_compatible = set(filtered_ckpt_structure) >= set(filtered_target_structure)
        if not is_compatible:
            msg = (
                "Missing:\n"
                + "\n".join(sorted(set(filtered_target_structure) - set(filtered_ckpt_structure)))
                + "\n"
            )
    else:
        raise ValueError(f"Unknown validation type: {validation}")
    if not is_compatible:
        msg += "Diff:\n" + "\n".join(
            difflib.ndiff(sorted(filtered_ckpt_structure), sorted(filtered_target_structure))
        )
        raise ValueError(
            f"Unable to restore checkpoint ({validation}). A mismatch between the saved "
            "checkpoint tree dtypes or shapes and the current one has been detected:\n"
            f"{msg}"
        )


def _cleanup_checkpoint(ckpt_dir: str):
    """Removes ckpt_dir if it exists."""
    if jax.process_index() == 0:
        if tf.io.gfile.exists(ckpt_dir):
            tf.io.gfile.rmtree(ckpt_dir)
    # Wait for cleanup to complete.
    multihost_utils.sync_global_devices(f"{ckpt_dir}_cleanup")


def _validate_checkpoint(ckpt_dir: str):
    """Ensures a checkpoint is complete, i.e. `ckpt_dir/index` exists.

    Args:
        ckpt_dir: Directory of a checkpoint at a specific step.

    Raises:
        ValueError: If the checkpoint is not complete.
    """
    ckpt_index = os.path.join(ckpt_dir, "index")
    if not tf.io.gfile.exists(ckpt_index):
        raise ValueError(
            f"Checkpoint {ckpt_dir} is incomplete -- expected {ckpt_index} to be present."
        )


# pylint: disable-next=redefined-builtin
def save_tf_savables(value_map: Dict[str, Any], *, dir: str):
    """Saves TF savables from `value_map` into `dir`."""
    for path, value in value_map.items():
        tf_checkpoint = tf.train.Checkpoint(value)
        tf_checkpoint.write(os.path.join(dir, path))


# pylint: disable-next=redefined-builtin
def restore_tf_savables(value_map: Dict[str, Any], *, dir: str):
    """Restores TF savables from `dir` into `value_map` in-place."""
    for path, value in value_map.items():
        tf_checkpoint = tf.train.Checkpoint(value)
        tf_checkpoint.read(os.path.join(dir, path))


class StateStorageCommitCallback(Protocol):
    """StateStorage commit callback protocol."""

    def __call__(self, *, ckpt_dir: str, index: Any):
        """Commits a checkpoint with the given directory and index."""


class StateStorage(Configurable):
    """Base StateStorage."""

    def save_to_dir(
        self,
        *,
        step: int,
        state: NestedTensor,
        ckpt_dir: str,
        on_commit_callback: StateStorageCommitCallback,
    ):
        """Starts a save to the given directories.

        The writes may happen in the background and not finish until wait_until_finished().
        Storage implementations should invoke on_commit_callback(ckpt_dir, commit_data) when async
        write has completed.
        """
        raise NotImplementedError(type(self))

    def wait_until_finished(self):
        """Waits for async writes to finish."""
        raise NotImplementedError(type(self))

    def restore_from_dir(
        self,
        step: int,
        state: Union[NestedTensor, NestedTensorSpec],
        *,
        ckpt_dir: str,
    ) -> NestedTensor:
        raise NotImplementedError(type(self))


def write_index_file(*, ckpt_dir: str, index: Any):
    """An on_commit_callback that writes an index file to ckpt_dir."""
    index_path = os.path.join(ckpt_dir, "index")
    logging.info("Writing index file to %s", index_path)
    with tf.io.gfile.GFile(index_path, "w") as f:
        f.write(json.dumps(index))


def _parse_tensor_spec(spec_dict: Dict[str, str]) -> TensorSpec:
    # The shape string is of format `(dim...)`. [1:-1] removes the parentheses.
    shape = [int(x) for x in spec_dict["shape"][1:-1].split(",") if x]
    dtype_str = spec_dict["dtype"]
    dtype_dict = {
        str(dtype.dtype): dtype
        for dtype in (
            jnp.bool_,
            jnp.bfloat16,
            jnp.float16,
            jnp.float32,
            jnp.float64,
            jnp.int8,
            jnp.int16,
            jnp.int32,
            jnp.int64,
            jnp.uint32,
        )
    }
    if dtype_str not in dtype_dict:
        raise NotImplementedError(
            f"Cannot convert {dtype_str} to jnp.dtype. Possible values are {dtype_dict.keys()}"
        )
    return TensorSpec(shape=tuple(shape), dtype=dtype_dict[dtype_str])


def read_state_spec(ckpt_dir: str) -> NestedTensorSpec:
    """Reads TensorSpecs from the given checkpoint dir.

    Args:
        ckpt_dir: The checkpoint directory corresponding to a specific step, e.g., the directory
            returned by `latest_checkpoint_path(checkpointer.config.dir)`.

    Returns:
        A NestedTensorSpec representing the tensors stored under `ckpt_dir`. Each TensorSpec
        should have `shape` and `dtype` filled in, but will not contain `mesh_axes`. The returned
        NestedTensorSpec can be passed as `state` to Checkpointer.restore().

        If a checkpoint is too large to load onto a single host, the caller can further specify
        `mesh_axes` of the TensorSpecs to load the checkpoint across multiple processes.
    """
    with tf.io.gfile.GFile(os.path.join(ckpt_dir, "index"), "r") as f:
        restored_index_entries = json.loads(f.read())
        state = {}
        for path, value in restored_index_entries:
            if isinstance(value, dict):
                set_recursively(state, value=_parse_tensor_spec(value), path=path, separator="/")
            else:
                # Ignore step or tf.data.Iterator.
                logging.vlog(1, "read_index_file ignores %s", path)
        return state


class TensorStoreStateStorage(StateStorage):
    """A StateStorage implementation using TensorStore."""

    @config_class
    class Config(StateStorage.Config):
        """Configures TensorStoreStateStorage."""

        timeout_secs: float = 3600

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._manager = array_serialization.GlobalAsyncCheckpointManager(cfg.timeout_secs)

    @dataclasses.dataclass
    class CheckpointSpec:  # pylint: disable=too-many-instance-attributes
        index: List[Tuple[str, Any]]
        storage_paths: List[str]
        tensorstore_specs: List[Dict]
        shapes: List[Any]
        dtypes: List[jnp.dtype]
        shardings: List[jax.sharding.Sharding]
        gda_values: List[Tensor]
        tf_ckpt_map: Dict[str, Any]

    def _spec_from_path(self, ckpt_path: str):
        return array_serialization.get_tensorstore_spec(ckpt_path)

    def _get_spec(self, step: int, state: NestedTensor, ckpt_dir: str) -> CheckpointSpec:
        spec = self.CheckpointSpec(
            index=[],
            storage_paths=[],
            tensorstore_specs=[],
            shapes=[],
            dtypes=[],
            shardings=[],
            gda_values=[],
            tf_ckpt_map={},
        )

        mesh = maps.thread_resources.env.physical_mesh
        if not mesh.shape:
            raise RuntimeError(
                "Checkpoint restoration must take place within the context of a Mesh"
            )
        spec.index = [("step", step)]
        for path, value in utils.flatten_items(state, separator="/"):
            if isinstance(value, (Tensor, TensorSpec)):
                logging.vlog(
                    3, "Adding array value %s %s(%s)", type(value), value.dtype, value.shape
                )
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec.index.append((path, {"dtype": str(dtype), "shape": str(tuple(value.shape))}))
                gda_path = os.path.join(ckpt_dir, "gda", path)
                spec.storage_paths.append(gda_path)
                spec.tensorstore_specs.append(self._spec_from_path(gda_path))
                spec.shapes.append(value.shape)
                spec.dtypes.append(dtype)
                if isinstance(value, Tensor):
                    spec.gda_values.append(value)
                    spec.shardings.append(value.sharding)
                else:
                    spec.shardings.append(jax.sharding.NamedSharding(mesh, value.mesh_axes))
            elif isinstance(value, tf.data.Iterator):
                logging.vlog(3, "Adding value (%s) to tf_ckpt_map", value)
                spec.index.append((path, str(type(value))))
                spec.tf_ckpt_map[path] = value
            else:
                logging.vlog(3, "Adding value (%s) to index", value)
                spec.index.append((path, value))
        return spec

    def save_to_dir(
        self,
        *,
        step: int,
        state: NestedTensor,
        ckpt_dir: str,
        on_commit_callback: StateStorageCommitCallback = write_index_file,
    ):
        # We write data files directly to `ckpt_dir`. `index` is written into `ckpt_dir` in
        # `on_commit_callback` to finalize the checkpoint.
        spec = self._get_spec(step, state, ckpt_dir)
        if jax.process_index() == 0:
            if not ckpt_dir.startswith("gs://"):
                storage_dirs = sorted(
                    list(set(os.path.dirname(path) for path in spec.storage_paths))
                )
                logging.info("Creating directories: %s", storage_dirs)
                with futures.ThreadPoolExecutor() as executor:
                    executor.map(tf.io.gfile.makedirs, storage_dirs)  # pytype: disable=module-attr
                logging.info("All directories created")
        # Wait for directory and index creation.
        multihost_utils.sync_global_devices(ckpt_dir)
        # Each worker writes its tf checkpoints under a different path.
        save_tf_savables(spec.tf_ckpt_map, dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}"))
        # Run serialization of GDA values in parallel.
        logging.info(
            "array_values=%s tensorstore=%s", utils.shapes(spec.gda_values), spec.tensorstore_specs
        )
        self._manager.serialize(
            spec.gda_values,
            spec.tensorstore_specs,
            on_commit_callback=lambda: on_commit_callback(ckpt_dir=ckpt_dir, index=spec.index),
        )
        logging.info("GlobalAsyncCheckpointManager.serialize done")

    def wait_until_finished(self):
        self._manager.wait_until_finished()

    def restore_from_dir(
        self,
        step: int,
        state: Union[NestedTensor, NestedTensorSpec],
        *,
        ckpt_dir: str,
        validation: CheckpointValidationType = CheckpointValidationType.EXACT,
        concurrent_gb: int = 32,
    ) -> NestedTensor:
        spec = self._get_spec(step, state, ckpt_dir)
        logging.info("Restoring checkpoint from directory %s", ckpt_dir)
        with tf.io.gfile.GFile(os.path.join(ckpt_dir, "index"), "r") as f:
            restored_index_entries = json.loads(f.read())
        check_state_structure(
            restored_index_entries, target_structure=spec.index, validation=validation
        )
        restore_tf_savables(
            spec.tf_ckpt_map, dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}")
        )

        restored_gda_values = array_serialization.run_deserialization(
            shardings=spec.shardings,
            tensorstore_specs=spec.tensorstore_specs,
            global_shapes=spec.shapes,
            dtypes=spec.dtypes,
            concurrent_gb=concurrent_gb,
        )
        state_leaves = []
        for path, value in spec.index:
            if path == "step":
                pass
            elif path in spec.tf_ckpt_map:
                state_leaves.append(spec.tf_ckpt_map[path])
            elif isinstance(value, dict):
                state_leaves.append(restored_gda_values.pop(0))
            else:
                raise RuntimeError(f"Unknown index entry '{value}'")

        restored_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(state), state_leaves
        )
        multihost_utils.sync_global_devices(ckpt_dir)
        return restored_state


class CheckpointPolicy(Protocol):
    """Decides whether checkpointer should save at the given step."""

    def __call__(self, *, step: int, evaler_summaries: Dict[str, Any]) -> bool:
        """Implements the policy.

        Args:
            step: Current step.
            evaler_summaries: A mapping from evaler name to eval summaries.
                Note that some of the values in `evaler_summaries` can be None, if the evaler did
                not run that step. In this case, the evaler name will still appear as a key and
                `evaler_summaries` itself will not be None.

        Returns:
            True iff we should save at the current step.
        """
        raise NotImplementedError(type(self))


class EvalMetric(NamedTuple):
    """Tuple used to fetch a metric from evaler_summaries dict.

    Usage: evaler_summaries[evaler_name][metric_name]
    """

    evaler_name: str
    metric_name: str


class BestMetricPolicy(Configurable):
    """A CheckpointPolicy that saves checkpoint only when there is a better eval metric.

    The monitored eval metric must be a scalar.

    Note that if the evaler did not run current step, evaler name will still appear as a key in
    evaler_summaries but the evalue will be None. The policy will not save the checkpoint.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures BestMetricPolicy."""

        # Evaler and metric name in evaler_summaries dict.
        metric: Required[EvalMetric] = REQUIRED
        # Mode when comparing metrics.
        # When "max", save checkpoint when there is a new higher metric.
        # When "min", save checkpoint when there is a new lower metric.
        mode: Optional[str] = "max"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.best_metric: Optional[Tensor] = None

    def __call__(self, *, step: int, evaler_summaries: Dict[str, Any]) -> bool:
        cfg = self.config
        evaler_name, metric_name = cfg.metric

        if evaler_summaries == {}:
            raise ValueError("evaler_summaries is empty!")
        if evaler_name not in evaler_summaries:
            raise ValueError(f"{evaler_name} not found in evaler_summaries!")
        if cfg.mode not in {"max", "min"}:
            raise ValueError(f"Unsupported mode {cfg.mode}!")

        if evaler_summaries[evaler_name] is None:
            return False

        if metric_name not in evaler_summaries[evaler_name]:
            raise ValueError(f'{metric_name} not in evaler_summaries["{evaler_name}"]!')
        if evaler_summaries[evaler_name][metric_name] is None:
            raise ValueError(f'evaler_summaries["{evaler_name}"]["{metric_name}"] is None!')

        metric = evaler_summaries[evaler_name][metric_name]
        if isinstance(metric, WeightedScalar):
            metric = metric.mean

        if metric.shape != ():
            raise ValueError("Monitored metric must be a scalar!")

        logging.info(
            f"Comparing metric: New: {float(metric):.5f} Before: "
            + (
                f"{self.best_metric}"
                if self.best_metric is None
                else f"{float(self.best_metric):.5f}"
            )
        )

        if (
            self.best_metric is None
            or (cfg.mode == "max" and metric > self.best_metric)
            or (cfg.mode == "min" and metric < self.best_metric)
        ):
            logging.info("Found new best metric")
            self.best_metric = metric
            return True
        else:
            logging.info("No new best metric")
            return False


def every_n_steps_policy(n: int = 1, *, min_step: int = 1) -> CheckpointPolicy:
    """Checkpoints every n steps, but not before `min_step`."""

    def fn(*, step: int, evaler_summaries: Dict[str, Any]) -> bool:
        del evaler_summaries
        return step >= min_step and step % n == 0

    return fn


def every_n_steps_and_last_policy(
    n: int = 1, *, min_step: int = 1, max_step: int
) -> CheckpointPolicy:
    """Checkpoints every n steps, but not before `min_step`,
    and at the last training iteration `max_step`.

    Args:
        n: The checkpointing frequency. Checkpointing will be triggered every `n` steps
        min_step: The minimum step to start checkpointing.
        max_step: The maximum number of training steps.
            Checkpointing will be triggered at step `max_step`.
    """
    every_n_steps_fn = every_n_steps_policy(n=n, min_step=min_step)

    def fn(*, step: int, evaler_summaries: Dict[str, Any]) -> bool:
        return every_n_steps_fn(step=step, evaler_summaries=evaler_summaries) or step == max_step

    return fn


class Checkpointer(Module):
    """A checkpointer that supports various StateStorage implementations."""

    @config_class
    class Config(Module.Config):
        """Configures Checkpointer."""

        dir: Required[str] = REQUIRED  # The output directory.
        keep_last_n: int = 1  # Keeps this many past ckpts.
        # If > 0, keeps at least one checkpoint every N steps.
        keep_every_n_steps: Optional[int] = None
        # Interval between garbage collection runs.
        gc_loop_interval_seconds: float = 60
        # A config that instantiates to a CheckpointPolicy.
        save_policy: InstantiableConfig = config_for_function(every_n_steps_policy)
        # A config that instantiates to a StateStorage.
        storage: StateStorage.Config = TensorStoreStateStorage.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._storage: StateStorage = cfg.storage.instantiate()
        self._gc_stopping = None
        self._gc_thread = None
        self._within_context = False
        self._save_policy: CheckpointPolicy = cfg.save_policy.instantiate()

    def __enter__(self):
        if self._within_context:
            raise ValueError("Already in a context.")
        self._within_context = True
        self.start_gc_thread()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # Note: returning None here lets the caller handle the exception, if any.
        self.stop()
        self._within_context = False

    def start_gc_thread(self):
        if self._gc_thread is None and jax.process_index() == 0:
            self._gc_stopping = threading.Event()
            self._gc_thread = threading.Thread(
                name=f"{self.path()}.gc_loop",
                target=self._gc_loop,
                kwargs=dict(context_stack=clone_context_stack()),
            )
            self._gc_thread.start()

    def stop(self):
        """Stops the checkpointer. Waits for async writes and garbage collection loop to finish."""
        self.wait_until_finished()
        logging.info("Waiting for gc_thread to finish")
        if self._gc_thread is not None:
            self._gc_stopping.set()
            self._gc_thread.join()
            self._gc_thread = None
            logging.info("gc_thread finished")

    def _gc_loop(self, *, context_stack: List[InvocationContext]):
        cfg = self.config
        install_context_stack(context_stack)
        while True:
            if self._gc_stopping.wait(timeout=cfg.gc_loop_interval_seconds):
                break
            self.run_garbage_collection()
        logging.info("GC loop done")

    def ckpt_dir(self, step: int) -> str:
        cfg = self.config
        return os.path.join(cfg.dir, f"step_{step:08d}")

    def save(
        self, *, step: int, state: NestedTensor, evaler_summaries: Optional[Dict[str, Any]] = None
    ):
        """Saves `state` at the given `step` according to the configured checkpoint policy."""
        if not self._save_policy(step=step, evaler_summaries=(evaler_summaries or {})):
            return
        if step < 0 or step >= 10**8:
            raise ValueError(f"Out-of-range: {step}")
        ckpt_dir = self.ckpt_dir(step)
        start_time = time.perf_counter()
        _cleanup_checkpoint(ckpt_dir)
        self._storage.save_to_dir(
            step=step, state=state, ckpt_dir=ckpt_dir, on_commit_callback=write_index_file
        )
        end_time = time.perf_counter()
        logging.info(
            "Saved checkpoint with %s in %s seconds", type(self._storage), end_time - start_time
        )

    def run_garbage_collection(self):
        """Runs one round of garbage collection of past checkpoints."""
        cfg = self.config
        # Garbage collection.
        dirs = sorted(tf.io.gfile.glob(os.path.join(cfg.dir, "step_*")))  # type: ignore
        last_kept_step = float("-inf")
        remaining_dirs = []
        for saved_dir in dirs[: -cfg.keep_last_n]:
            saved_step = int(saved_dir[-8:])
            if cfg.keep_every_n_steps and saved_step - last_kept_step >= cfg.keep_every_n_steps:
                logging.vlog(
                    2, "Keeping %s >= %s + %s", saved_dir, last_kept_step, cfg.keep_every_n_steps
                )
                remaining_dirs.append(saved_dir)
                last_kept_step = saved_step
            else:
                logging.info("Removing %s", saved_dir)
                try:
                    tf.io.gfile.rmtree(saved_dir)  # type: ignore
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning("Ignoring error in removing %s: %s.", saved_dir, e)
        remaining_dirs += dirs[-cfg.keep_last_n :]
        logging.log_every_n_seconds(
            logging.INFO,
            "Garbage collection done on %s. Remaining=%s",
            3600,
            cfg.dir,
            remaining_dirs,
        )

    def wait_until_finished(self):
        """Waits for pending asynchronous saves to finish."""
        self._storage.wait_until_finished()

    def _validate_and_restore(
        self, *, step: int, state: NestedTensor, ckpt_dir: str
    ) -> NestedTensor:
        """Validates a checkpoint is not incomplete and then restores it."""
        _validate_checkpoint(ckpt_dir)
        return self._storage.restore_from_dir(step=step, state=state, ckpt_dir=ckpt_dir)

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[NestedTensor, NestedTensorSpec],
    ) -> Tuple[Optional[int], NestedTensor]:
        """Restores from the checkpoint directory.

        Args:
            step: If None, restores from the latest complete checkpoint. Otherwise from the
                specified step. A complete checkpoint is one with an "index" file, which is only
                written after the entire checkpoint has been written.
            state: Ensures that the restored state have the same structure, dtypes, and shapes as
                `state`.

        Returns:
            (restored_step, restored_checkpoint_state).
            If no complete checkpoint is found, returns None as restored_step and the input `state`
            as restored_checkpoint_state.
        """
        cfg = self.config
        if step is not None:
            # For a specified step, we try to load it.
            return step, self._validate_and_restore(
                step=step, state=state, ckpt_dir=self.ckpt_dir(step)
            )
        try:
            # Latest checkpoint path, if it exists, is guaranteed to be complete.
            ckpt_dir = latest_checkpoint_path(cfg.dir)
            step = parse_step_from_dir(ckpt_dir)
            restored_state = self._validate_and_restore(step=step, state=state, ckpt_dir=ckpt_dir)
            logging.info("Restored state from ckpt at step %s", step)
        except IndexError:
            # No checkpoint path exists. Return with input state.
            logging.info("Could not find any completed checkpoints under %s", cfg.dir)
            restored_state = state
        return step, restored_state
