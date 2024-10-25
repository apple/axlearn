# Copyright © 2023 Apple Inc.

"""Checkpointing utilities."""

import dataclasses
import difflib
import enum
import json
import os.path
import tempfile
import threading
import time
from concurrent import futures
from types import TracebackType
from typing import Any, NamedTuple, Optional, Protocol, TypeAlias, Union

import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import logging
from jax._src.mesh import thread_resources
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serialization

from axlearn.common import file_system as fs
from axlearn.common import utils
from axlearn.common.array_serialization import (
    BoundedDataShardedAsyncCheckpointManager,
    GlobalAsyncCheckpointManager,
)
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
from axlearn.common.summary_writer import CheckpointerAction, SummaryWriter
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    NestedTensorSpec,
    Tensor,
    TensorSpec,
    set_recursively,
)

try:
    import grain.python as grain

    _GrainIterator: TypeAlias = Union[grain.DatasetIterator, grain.PyGrainDatasetIterator]
    _GRAIN_INSTALLED = True
except ImportError:
    logging.warning("grain is not installed. Will not be able to checkpoint grain iterators.")
    _GRAIN_INSTALLED = False

# Number of digits in the step directory.
STEP_NUM_DIGITS = 8
# Prefix for step directory.
STEP_PREFIX = "step"


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
    # TODO(markblee): use regex.
    return int(step_dir[-STEP_NUM_DIGITS:])


def check_state_structure(
    ckpt_structure: list[tuple[str, Any]],
    target_structure: list[tuple[str, Any]],
    *,
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


def _upload_dir(src_dir_handle: tempfile.TemporaryDirectory, *, dst_dir: str):
    """Upload a directory (non-recursively) from a temporary dir to dst_dir.

    Temporary dir will be deleted after the upload is complete.
    """
    src_dir = src_dir_handle.name
    fs.makedirs(dst_dir)
    for item in fs.listdir(src_dir):
        src_file = os.path.join(src_dir, item)
        dst_file = os.path.join(dst_dir, item)
        assert not fs.isdir(src_file)
        fs.copy(src_file, dst_file, overwrite=True)
    src_dir_handle.cleanup()


# pylint: disable=redefined-builtin
def async_save_tf_savables(
    value_map: Nested[Any], *, executor: futures.ThreadPoolExecutor, dir: str
) -> futures.Future:
    """Asynchronously saves TF savables from `value_map` into `dir`.

    When this call returns, `value_map` can be safely mutated, but saving to `dir` will not
    complete unless the returned future is set.
    """
    # pylint: disable-next=consider-using-with
    f = tempfile.TemporaryDirectory()
    for path, value in utils.flatten_items(value_map):
        tf_checkpoint = tf.train.Checkpoint(value)
        tf_checkpoint.write(os.path.join(f.name, path))
    return executor.submit(_upload_dir, f, dst_dir=dir)


# pylint: disable-next=redefined-builtin
def restore_tf_savables(value_map: Nested[Any], *, dir: str) -> Nested[Any]:
    """Restores TF savables from `dir` into `value_map` in-place."""

    for path, value in utils.flatten_items(value_map):
        tf_checkpoint = tf.train.Checkpoint(value)
        tf_checkpoint.read(os.path.join(dir, path))

    return value_map


# pylint: disable-next=redefined-builtin
def maybe_save_grain_savables(value_map: Nested[Any], *, dir: str):
    """Saves grain savables from `value_map` into `dir`.

    Is a no-op if grain is not installed.
    """
    if not _GRAIN_INSTALLED:
        return
    for path, value in utils.flatten_items(value_map):
        if not callable(getattr(value, "get_state", None)):
            continue
        state = value.get_state()
        if isinstance(state, bytes):
            state = state.decode("utf-8")
        dst = os.path.join(dir, path)
        fs.makedirs(os.path.dirname(dst))
        with fs.open(dst, "w") as f:
            json.dump(state, f, indent=4)


# pylint: disable-next=redefined-builtin
def maybe_restore_grain_savables(value_map: Nested[Any], *, dir: str) -> Nested[Any]:
    """Restores grain savables from `dir` into `value_map`.

    Is a no-op if grain is not installed.
    """
    if not _GRAIN_INSTALLED:
        return
    for path, value in utils.flatten_items(value_map):
        if not callable(getattr(value, "set_state", None)):
            continue
        with fs.open(os.path.join(dir, path), "rb") as f:
            state = f.read()
        if isinstance(value, grain.DatasetIterator):
            if isinstance(state, bytes):
                state = state.decode("utf-8")
            state = json.loads(state)
        value.set_state(state)

    return value_map


# pylint: enable=redefined-builtin
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

    def stop(self):
        """Stops and disposes resources."""
        raise NotImplementedError(type(self))


def write_index_file(*, ckpt_dir: str, index: Any):
    """An on_commit_callback that writes an index file to ckpt_dir."""
    index_path = os.path.join(ckpt_dir, "index")
    logging.info("Writing index file to %s", index_path)
    with fs.open(index_path, "w") as f:
        f.write(json.dumps(index))


def read_index_file(ckpt_dir: str) -> Nested[Any]:
    """Reads index files written with `write_index_file`."""
    with fs.open(os.path.join(ckpt_dir, "index"), "r") as f:
        return json.loads(f.read())


def _parse_tensor_spec(spec_dict: dict[str, str]) -> TensorSpec:
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
            returned by `<BaseCheckpointer>.latest_checkpoint_path(checkpointer.config.dir)`, where
            `<BaseCheckpointer>` is a subclass of `BaseCheckpointer`.

    Returns:
        A NestedTensorSpec representing the tensors stored under `ckpt_dir`. Each TensorSpec
        should have `shape` and `dtype` filled in, but will not contain `mesh_axes`. The returned
        NestedTensorSpec can be passed as `state` to `<BaseCheckpointer>.restore()`.

        If a checkpoint is too large to load onto a single host, the caller can further specify
        `mesh_axes` of the TensorSpecs to load the checkpoint across multiple processes.
    """
    state = {}
    # Look for index file under `<base_dir>/<step_dir>` or `<base_dir>/<step_dir>/index/`.
    # TODO(markblee): Move this fn into corresponding checkpointer class instead.
    if fs.isdir(os.path.join(ckpt_dir, "index")):
        ckpt_dir = os.path.join(ckpt_dir, "index")
    for path, value in read_index_file(ckpt_dir):
        if isinstance(value, dict):
            set_recursively(state, value=_parse_tensor_spec(value), path=path, separator="/")
        else:
            # Ignore step or tf.data.Iterator.
            logging.vlog(1, "read_state_spec ignores %s", path)
    return state


class TensorStoreStateStorage(StateStorage):
    """A StateStorage implementation using TensorStore.

    It uses `jax.experimental.array_serialization` and additionally provides:
    (1) Additional guards to verify that dtypes and shapes match those of the model parameters.
    (2) Memory-bounded and data-sharded checkpoint serialization for large-scale training.
    """

    @config_class
    class Config(StateStorage.Config):
        """Configures TensorStoreStateStorage.

        Attributes:
            timeout_secs: Barrier timeout in seconds.
            max_data_shard_degree: Max sharding degree of model weights along data-parallel axis.
                `None` and `1` means no sharding. `-1` means fully shard along data-parallel
                replicas. `>1` means custom sharding degree (currently not implemented).
            max_concurrent_gb: Max concurrent shards (in GB) to write.
            max_concurrent_restore_gb: Max concurrent shards (in GB) to read during checkpoint
                restore. `None` or `0` means using a default value of 32GB.
        """

        timeout_secs: float = 3600
        max_data_shard_degree: Optional[int] = None
        # TODO(hanzhi-zhou): rename this to max_concurrent_save_gb.
        max_concurrent_gb: Optional[int] = None
        max_concurrent_restore_gb: Optional[int] = None

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # TODO(markblee): Consider making BoundedDataShardedAsyncCheckpointManager
        # the default once stable.
        if cfg.max_concurrent_gb is not None or cfg.max_data_shard_degree:
            self._manager = BoundedDataShardedAsyncCheckpointManager(
                max_concurrent_gb=cfg.max_concurrent_gb,
                timeout_secs=cfg.timeout_secs,
                max_data_shard_degree=cfg.max_data_shard_degree,
            )
        else:
            self._manager = GlobalAsyncCheckpointManager(timeout_secs=cfg.timeout_secs)
        if cfg.max_concurrent_restore_gb is not None and cfg.max_concurrent_restore_gb <= 0:
            raise ValueError(
                f"max_concurrent_restore_gb must be strictly positive. "
                f"Got {cfg.max_concurrent_restore_gb}"
            )
        self._max_concurrent_restore_gb = cfg.max_concurrent_restore_gb or 32
        self._executor = futures.ThreadPoolExecutor()

    @dataclasses.dataclass
    class CheckpointSpec:  # pylint: disable=too-many-instance-attributes
        index: list[tuple[str, Any]]
        storage_paths: list[str]
        tensorstore_specs: list[dict]
        shapes: list[Any]
        dtypes: list[jnp.dtype]
        shardings: list[jax.sharding.Sharding]
        gda_values: list[Tensor]
        tf_ckpt_map: dict[str, Any]
        grain_ckpt_map: dict[str, Any]

    def _spec_from_path(self, ckpt_path: str):
        # TODO(markblee): Enable ocdbt driver.
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
            grain_ckpt_map={},
        )

        mesh = thread_resources.env.physical_mesh
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
            elif _GRAIN_INSTALLED and isinstance(value, _GrainIterator):
                spec.index.append((path, str(type(value))))
                spec.grain_ckpt_map[path] = value
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
        start_time = time.perf_counter()
        # We write data files directly to `ckpt_dir`. `index` is written into `ckpt_dir` in
        # `on_commit_callback` to finalize the checkpoint.
        spec = self._get_spec(step, state, ckpt_dir)
        if jax.process_index() == 0:
            if not ckpt_dir.startswith("gs://"):
                dirs = sorted(list(set(os.path.dirname(path) for path in spec.storage_paths)))
                logging.info("Creating directories: %s", dirs)
                list(self._executor.map(fs.makedirs, dirs))
                logging.info("All directories created")
        # Wait for directory and index creation.
        multihost_utils.sync_global_devices(ckpt_dir)
        # Each worker writes its tf checkpoints under a different path.
        save_tf_future = async_save_tf_savables(
            spec.tf_ckpt_map,
            executor=self._executor,
            dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}"),
        )
        maybe_save_grain_savables(
            spec.grain_ckpt_map, dir=os.path.join(ckpt_dir, f"grain_{jax.process_index()}")
        )

        def commit():
            on_commit_callback(ckpt_dir=ckpt_dir, index=spec.index)
            logging.info(
                "Serialization of %s completed in %s seconds.",
                ckpt_dir,
                time.perf_counter() - start_time,
            )

        # Run serialization of GDA values in parallel.
        logging.debug(
            "array_values=%s tensorstore=%s", utils.shapes(spec.gda_values), spec.tensorstore_specs
        )
        self._manager.serialize(
            spec.gda_values,
            spec.tensorstore_specs,
            on_commit_callback=commit,
            additional_futures=[save_tf_future],
        )

    def wait_until_finished(self):
        self._manager.wait_until_finished()

    def restore_from_dir(
        self,
        step: int,
        state: Union[NestedTensor, NestedTensorSpec],
        *,
        ckpt_dir: str,
        validation: CheckpointValidationType = CheckpointValidationType.EXACT,
    ) -> NestedTensor:
        spec = self._get_spec(step, state, ckpt_dir)
        logging.info("Restoring checkpoint from directory %s", ckpt_dir)
        check_state_structure(
            read_index_file(ckpt_dir), target_structure=spec.index, validation=validation
        )
        restore_tf_savables(
            spec.tf_ckpt_map, dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}")
        )
        maybe_restore_grain_savables(
            spec.grain_ckpt_map, dir=os.path.join(ckpt_dir, f"grain_{jax.process_index()}")
        )

        restored_gda_values = self._manager.deserialize(
            shardings=spec.shardings,
            tensorstore_specs=spec.tensorstore_specs,
            global_shapes=spec.shapes,
            dtypes=spec.dtypes,
            concurrent_gb=self._max_concurrent_restore_gb,
        )
        state_leaves = []
        for path, value in spec.index:
            if path == "step":
                pass
            elif path in spec.tf_ckpt_map:
                state_leaves.append(spec.tf_ckpt_map[path])
            elif path in spec.grain_ckpt_map:
                state_leaves.append(spec.grain_ckpt_map[path])
            elif isinstance(value, dict):
                state_leaves.append(restored_gda_values.pop(0))
            else:
                raise RuntimeError(f"Unknown index entry '{value}'")

        restored_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(state), state_leaves
        )
        multihost_utils.sync_global_devices(ckpt_dir)
        return restored_state

    def stop(self):
        self._executor.shutdown(wait=True)


class CheckpointPolicy(Protocol):
    """Decides whether checkpointer should save at the given step."""

    def __call__(self, *, step: int, evaler_summaries: dict[str, Any]) -> bool:
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

    def __call__(self, *, step: int, evaler_summaries: dict[str, Any]) -> bool:
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

    def fn(*, step: int, evaler_summaries: dict[str, Any]) -> bool:
        del evaler_summaries
        return step >= min_step and step % n == 0

    return fn


def every_n_steps_and_last_policy(
    n: int = 1, *, min_step: int = 1, max_step: int
) -> CheckpointPolicy:
    """Checkpoints every n steps, but not before `min_step`, and at the last training iteration
    `max_step`.

    Args:
        n: The checkpointing frequency. Checkpointing will be triggered every `n` steps
        min_step: The minimum step to start checkpointing.
        max_step: The maximum number of training steps.
            Checkpointing will be triggered at step `max_step`.
    """
    every_n_steps_fn = every_n_steps_policy(n=n, min_step=min_step)

    def fn(*, step: int, evaler_summaries: dict[str, Any]) -> bool:
        return every_n_steps_fn(step=step, evaler_summaries=evaler_summaries) or step == max_step

    return fn


class BaseCheckpointer(Module):
    """A base checkpointer interface.

    Checkpointers are required to implement `save`, `restore`, `stop`, and `checkpoint_paths`.

    Subclasses may optionally also override `__enter__` and `__exit__` for setup or teardown logic
    when checkpointers are used as context managers. Checkpointer contexts are typically entered
    prior to the training loop and exited after the training loop has exited.
    """

    @config_class
    class Config(Module.Config):
        """Configures BaseCheckpointer.

        Attributes:
            dir: The output directory.
            save_policy: A config that instantiates to a CheckpointPolicy.
        """

        dir: Required[str] = REQUIRED
        save_policy: InstantiableConfig[CheckpointPolicy] = config_for_function(
            every_n_steps_policy
        )

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> list[str]:
        """Returns complete checkpoint paths under base dir.

        Args:
            base_dir: Path to checkpoints dir.

        Returns:
            A list of committed checkpoint paths. Incomplete checkpoints are dropped.
        """
        raise NotImplementedError(cls)

    @classmethod
    def latest_checkpoint_path(cls, base_dir: str) -> str:
        """Returns the most recent (highest step count) complete checkpoint under base dir.

        Args:
            base_dir: Path to checkpoints dir.

        Returns:
            The path to the checkpoint directory under base_dir with the highest step count.
            The checkpoint is guaranteed to be complete.
        """
        # Note: checkpoint_paths should already filter incomplete checkpoints.
        return sorted(cls.checkpoint_paths(base_dir)).pop()

    def __init__(self, cfg: Module.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._within_context = False

    def __enter__(self):
        """Enters the checkpointer context manager.

        This is useful for implementing any setup logic (such as starting a garbage collection
        thread).

        This is typically invoked prior to the training loop.
        """
        if self._within_context:
            raise ValueError("Already in a context.")
        self._within_context = True

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exits the checkpointer context manager.

        Typically, teardown logic should be implemented in the `stop()` method instead, which is by
        default invoked from `__exit__`.

        This is typically invoked after the training loop has exited.
        """
        del exc_type, exc, traceback
        self.stop()
        # Note: returning None here lets the caller handle the exception, if any.
        self._within_context = False

    def save(
        self, *, step: int, state: NestedTensor, evaler_summaries: Optional[dict[str, Any]] = None
    ):
        """Saves `state` at the given `step`.

        Args:
            step: The training step corresponding to `state`.
            state: The state to save.
            evaler_summaries: Evaler summaries from the current `step`. Can be used to decide
                whether to save or not (e.g., only checkpointing if a new best metric is achieved).
        """
        raise NotImplementedError(type(self))

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[NestedTensor, NestedTensorSpec],
    ) -> tuple[Optional[int], NestedTensor]:
        """Restores from the checkpoint directory.

        Args:
            step: If None, restores from the latest complete checkpoint, otherwise from the
                specified step.
            state: Ensures that the restored state have the same structure, dtypes, and shapes as
                `state`.

        Returns:
            (restored_step, restored_checkpoint_state).
            If no complete checkpoint is found, returns None as restored_step and the input `state`
            as restored_checkpoint_state.
        """
        raise NotImplementedError(type(self))

    def wait_until_finished(self):
        """Waits for pending asynchronous saves to finish."""
        raise NotImplementedError(type(self))

    def stop(self):
        """Stops the checkpointer. Waits for async writes, garbage collection, etc. to finish."""
        raise NotImplementedError(type(self))


class Checkpointer(BaseCheckpointer):
    """A checkpointer that supports various StateStorage implementations.

    Note that checkpoints are committed via an "index" file. A few utilities for interacting with
    checkpoints committed in this way are provided as static methods on the class.

    In addition to functionality provided by the StateStorage implementation, it provides:
    (1) Global synchronization across processes to ensure that a checkpoint directory is visible
        only after all processes have completed saving the checkpoint (via the "index" file).
    (2) Checkpoint garbage collection.
    """

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures Checkpointer."""

        keep_last_n: int = 1  # Keeps this many past ckpts.
        # If > 0, keeps at least one checkpoint every N steps.
        keep_every_n_steps: Optional[int] = None
        # Interval between garbage collection runs.
        gc_loop_interval_seconds: float = 60
        # A config that instantiates to a StateStorage.
        storage: StateStorage.Config = TensorStoreStateStorage.default_config()
        # A config that instantiates an optional SummaryWriter, and is used to log checkpoints.
        summary_writer: Optional[SummaryWriter.Config] = None

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> list[str]:
        """See `BaseCheckpointer.checkpointer_paths`."""

        # The default checkpointer commits under "<base_dir>/<step_prefix>_<step>/index". Using a
        # concurrent `exists` check for the index file can be several times faster than `glob` on
        # gcs when there are many checkpoint files, even if using a "native" solution like
        # `google-cloud-python` SDK.
        try:
            paths = fs.listdir(base_dir)
        except fs.NotFoundError:
            return []

        paths = [
            os.path.join(base_dir, path, "index") for path in paths if path.startswith(STEP_PREFIX)
        ]
        with futures.ThreadPoolExecutor() as pool:
            index_exists = pool.map(fs.exists, paths)
        return [os.path.dirname(path) for path, committed in zip(paths, index_exists) if committed]

    @classmethod
    def cleanup_checkpoint(cls, ckpt_dir: str, *, sync: bool = True):
        """Removes ckpt_dir if it exists.

        Args:
            ckpt_dir: Checkpoint directory (including step).
            sync: If True, creates a barrier to sync all devices, since removes only happen on
                process 0.
        """
        if jax.process_index() == 0:
            # We always remove the index file as the first step -- otherwise, the partially-removed
            # dir can still be considered a valid checkpoint if rmtree is interrupted.
            index_path = os.path.join(ckpt_dir, "index")
            if fs.exists(index_path):
                fs.remove(index_path)
            if fs.exists(ckpt_dir):
                fs.rmtree(ckpt_dir)
        if sync:
            # Wait for cleanup to complete.
            multihost_utils.sync_global_devices(f"{ckpt_dir}_cleanup")

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: Checkpointer.Config = self.config

        self._storage: StateStorage = cfg.storage.instantiate()
        self._gc_stopping = None
        self._gc_thread = None
        self._save_policy: CheckpointPolicy = cfg.save_policy.instantiate()
        if cfg.summary_writer is not None:
            cfg.summary_writer.dir = cfg.summary_writer.dir or cfg.dir
            self._add_child("summary_writer", cfg.summary_writer)

    def __enter__(self):
        super().__enter__()
        self._start_gc_thread()

    def _start_gc_thread(self):
        """Starts garbage collection (if not already started) in a separate thread."""
        if self._gc_thread is None and jax.process_index() == 0:
            self._gc_stopping = threading.Event()
            self._gc_thread = threading.Thread(
                name=f"{self.path()}.gc_loop",
                target=self._gc_loop,
                kwargs=dict(context_stack=clone_context_stack()),
            )
            self._gc_thread.start()

    def stop(self):
        """See `BaseCheckpointer.stop` for details."""
        self.wait_until_finished()
        self._storage.stop()
        logging.info("Waiting for gc_thread to finish")
        if self._gc_thread is not None:
            self._gc_stopping.set()
            self._gc_thread.join()
            self._gc_thread = None
            logging.info("gc_thread finished")

    def _gc_loop(self, *, context_stack: list[InvocationContext]):
        """Starts garbage collection loop. Will block the current thread."""
        cfg: Checkpointer.Config = self.config
        install_context_stack(context_stack)
        while True:
            if self._gc_stopping.wait(timeout=cfg.gc_loop_interval_seconds):
                break
            self._run_garbage_collection()
        logging.info("GC loop done")

    def ckpt_dir(self, step: int) -> str:
        """Obtains the checkpoint dir for the given step."""
        cfg: Checkpointer.Config = self.config
        return os.path.join(cfg.dir, f"{STEP_PREFIX}_{step:0{STEP_NUM_DIGITS}d}")

    def save(
        self, *, step: int, state: NestedTensor, evaler_summaries: Optional[dict[str, Any]] = None
    ):
        """See `BaseCheckpointer.save` for details.

        In addition to behavior in `BaseCheckpointer`, saving only happens if the configured
        checkpoint policy returns True for the given step and evaler summaries.
        """
        if not self._save_policy(step=step, evaler_summaries=(evaler_summaries or {})):
            return
        if step < 0 or step >= 10**8:
            raise ValueError(f"Out-of-range: {step}")
        ckpt_dir = self.ckpt_dir(step)
        self.cleanup_checkpoint(ckpt_dir)
        self._storage.save_to_dir(
            step=step, state=state, ckpt_dir=ckpt_dir, on_commit_callback=write_index_file
        )
        if "summary_writer" in self.children:
            self.summary_writer.log_checkpoint(
                step=step,
                state=state,
                ckpt_dir=ckpt_dir,
                action=CheckpointerAction.SAVE,
            )

    def _run_garbage_collection(self):
        """Runs one round of garbage collection of past checkpoints.

        We keep as many dirs to satisfy `keep_last_n` and `keep_every_n_steps`, considering only
        those dirs which are fully committed. For example, supposing that `keep_last_n=1`, if we
        count the latest (possibly partially-written) checkpoint as the one to keep, we may end up
        gc'ing the previous (committed) checkpoint. However, if the commit for the current
        checkpoint is pre-empted, this can cause both checkpoints to be corrupted.
        """
        cfg: Checkpointer.Config = self.config
        remaining_dirs, gc_dirs = [], []

        try:
            step_dirs = [
                step.rstrip("/") for step in fs.listdir(cfg.dir) if step.startswith(STEP_PREFIX)
            ]
        except fs.NotFoundError:
            step_dirs = []

        # Gather all candidate checkpoint dirs, as well as all committed checkpoint dirs.
        dirs = sorted([os.path.join(cfg.dir, step) for step in step_dirs], reverse=True)
        committed_dirs = set(self.checkpoint_paths(cfg.dir))

        # Collect the recent non-committed checkpoints, since any of them could be in-progress.
        # (Note that keeping just the first one is not sufficient, e.g., if we restarted with a more
        # frequent saving policy after a prior failure.)
        for saved_dir in dirs:
            if saved_dir in committed_dirs:
                break
            remaining_dirs.append(saved_dir)

        # Always keep the last N committed ckpts. These may not be consecutive -- e.g., we're
        # potentially retaining multiple non-committed ckpts at head.
        num_uncommitted = len(remaining_dirs)
        for saved_dir in dirs[num_uncommitted:]:
            if len(remaining_dirs) >= num_uncommitted + cfg.keep_last_n:
                break
            elif saved_dir in committed_dirs:
                remaining_dirs.append(saved_dir)
            else:
                gc_dirs.append(saved_dir)

        # For subsequent dirs, non-committed dirs are gc'ed, and committed dirs are kept according
        # to keep_n_steps. Note that we iterate in order of oldest to newest.
        last_kept_step = float("-inf")
        for saved_dir in reversed(dirs[len(remaining_dirs) + len(gc_dirs) :]):
            saved_step = parse_step_from_dir(saved_dir)
            if not (
                saved_dir in committed_dirs
                and cfg.keep_every_n_steps
                and saved_step - last_kept_step >= cfg.keep_every_n_steps
            ):
                gc_dirs.append(saved_dir)
                continue
            logging.vlog(
                2,
                "Keeping %s >= %s + %s",
                saved_dir,
                last_kept_step,
                cfg.keep_every_n_steps,
            )
            remaining_dirs.append(saved_dir)
            last_kept_step = saved_step

        for gc_dir in gc_dirs:
            logging.info("Removing %s", gc_dir)
            try:
                # Don't need to sync here since gc only runs on process 0.
                self.cleanup_checkpoint(gc_dir, sync=False)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Ignoring error in removing %s: %s.", gc_dir, e)

        logging.log_every_n_seconds(
            logging.INFO,
            "Garbage collection done on %s. Remaining=%s",
            3600,
            cfg.dir,
            remaining_dirs,
        )

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._storage.wait_until_finished()

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[NestedTensor, NestedTensorSpec],
    ) -> tuple[Optional[int], NestedTensor]:
        """See `BaseCheckpointer.restore` docstring for details.

        A complete checkpoint is one with an "index" file, which is only written after the entire
        checkpoint has been written.
        """
        cfg: Checkpointer.Config = self.config

        def validate_and_restore(*, step: int, ckpt_dir: str):
            ckpt_index = os.path.join(ckpt_dir, "index")
            if not fs.exists(ckpt_index):
                raise ValueError(
                    f"Checkpoint {ckpt_dir} is incomplete -- expected {ckpt_index} to be present."
                )
            restored_state = self._storage.restore_from_dir(
                step=step, state=state, ckpt_dir=ckpt_dir
            )
            logging.info("Restored state from ckpt at step %s", step)
            if "summary_writer" in self.children:
                self.summary_writer.log_checkpoint(
                    step=step,
                    state=state,
                    ckpt_dir=ckpt_dir,
                    action=CheckpointerAction.RESTORE,
                )
            return restored_state

        if step is not None:
            # For a specified step, we try to load it.
            ckpt_dir = self.ckpt_dir(step)
            return step, validate_and_restore(step=step, ckpt_dir=ckpt_dir)

        try:
            # Latest checkpoint path, if it exists, is guaranteed to be complete.
            ckpt_dir = self.latest_checkpoint_path(cfg.dir)
            step = parse_step_from_dir(ckpt_dir)
            restored_state = validate_and_restore(step=step, ckpt_dir=ckpt_dir)
        except IndexError:
            # No checkpoint path exists. Return with input state.
            logging.info("Could not find any completed checkpoints under %s", cfg.dir)
            restored_state = state

        return step, restored_state
