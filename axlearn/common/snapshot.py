# Copyright © 2024 Apple Inc.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
import queue
import threading
from typing import Any, Optional

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training  # pytype: disable=import-error
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types  # pytype: disable=import-error
from pathwaysutils.experimental import concatenate_by_mesh_axis  # pytype: disable=import-error
from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
import jax.numpy as jnp
from axlearn.common.utils import Nested, TensorSpec, get_current_abstract_or_physical_mesh

_logger = logging.getLogger(__name__)


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0, trainer_state_specs: Optional[Nested[TensorSpec]] = None):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index
    self.trainer_state_specs = trainer_state_specs
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    while True:
      pinned_state, step = self._queue.get()
      print("In snapshot worker")
      try:
        _logger.info(
            "[*] [Snapshot Thread] Waiting for snapshot at step %d to be ready...",
            step,
        )
        jax.block_until_ready(pinned_state)
        _logger.info(
            "[*] [Snapshot Thread] Snapshot at step %d is ready and secured.",
            step,
        )
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
      except Exception as e:  # pylint: disable=broad-except
        _logger.warning(
            "[*] [Snapshot Thread] Failed to secure snapshot at step %d: %s.",
            step,
            e,
        )
      finally:
        print("In snapshot worker finally")
        self._queue.task_done()

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    print("In snapshot pytree")
    if self._queue.full():
      _logger.warning("Snapshotter busy. Skipping snapshot for step %d", step)
      return

    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") else None, state
    )
    print("before jax.put")

    pinned_state = jax.device_put(state, pinned_shardings)
    print("after jax.put")
    self._queue.put((pinned_state, step))

  def load_pytree(
      self,
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Initializes a state and restores from the latest snapshot.

    Uses `self.trainer_state_specs` to properly re-partition onto the new mesh.

    Args:
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory).

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
      ValueError: If `trainer_state_specs` is not provided during initialization.
    """
    if self.trainer_state_specs is None:
        raise ValueError("trainer_state_specs must be provided to Snapshotter to use load_pytree.")

    def spec_to_sds(spec):
        if not hasattr(spec, "shape"):
            return spec
        mesh = get_current_abstract_or_physical_mesh()
        sharding = jax.sharding.NamedSharding(mesh, getattr(spec, "mesh_axes", None))
        return jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding)

    abstract_state = jax.tree.map(spec_to_sds, self.trainer_state_specs, is_leaf=lambda x: hasattr(x, "shape"))

    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def get_active_pytree(x, target_x):
      if not hasattr(x, "shape") or not hasattr(target_x, "shape"):
        return x
      if x.shape == target_x.shape:
        return x
      starts = [0] * x.ndim
      stops = [min(s1, s2) for s1, s2 in zip(x.shape, target_x.shape)]
      sliced_x = jax.lax.slice(x, starts, stops)
      pad_widths = [(0, max(0, s2 - s1)) for s1, s2 in zip(x.shape, target_x.shape)]
      if any(p > 0 for _, p in pad_widths):
          sliced_x = jnp.pad(sliced_x, pad_widths)
      return sliced_x

    _logger.info("Restoring from snapshot at step %d...", step)
    pinned_state = jax.tree.map(get_active_pytree, pinned_state, abstract_state)

    # Re-shard on host to the target device mesh
    host_target_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") else None, abstract_state
    )

    host_target_state = jax.device_put(
        pinned_state, host_target_shardings
    )

    # Move from host back to device (TPU) memory.
    restored_state = jax.device_put(
        host_target_state, jax.tree.map(lambda x: x.sharding if hasattr(x, "sharding") else None, abstract_state)
    )
    jax.block_until_ready(restored_state)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )