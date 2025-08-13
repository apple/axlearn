# Copyright © 2024 Apple Inc.

"""Implements Orbax replicator checkpointing and provide utilities for correct store.

See the docstring of `OrbaxEmergencyReplicatorCheckpointer` for more details.
"""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.lib
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as oercp
import tensorflow as tf
from absl import flags, logging
from etils import epath
from jax._src.distributed import global_state
from jax._src.mesh import thread_resources
from jax.experimental.array_serialization import serialization

from axlearn.common import utils
from axlearn.common.checkpointer import (
    BaseCheckpointer,
    Checkpointer,
    CheckpointPolicy,
    CheckpointValidationType,
    InstantiableConfig,
    StateStorage,
    StateStorageCommitCallback,
    async_save_tf_savables,
    check_state_structure,
    config_for_function,
    every_n_steps_policy,
    multihost_utils,
    read_index_file,
    restore_tf_savables,
    write_index_file,
)
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec

FLAGS = flags.FLAGS


@contextmanager
def setup(spec: str):
    """Setups FLAGS.process_id and FLAGS.distributed_coordinator as required by Orbax.

    Args:
        spec: Key=Value pairs separated by comma.
    """
    parsed_args = {}
    allowed_fields = ["local_ckpt_dir"]
    for field in spec.split(","):
        k, v = field.split("=")
        if k not in allowed_fields:
            raise ValueError(f"Expected key in {allowed_fields}, got key={k}.")
        parsed_args[k] = v
    if "local_ckpt_dir" not in parsed_args:
        raise ValueError("local_ckpt_dir must be specified.")
    # Get process ID and IP of coordinator
    process_id, coordinator_address = _retrieve_jax_init_info(parsed_args["local_ckpt_dir"])
    FLAGS.process_id = int(process_id)
    FLAGS.distributed_coordinator = coordinator_address
    FLAGS.experimental_orbax_use_distributed_process_id = True

    yield


class _TFSavablesStateStorage(StateStorage):
    """A StateStorage implementation that only saves the index file and tf savables."""

    @config_class
    class Config(StateStorage.Config):
        timeout_secs: int = 300

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # One thread is sufficient because `async_save_tf_savables` only creates one future.
        self._executor = ThreadPoolExecutor(1)
        self._manager = serialization.AsyncManager(timeout_secs=cfg.timeout_secs)

    def _get_spec(self, *, step: int, state: Nested[Any]) -> Nested[Any]:
        spec = {"index": [("step", int(step))], "tf_ckpt_map": {}}
        for path, value in utils.flatten_items(state):
            if isinstance(value, (Tensor, TensorSpec)):
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec["index"].append(
                    (path, {"dtype": str(dtype), "shape": str(tuple(value.shape))})
                )
            elif isinstance(value, tf.data.Iterator):
                spec["index"].append((path, str(type(value))))
                spec["tf_ckpt_map"][path] = value
            else:
                spec["index"].append((path, value))
        logging.log_first_n(logging.INFO, "TF savables spec: %s", 1, str(spec))
        return spec

    def save_to_dir(
        self,
        *,
        step: int,
        state: Nested[Tensor],
        ckpt_dir: str,
        on_commit_callback: StateStorageCommitCallback = write_index_file,
    ):
        start_time = time.perf_counter()
        # We write data files directly to `ckpt_dir`. `index` is written into `ckpt_dir` in
        # `on_commit_callback` to finalize the checkpoint.
        spec = self._get_spec(step=step, state=state)
        self.wait_until_finished()
        jax.block_until_ready(state)

        save_tf_future = async_save_tf_savables(
            spec["tf_ckpt_map"],
            executor=self._executor,
            dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}"),
        )

        def commit():
            on_commit_callback(ckpt_dir=ckpt_dir, index=spec["index"])
            logging.info(
                "Serialization of TF savables to %s completed in %s seconds.",
                ckpt_dir,
                time.perf_counter() - start_time,
            )

        # pylint: disable=protected-access
        self._manager._add_futures([save_tf_future])
        self._manager._start_async_commit(commit)

    def wait_until_finished(self):
        self._manager.wait_until_finished()

    def restore_from_dir(
        self,
        step: int,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
        *,
        ckpt_dir: str,
        validation: CheckpointValidationType = CheckpointValidationType.EXACT,
    ) -> Nested[Tensor]:
        spec = self._get_spec(step=step, state=state)
        logging.info("Restoring TF savables from directory %s", ckpt_dir)
        check_state_structure(
            read_index_file(ckpt_dir), target_structure=spec["index"], validation=validation
        )
        restore_tf_savables(
            spec["tf_ckpt_map"], dir=os.path.join(ckpt_dir, f"tf_{jax.process_index()}")
        )
        multihost_utils.sync_global_devices(ckpt_dir)
        return state

    def stop(self):
        self._executor.shutdown(wait=True)


def _wait_for_file_to_disappear(f, timeout=300):
    for _ in range(timeout):
        if not f.exists():
            return True
        time.sleep(1)
    return False


def _extract_step(f):
    # The base file name is formatted as {job_name}-s{step}-n{node_rank}-w{worker_rank}
    return f.rsplit("-", 3)[1][1:]


def _block_and_process_restore_dir(directory, timeout=300):
    """Block until the directory symlink ending with `.restore` appears, then extract
    the step number and rename the directory using the step number.
    """
    suffix = ".restore"
    for _ in range(timeout):
        files = os.listdir(directory)
        for f in files:
            if f.endswith(suffix):
                step = _extract_step(f)
                if step != "0":
                    os.rename(epath.Path(directory) / f, epath.Path(directory) / step)
                    logging.info(
                        "Renamed restore directory at step %s to %s.",
                        step,
                        epath.Path(directory) / step,
                    )
                else:
                    logging.info("Found a restore directory at step 0, skipping renaming.")
                return
        time.sleep(1)
    logging.info("%d seconds have passed but no .restore file was found.", timeout)


def _retrieve_jax_init_info(local_ckpt_dir):
    """Retrieve JAX init info from a local file."""
    jax_init_info_file = "jax-init-info.txt"
    local_jax_init_info_file = epath.Path(local_ckpt_dir) / jax_init_info_file
    # Allow time for the JAX init info file to be populated by GKE.
    # File only populated when the worker with process id of 0 is determined.
    for i in range(900):
        if local_jax_init_info_file.exists():
            return local_jax_init_info_file.read_text().split("\n")[:2]
        logging.info(
            "Unable to locate %s after %d seconds, sleeping for 1 second before retrying...",
            jax_init_info_file,
            i,
        )
        time.sleep(1)
    logging.info(
        "Unable to locate i%s after 900 seconds,"
        "returning empty process id and coordinator address.",
        jax_init_info_file,
    )
    return "", ""


class OrbaxEmergencyReplicatorCheckpointer(BaseCheckpointer):
    """Checkpointer implementation that uses Orbax emergency replicator checkpointer.

    EXPERIMENTAL. Do not use for actual training runs since the checkpoint layout will likely
    change in the future."""

    #    _NON_TENSORS_PREFIX: str = "non-tensors"
    #    _TENSORS_PREFIX: str = "tensors"

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxEmergencyReplicatorCheckpointer.

        Attributes:
            keep_last_n: Keep this many past ckpts.
            keep_every_n_steps: If > 0, keeps at least one persistent checkpoint every N steps.
            local_keep_last_n: Keep this many past ckpts in local storage (e.g. node memory).
                This should almost always set to 1 to avoid OOM.
            local_dir: Ckpt base path for local storage. The content in this path must persist
                across pod restarts unless the restart is caused by node failure. `local_dir` must
                be the same for all processes or processes may hang.
            trainer_dir: A string that's unique for the current run. Typically, this is set to
                trainer_dir. Local checkpoint will be stored in local_dir/sha256(trainer_dir).
                During init, all other folders in local_dir will be removed to prevent unexpected
                memory usage.
            save_policy: Save policy for persistent checkpoints.
            local_save_policy: Save policy for local checkpoints. This should be more frequent than
                `save_policy`. Note that data iterator will be saved with either `save_policy` or
                `local_save_policy` indicate we should save.
            non_tensor_async_timeout_secs: Timeout for async barrier in seconds when saving
                non-tensor states.
            async_timeout_secs: Timeout for async barrier in seconds when saving tensors.
            replica_axis_index: The index of the "data" axis.
        """

        keep_last_n: int = 1
        keep_every_n_steps: Optional[int] = None
        local_keep_last_n: int = 1
        local_save_policy: InstantiableConfig[CheckpointPolicy] = config_for_function(
            every_n_steps_policy
        ).set(n=10)
        local_dir: str = "/checkpoint"
        trainer_dir: Required[str] = REQUIRED
        non_tensor_async_timeout_secs: int = 300
        async_timeout_secs: int = 3600
        replica_axis_index: Required[int] = REQUIRED

    # @classmethod
    # def checkpoint_paths(cls, base_dir: str) -> List[str]:
    #    """See `BaseCheckpointer.checkpointer_paths`.
    #
    #    Only persistent checkpoint paths are returned. There's no guarantee that the paths returned
    #    have committed TF savables. Use `checkpoint_steps` to get steps with both tensors and
    #    committed TF savables.
    #    """
    #    logging.log_first_n(
    #        logging.WARNING,
    #        msg="checkpoint_paths is deprecated. Use checkpoint_steps instead.",
    #        n=1,
    #    )
    #    tensors_dir = os.path.join(base_dir, cls._TENSORS_PREFIX)
    #    return [str(path) for path in ocp.utils.checkpoint_steps_paths(tensors_dir)]
    #
    # @classmethod
    # def checkpoint_steps(cls, base_dir) -> list[int]:
    #    """See `BaseCheckpointer.checkpointer_steps`.
    #
    #    Only persistent checkpoint steps are returned.
    #    """
    #    return list(
    #        set(
    #            ocp.utils.checkpoint_steps(os.path.join(base_dir, cls._TENSORS_PREFIX))
    #        ).intersection(
    #            set(Checkpointer.checkpoint_steps(os.path.join(base_dir, cls._NON_TENSORS_PREFIX)))
    #        )
    #    )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: OrbaxEmergencyReplicatorCheckpointer.Config = self.config
        self._name_format = ocp.step.standard_name_format(
            step_prefix=None,
            step_format_fixed_length=None,
        )
        # if jax.process_index() == 0:
        #     fs.makedirs(os.path.join(cfg.dir, self._NON_TENSORS_PREFIX))
        #     fs.makedirs(os.path.join(cfg.dir, self._TENSORS_PREFIX))
        self._local_dir = cfg.local_dir
        # Orbax replicator ckpt requires this function to be called prior to checkpointer
        # operations. This function also serves as a barrier.
        ocp.multihost.initialize_runtime_to_distributed_ids()
        ocp.multihost.initialize_distributed_to_device_ids()

        num_slices = int(os.environ["MEGASCALE_NUM_SLICES"])

        replicator_file = "replicator.yaml"
        temp_file = replicator_file + ".tmp"
        replicator_file_path = epath.Path(self._local_dir) / replicator_file
        if not _wait_for_file_to_disappear(replicator_file_path):
            logging.info("Existing replicator.yaml did not disappear in time.")
        else:
            logging.info("replicator.yaml no longer exists, creating new replicator.yaml.")
        temp_file = epath.Path(self._local_dir) / temp_file
        num_nodes = jax.process_count()
        nodes_per_slice = num_nodes // num_slices

        node_rank = global_state.process_id
        my_process_index = jax.process_index()
        proc_index_to_node_rank = ocp.multihost.runtime_to_distributed_ids()

        my_in_pipeline_index = my_process_index % nodes_per_slice
        peer_ranks = []
        for i in range(num_slices):
            peer_process_index = i * nodes_per_slice + my_in_pipeline_index
            if peer_process_index != my_process_index:
                peer_process_rank = proc_index_to_node_rank[peer_process_index]
                peer_ranks.append(peer_process_rank)

        logging.info("Peers for NodeRank %s: %s", node_rank, peer_ranks)

        run_name = os.environ.get("HOSTNAME").split("job")[0].rstrip("-")

        replicator_yaml = f"""job-name: {run_name}
      framework: orbax
      assume-data-parallelism: 2
      node-rank: {node_rank}
      nodes: {num_nodes}
      peer-ranks: {peer_ranks}
      backup-interval-minutes: 5"""

        temp_file.write_text("\n".join([l.strip() for l in replicator_yaml.split("\n")]))
        os.rename(temp_file, replicator_file_path)
        if not _wait_for_file_to_disappear(replicator_file_path):
            logging.info("The newly created replicator.yaml was not deleted in time.")
        else:
            logging.info("The newly created replicator.yaml was deleted, moving forward.")
            _block_and_process_restore_dir(self._local_dir)

        ckpt_cfg: Checkpointer.Config = Checkpointer.default_config()
        # TODO(hanzhi-zhou): this `keep_last_n` may not be what users expect since non-tensor
        # states will save when either local or persistent checkpoint will save.
        ckpt_cfg.keep_last_n = cfg.keep_last_n
        ckpt_cfg.keep_every_n_steps = cfg.keep_every_n_steps
        ckpt_cfg.storage = _TFSavablesStateStorage.default_config()
        ckpt_cfg.storage.timeout_secs = cfg.non_tensor_async_timeout_secs
        #        ckpt_cfg.dir = os.path.join(cfg.dir, self._NON_TENSORS_PREFIX)
        ckpt_cfg.name = "non-tensors-checkpointer"

        save_policy = cfg.save_policy.instantiate()
        local_save_policy = cfg.local_save_policy.instantiate()

        # Non-tensor states must save when either local or persistent ckpt needs to be saved for
        # restore from either to succeed.
        def _composite_save_policy(*, step: int, evaler_summaries: dict[str, Any]):
            return (
                save_policy(step=step, evaler_summaries=evaler_summaries)
                or local_save_policy(step=step, evaler_summaries=evaler_summaries)
                or self._reached_preemption
            )

        self._composite_save_policy = _composite_save_policy
        ckpt_cfg.save_policy = config_for_function(lambda: _composite_save_policy)
        #        self._non_tensor_manager: Checkpointer = ckpt_cfg.instantiate(parent=self)
        #        self._non_tensor_manager: Optional[oercp.ReplicatorCheckpointManager] = None
        self._tensor_manager: Optional[oercp.ReplicatorCheckpointManager] = None
        # See comments of _eval_summaries in `OrbaxCheckpointer`.
        self._eval_summaries = None
        self._reached_preemption = False

    def _get_abstract_state(
        self, state_with_tensors: Nested[Tensor]
    ) -> Nested[jax.ShapeDtypeStruct]:
        """Generate the abstract states required by the Orbax replicator checkpointer."""
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
            state_with_tensors,
        )

    def _get_tensor_manager(
        self,
    ) -> oercp.ReplicatorCheckpointManager:
        """Creates the replicator checkpoint manager if not exists.

        We defer the creation of this checkpoint manager because it requires the state dict,
        which is not present during __init__.
        """
        if self._tensor_manager is not None:
            return self._tensor_manager

        # For meaning of these options, refer to
        # https://github.com/google/orbax/blob/de0b6d0bca643d12840ae73a1f7cfee80af73dcd/checkpoint/orbax/checkpoint/experimental/emergency/replicator_checkpoint_manager.py#L87
        self._tensor_manager = oercp.ReplicatorCheckpointManager(
            self._local_dir,
            options=oercp.ReplicatorCheckpointManagerOptions(
                save_interval_steps=100,
                step_name_format=self._name_format,
            ),
            global_mesh=thread_resources.env.physical_mesh,
        )
        return self._tensor_manager

    def save(
        self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[Dict[str, Any]] = None
    ):
        """See `BaseCheckpointer.save` for details."""
        assert self._eval_summaries is None, self._eval_summaries
        self._eval_summaries = copy.deepcopy(evaler_summaries or {})
        self._reached_preemption = self._tensor_manager.reached_preemption(step)

        state_with_tensors = jax.tree.map(
            lambda x: x if isinstance(x, (Tensor, TensorSpec)) else None, state
        )
        # Note that save() waits for prior serialization to finish.
        # self._non_tensor_manager.save(step=step, state=state)
        # _non_tensor_manager will block for train step to finish. Start the timer here to avoid
        # including step time in total blocking time.
        start_t = time.perf_counter()
        self._get_tensor_manager().save(
            step=step, args=ocp.args.Composite(state=ocp.args.PyTreeSave(item=state_with_tensors))
        )
        time_diff = time.perf_counter() - start_t
        if self._composite_save_policy(step=step, evaler_summaries=self._eval_summaries):
            logging.info("In-mem ckpt blocking time is %fs.", time_diff)
        self._eval_summaries = None
        if self._reached_preemption:
            self.wait_until_finished()
            raise SystemExit(f"Exiting after saving checkpoint at {step=} due to pre-emption.")

    def _checkpoint_steps_include_local(self) -> list[int]:
        """Returns a sorted list of complete checkpoints, including both persistent and local.

        This is done by finding the intersection of the checkpoint steps managed by tensor and
        non-tensor manager. `all_steps` from tensor manager gives the steps of complete local and
        persistent checkpoints.

        This function assumes tensor manager has already been initialized.
        """
        return sorted(
            set(self._tensor_manager.all_steps())
            #            set(self._tensor_manager.all_steps()).intersection(
            #                set(
            #                    (
            #                        parse_step_from_dir(d)
            #                        for d in self._non_tensor_manager.checkpoint_paths(
            #                            self._non_tensor_manager.config.dir
            #                        )
            #                    )
            #                )
            #            )
        )

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """Restores state from either local or persistent checkpoint."""
        start_t = time.perf_counter()
        cfg: OrbaxEmergencyReplicatorCheckpointer.Config = self.config
        state_with_tensors = jax.tree.map(
            lambda x: x if isinstance(x, (Tensor, TensorSpec)) else None, state
        )
        tensor_manager = self._get_tensor_manager()
        if step is None:
            common_steps = self._checkpoint_steps_include_local()

            if not common_steps:
                logging.warning("Could not find any completed checkpoints under %s.", cfg.dir)
                return None, state

            step = max(common_steps)

        #        restore_step, state = self._non_tensor_manager.restore(step=step, state=state)
        #        assert step == restore_step

        def _restore_args(x: Any) -> ocp.RestoreArgs:
            return ocp.checkpoint_utils.construct_restore_args(
                jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
            )

        restore_args = jax.tree.map(_restore_args, state)

        restored_state_with_tensors = tensor_manager.restore(
            step=step,
            args=ocp.args.Composite(
                state=ocp.args.PyTreeRestore(
                    item=self._get_abstract_state(state_with_tensors),
                    restore_args=restore_args,
                )
            ),
        )

        restored_state_with_tensors = restored_state_with_tensors["state"]

        restored_state = jax.tree.map(
            lambda non_tensor, tensor: non_tensor if tensor is None else tensor,
            state,
            restored_state_with_tensors,
        )

        time_diff = time.perf_counter() - start_t
        logging.info("Took %ss to restore replicator checkpoint from %s.", time_diff, cfg.dir)
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        # self._non_tensor_manager.wait_until_finished()
        self._tensor_manager.wait_until_finished()

    def stop(self, *, has_exception: bool = False):
        """See `BaseCheckpointer.stop` for details."""
        # self._non_tensor_manager.stop(has_exception=has_exception)
        if self._tensor_manager:
            self._tensor_manager.close()
