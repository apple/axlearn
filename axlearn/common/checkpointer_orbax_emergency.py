# Copyright © 2024 Apple Inc.

"""Implements Orbax emergency checkpointing and provide utilities for correct store.

See the docstring of `OrbaxEmergencyCheckpointer` for more details.
"""

import copy
import functools
import hashlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.lib
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as oecp
import tensorflow as tf
from absl import flags, logging
from jax._src.distributed import global_state
from jax._src.mesh import thread_resources
from jax.experimental.array_serialization import serialization

from axlearn.common import file_system as fs
from axlearn.common import utils, utils_spmd
from axlearn.common.checkpointer import (
    STEP_NUM_DIGITS,
    STEP_PREFIX,
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
    parse_step_from_dir,
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

    See the docstring of `get_consistent_proc_info` for more details.

    Args:
        spec: Key=Value pairs separated by comma. Key must be one of ("local_address",
            "barrier_timeout_seconds", "local_ckpt_dir"). See the docstring of
            `get_consistent_proc_info`.
    """
    parsed_args = {}
    allowed_fields = ["local_address", "barrier_timeout_seconds", "local_ckpt_dir"]
    for field in spec.split(","):
        k, v = field.split("=")
        if k not in allowed_fields:
            raise ValueError(f"Expected key in {allowed_fields}, got key={k}.")
        parsed_args[k] = v
    if "barrier_timeout_seconds" in parsed_args:
        parsed_args["barrier_timeout_seconds"] = int(parsed_args["barrier_timeout_seconds"])
    if "local_ckpt_dir" not in parsed_args:
        raise ValueError("local_ckpt_dir must be specified.")
    # pylint: disable-next=missing-kwoa
    info = get_consistent_proc_info(
        **parsed_args,
        trainer_dir=FLAGS.trainer_dir,
        distributed_coordinator=FLAGS.distributed_coordinator,
        num_processes=FLAGS.num_processes,
        process_id=FLAGS.process_id,
        jax_backend=FLAGS.jax_backend,
        initialization_timeout=FLAGS.initialization_timeout,
    )
    FLAGS.process_id = info.inv_proc_id
    FLAGS.distributed_coordinator = info.address
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


_PROCESS_ID_FILE_NAME: str = "process_id.txt"


@dataclass
class _ProcessInfo:
    """Records the process id and address information for this node.

    Attributes:
        address: The global coordinator address. This is set during the first run and stays and
            stays the same unless process 0 failed.
        inv_proc_id: The invariant process id of this node. This process id is set during the first
            run and stays the same for all subsequent runs unless this node failed.
        cur_proc_id: Internal field. The new process id assigned externally after failover. Used
            during ID negotiation after failover.
        key: Internal field. Key used during ID negotiation after failover.
        num_proc_per_slice: Internal field. Used to calculate slice ID for TPU.
    """

    address: str
    inv_proc_id: int
    cur_proc_id: int
    key: Optional[str] = None
    num_proc_per_slice: Optional[int] = None

    def to_string(self):
        return "|".join(str(x) for x in [self.address, self.inv_proc_id, self.cur_proc_id])

    @property
    def prev_slice_id(self):
        assert self.num_proc_per_slice is not None
        return self.inv_proc_id // self.num_proc_per_slice

    @property
    def cur_slice_id(self):
        assert self.num_proc_per_slice is not None
        return self.cur_proc_id // self.num_proc_per_slice

    @classmethod
    def from_string(
        cls, data: str, *, key: Optional[str] = None, num_proc_per_slice: Optional[int] = None
    ):
        ls = data.split("|")
        assert len(ls) == 3
        return cls(ls[0], int(ls[1]), int(ls[2]), key=key, num_proc_per_slice=num_proc_per_slice)


def _get_previous_process_info(local_dir: str, *, trainer_dir: str) -> _ProcessInfo:
    """Gets process info from local checkpoint directory."""
    path = os.path.join(local_dir, _get_unique_id(trainer_dir), _PROCESS_ID_FILE_NAME)
    if not fs.exists(path):
        return _ProcessInfo(address="", inv_proc_id=-1, cur_proc_id=-1)

    with fs.open(path) as f:
        return _ProcessInfo.from_string(f.read())


def _dump_process_info(local_dir: str, *, trainer_dir: str, proc_info: _ProcessInfo):
    """Dumps process info to local checkpoint directory."""
    local_dir = os.path.join(local_dir, _get_unique_id(trainer_dir))
    fs.makedirs(local_dir)
    process_id_file = os.path.join(local_dir, _PROCESS_ID_FILE_NAME)
    with fs.open(process_id_file, "w") as f:
        f.write(proc_info.to_string())


def _get_unique_id(trainer_dir: str) -> str:
    return hashlib.sha256(trainer_dir.encode(), usedforsecurity=False).hexdigest()


def _logger_init():
    """Init logger in spawned processes that don't inherit parent's logger."""
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def _init_consistent_proc_ids(
    *,
    local_address: Optional[str] = None,
    barrier_timeout_seconds: int = 300,
    trainer_dir: str,
    local_ckpt_dir: str,
    **setup_kwargs,
):
    """Exchanges id info through jax coordinator and dumps to local file.

    During failover, healthy nodes will read their locally stored process id file, but failed nodes
    will lost their process ids. To assign ids that are free in the global id range (i.e. 0 to
    num_processes - 1), we let each node report its process id (-1 if missing) to rank 0, and rank
    0 will figure out suitable IDs to assign to each failed node. We reuse Jax's distributed client
    to avoid writing our own coordinator.
    """
    _logger_init()

    jax_backend = setup_kwargs["jax_backend"]
    timeout_ms = barrier_timeout_seconds * 1000
    utils_spmd.setup(**setup_kwargs)
    client: jax.lib.xla_extension.DistributedRuntimeClient = global_state.client
    local_proc_info = _get_previous_process_info(local_ckpt_dir, trainer_dir=trainer_dir)
    key_prefix = "axlearn/id_reassign"
    # Local key just needs to be unique for each process.
    local_proc_info.key = f"{key_prefix}/{jax.process_index()}"

    if jax_backend == "tpu":
        worker_hostnames = os.environ["TPU_WORKER_HOSTNAMES"].split(",")
        num_slices = int(os.environ["MEGASCALE_NUM_SLICES"])
        num_proc_per_slice = len(worker_hostnames)
        worker_id = int(os.environ["TPU_WORKER_ID"])

        # Coordinator port for TPU is hardcoded. Reference:
        # https://github.com/jax-ml/jax/blob/1aa5de66a8f3c910115cac2fbe118e0facd7a3be/jax/_src/clusters/cloud_tpu_cluster.py#L29
        local_proc_info.address = f"{worker_hostnames[worker_id]}:8476"
        # Note: cannot use jax.process_index() here because it may be different from the
        # distributed id. This is a jax problem.
        local_proc_info.cur_proc_id = (
            int(os.environ["MEGASCALE_SLICE_ID"]) * num_proc_per_slice + worker_id
        )
    elif jax_backend == "gpu":
        if local_address is None:
            raise ValueError(
                "local_address must be set for GPU when using in-memory checkpointing."
            )
        local_proc_info.address = local_address
        local_proc_info.cur_proc_id = setup_kwargs["process_id"]
    else:
        raise RuntimeError(f"Unsupported backend {jax_backend}.")

    # Every worker reports its proc info to rank 0.
    client.key_value_set(local_proc_info.key, local_proc_info.to_string())
    client.wait_at_barrier("axlearn/id-reassign-gather-id", timeout_in_ms=timeout_ms)

    # Then, rank 0 assigns inv_proc_id for worker that's missing their inv_proc_id and find the
    # coordinator address.
    if local_proc_info.cur_proc_id == 0:
        ids = client.key_value_dir_get(key_prefix)
        proc_infos: list[_ProcessInfo] = []

        def first_run_assign_fn(info: _ProcessInfo):
            info.inv_proc_id = info.cur_proc_id

        inv_id_assign_fn = first_run_assign_fn
        if jax_backend == "tpu":
            # For TPUs, we have the additional requirement that process ids in slice id X must be
            # in range [X * num_processes_per_slice, (X + 1) * num_processes_per_slice). Therefore,
            # we first identify the healthy slices' ids and then figure out the slice ids to assign
            # to failed slices. Each process in the failed slice will then get id `new_slice_id *
            # num_proc_per_slice + cur_proc_id % num_proc_per_slice`. After id assignment, the
            # address of process that's assigned with id=0 will be broadcasted to every worker.

            # Mapping from new slice ids to assigned slice ids forfailed slices.
            failed_slices_new_ids = {}
            for k, data in ids:
                info = _ProcessInfo.from_string(data, key=k, num_proc_per_slice=num_proc_per_slice)
                proc_infos.append(info)
                if info.inv_proc_id == -1:
                    failed_slices_new_ids[info.cur_slice_id] = -1

            already_assigned_slice_ids = set()
            for info in proc_infos:
                if info.cur_slice_id not in failed_slices_new_ids:
                    already_assigned_slice_ids.add(info.prev_slice_id)

            # If there're no assigned slice ids, that means all slices have failed or we're in the
            # very first run. In that case, first_run_assign_fn will be used.
            if already_assigned_slice_ids:
                to_be_assigned_slice_ids = set(range(num_slices)) - already_assigned_slice_ids
                assert len(to_be_assigned_slice_ids) == len(failed_slices_new_ids)
                for k, new_id in zip(failed_slices_new_ids.keys(), to_be_assigned_slice_ids):
                    failed_slices_new_ids[k] = new_id

                def assign_fn(info: _ProcessInfo):
                    proc_id = info.inv_proc_id
                    if (new_slice_id := failed_slices_new_ids.get(info.cur_slice_id)) is not None:
                        proc_id = (
                            new_slice_id * num_proc_per_slice
                            + info.cur_proc_id % num_proc_per_slice
                        )
                    info.inv_proc_id = proc_id

                inv_id_assign_fn = assign_fn

        elif jax_backend == "gpu":
            num_processes = setup_kwargs["num_processes"]
            # For GPU backend, failed nodes are assigned with ids that are missing in the global id
            # range with arbitrary order.
            assigned_ids = set()
            for key, data in ids:
                info = _ProcessInfo.from_string(data, key=key)
                proc_infos.append(info)
                assigned_ids.add(info.inv_proc_id)

            # If there're no assigned ids, that means all slices have failed or we're in the
            # very first run. In that case, first_run_assign_fn will be used.
            if assigned_ids:
                to_be_assigned_ids = iter(set(range(num_processes)) - assigned_ids)

                def assign_fn(info: _ProcessInfo):
                    if info.inv_proc_id == -1:
                        info.inv_proc_id = next(to_be_assigned_ids)

                inv_id_assign_fn = assign_fn

        coordinator_address = None
        for info in proc_infos:
            inv_id_assign_fn(info)
            if info.inv_proc_id == 0:
                coordinator_address = info.address
        assert coordinator_address is not None
        for info in proc_infos:
            info.address = coordinator_address
            client.key_value_set(info.key + "/get", info.to_string())

    new_info = _ProcessInfo.from_string(
        client.blocking_key_value_get(local_proc_info.key + "/get", timeout_in_ms=timeout_ms)
    )
    logging.info(
        "Previous proc id: %d. Assigned proc id: %d. Global coordinator address: %s.",
        local_proc_info.inv_proc_id,
        new_info.inv_proc_id,
        new_info.address,
    )
    _dump_process_info(local_ckpt_dir, trainer_dir=trainer_dir, proc_info=new_info)
    # Block to avoid coordinator exiting too early.
    client.wait_at_barrier("axlearn/id-reassign-finalize", timeout_in_ms=timeout_ms)
    jax.distributed.shutdown()


def get_consistent_proc_info(
    *,
    local_address: Optional[str] = None,
    barrier_timeout_seconds: int = 300,
    trainer_dir: str,
    local_ckpt_dir: str,
    **setup_kwargs,
) -> _ProcessInfo:
    """Gets the invariant process id of the current process and global coordinator's address.

    This function guarantees process id <-> node mapping stays the same for healthy nodes after a
    failover. This is required to preserve shard order for in-memory checkpoint recovery. For GPU
    training, all healthy nodes will have their process id unchanged. For TPU, all nodes in the
    healthy slices will have their process id unchanged. See docstring of
    `_init_consistent_proc_ids` for implementation details.

    Args:
        local_address: A IP:Port that can be used as the coordinator if this rank is elected.
            This Port must be free in the coordinator pod and IP:Port must be reachable from all
            other processes.
        barrier_timeout_seconds: Timeout in seconds for the barrier and key_value_set operations.
        trainer_dir: Path to the trainer dir.
        local_ckpt_dir: Path to the local checkpoint dir.
        **setup_kwargs: Args to `utils_spmd.setup()`.

    Returns:
        A _ProcessInfo whose `inv_proc_id` should be used as the process id and `address` should be
        used as the global coordinator address.
    """
    platform = os.environ.get("JAX_PLATFORMS", "")
    try:
        start_t = time.perf_counter()
        # Patch platform so the process doesn't waste time initializing accelerators.
        os.environ["JAX_PLATFORMS"] = "cpu"
        proc = mp.get_context("spawn").Process(
            target=_init_consistent_proc_ids,
            kwargs=dict(
                local_address=local_address,
                barrier_timeout_seconds=barrier_timeout_seconds,
                trainer_dir=trainer_dir,
                local_ckpt_dir=local_ckpt_dir,
                **setup_kwargs,
            ),
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(
                "Expects id assignment process to finish normally. "
                f"Got exit code {proc.exitcode}. Please check the log above for errors."
            )

        info = _get_previous_process_info(local_ckpt_dir, trainer_dir=trainer_dir)
        if info.inv_proc_id == -1:
            raise RuntimeError("Expects inv process id != -1, but got -1.")
        logging.info(
            "Successfully finished process ID assignment in %fs", time.perf_counter() - start_t
        )
        return info
    finally:
        # Restore previous platform settings.
        if platform != "":
            os.environ["JAX_PLATFORMS"] = platform
        else:
            del os.environ["JAX_PLATFORMS"]


class OrbaxEmergencyCheckpointer(BaseCheckpointer):
    """Checkpointer implementation that uses Orbax emergency checkpoint.

    EXPERIMENTAL. Do not use for actual training runs since the checkpoint layout will likely
    change in the future.

    ## Summary:

    This checkpointer is designed to improve the goodput of large multi-slice training jobs that
    use data-parallelism across slices. At least two data-parallel slices are required. For other
    use cases where this is not applicable or ultimate goodput is not required, please use
    `OrbaxCheckpointer`.

    Why it can improve goodput:
    1. It can save to a local path (usually backed by a ramdisk) more frequently, so the progress
       lost during restart can be reduced. This is in contrast with saving to remote filesystem
       such as GCS directly, which has limited bandwidth to support frequent checkpointing.
    2. During restart, checkpoint can be broadcasted through network, which is faster than reading
       from a remote filesystem.

    To use the checkpointer, besides configuring it properly, it also requires
    `get_consistent_proc_info` to be called and pass `inv_proc_id` and `address` as
    `process_id` and `coordinator_address` to `jax.distributed.initialize`.

    ## How it works under the hood

    This checkpointer is intended for multi-slice training that uses data-parallelism across
    slices. Orbax emergency checkpoint works by exploiting the following properties:
    1.  Tensors are replicated across data-parallel replicas.
    2.  When a slice fails in a multi-slice training and failover is started, only nodes
        corresponding to the non-healthy slice may be restarted. Healthy nodes from healthy slices
        will not restart.

    Hence, all slices can write checkpoints to node's memory or disk, providing us with redundancy
    when there's a failure. This checkpoint frequency can be much higher than remote filesystem,
    which has limited bandwidth to support high frequency saving. Checkpoints on nodes are referred
    as local checkpoints. Checkpoints on remote filesystem are referred as persistent checkpoints.

    When a failure occurs, Orbax checkpointer will find the latest step from all local and
    persistent checkpoints. If the checkpoint is local, the slice on which that checkpoint is
    stored will read the checkpoint and broadcast the read values to other slices. Since local
    checkpoints are scattered across different hosts, the process id, which determines the shard id
    of locally stored shards, must stay the same for nodes in the healthy replicas to guarantee a
    correct restore. We provide an utility function `get_consistent_proc_info` that returns the
    process id and global coordinator address. They must be passed to `jax.distributed.initialize`.

    However, the above procedure doesn't apply to some non-tensor states such as data iterators.
    Data iterators are unique across jax processes, and thus cannot be stored on nodes. Orbax
    emergency checkpointer doesn't support non-tensor states. Therefore, we reuse axlearn
    Checkpointer to save, restore and garbage collect those states, which include the index file
    and tf iterators. These non-tensor states will be saved whenever local or persistent checkpoint
    need to be saved. As the result, the persistent checkpoint structure looks like this:

    ```
    ├── path_prefix
    │   ├── non-tensors
    │   │   └── step_00000010
    │   │       ├── index
    │   │       └── tf_xxx
    │   └── tensors
    │       └── step_00000010
    │           └── orbax_files_xxx
    ```

    A persistent training checkpoint `step_xxx` is commited when `non-tensors/step_xxx/index`
    exists and `tensors/step_xxx` is commited by Orbax. Refer to the docstring of
    `OrbaxCheckpointer` for Orbax's commit criteria.

    To abstract the details of the checkpoint layout, the `checkpoint_steps` API returns all steps
    for which both Tensor and non-Tensor states have been fully committed.
    """

    _NON_TENSORS_PREFIX: str = "non-tensors"
    _TENSORS_PREFIX: str = "tensors"

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxEmergencyCheckpointer.

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
        local_dir: str = "/host-tmp/checkpoints"
        trainer_dir: Required[str] = REQUIRED
        non_tensor_async_timeout_secs: int = 300
        async_timeout_secs: int = 3600
        replica_axis_index: Required[int] = REQUIRED

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> List[str]:
        """See `BaseCheckpointer.checkpointer_paths`.

        Only persistent checkpoint paths are returned. There's no guarantee that the paths returned
        have committed TF savables. Use `checkpoint_steps` to get steps with both tensors and
        committed TF savables.
        """
        logging.log_first_n(
            logging.WARNING,
            msg="checkpoint_paths is deprecated. Use checkpoint_steps instead.",
            n=1,
        )
        tensors_dir = os.path.join(base_dir, cls._TENSORS_PREFIX)
        return [str(path) for path in ocp.utils.checkpoint_steps_paths(tensors_dir)]

    @classmethod
    def checkpoint_steps(cls, base_dir) -> list[int]:
        """See `BaseCheckpointer.checkpointer_steps`.

        Only persistent checkpoint steps are returned.
        """
        return list(
            set(
                ocp.utils.checkpoint_steps(os.path.join(base_dir, cls._TENSORS_PREFIX))
            ).intersection(
                set(Checkpointer.checkpoint_steps(os.path.join(base_dir, cls._NON_TENSORS_PREFIX)))
            )
        )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        self._name_format = ocp.step.standard_name_format(
            step_prefix=STEP_PREFIX,
            step_format_fixed_length=STEP_NUM_DIGITS,
        )
        if jax.process_index() == 0:
            fs.makedirs(os.path.join(cfg.dir, self._NON_TENSORS_PREFIX))
            fs.makedirs(os.path.join(cfg.dir, self._TENSORS_PREFIX))
        # Cleanup local checkpoints from different runs.
        unique_id = _get_unique_id(cfg.trainer_dir)
        for fd in fs.listdir(cfg.local_dir):
            if not fd.startswith(".") and fd != unique_id:
                fs.rmtree(os.path.join(cfg.local_dir, fd))
        self._local_dir = os.path.join(cfg.local_dir, unique_id)
        fs.makedirs(self._local_dir)
        # Orbax emergency ckpt requires this function to be called prior to checkpointer
        # operations. This function also serves as a barrier.
        ocp.multihost.initialize_runtime_to_distributed_ids()
        ocp.multihost.initialize_distributed_to_device_ids()
        ckpt_cfg: Checkpointer.Config = Checkpointer.default_config()
        # TODO(hanzhi-zhou): this `keep_last_n` may not be what users expect since non-tensor
        # states will save when either local or persistent checkpoint will save.
        ckpt_cfg.keep_last_n = cfg.keep_last_n
        ckpt_cfg.keep_every_n_steps = cfg.keep_every_n_steps
        ckpt_cfg.storage = _TFSavablesStateStorage.default_config()
        ckpt_cfg.storage.timeout_secs = cfg.non_tensor_async_timeout_secs
        ckpt_cfg.dir = os.path.join(cfg.dir, self._NON_TENSORS_PREFIX)
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
        self._non_tensor_manager: Checkpointer = ckpt_cfg.instantiate(parent=self)
        self._tensor_manager: Optional[oecp.CheckpointManager] = None
        # See comments of _eval_summaries in `OrbaxCheckpointer`.
        self._eval_summaries = None
        self._reached_preemption = False

    # pylint: disable-next=redefined-builtin
    def ckpt_dir(self, step: int, dir: Optional[str] = None) -> str:
        """Obtains the checkpoint dir for the given step."""
        if dir is None:
            dir = self._non_tensor_manager.directory
        return str(ocp.step.build_step_path(dir, self._name_format, step))

    def _get_abstract_state(
        self, state_with_tensors: Nested[Tensor]
    ) -> Nested[jax.ShapeDtypeStruct]:
        """Generate the abstract states required by the Orbax emergency checkpointer."""
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
            state_with_tensors,
        )

    def _get_tensor_manager(self, state_with_tensors: Nested[Tensor]) -> oecp.CheckpointManager:
        """Creates the emergency checkpoint manager if not exists.

        We defer the creation of this checkpoint manager because it requires the state dict,
        which is not present during __init__.
        """
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        if self._tensor_manager is not None:
            return self._tensor_manager

        save_policy = cfg.save_policy.instantiate()
        local_save_policy = cfg.local_save_policy.instantiate()

        def _orbax_save_fn(
            step: int, last_saved_step: Optional[int], wrapped_save_policy: CheckpointPolicy
        ) -> bool:
            del last_saved_step
            return wrapped_save_policy(step=step, evaler_summaries=self._eval_summaries)

        # For meaning of these options, refer to
        # https://github.com/google/orbax/blob/95be2c021bc8cbf4badd83a053ff57b7a9f9b314/checkpoint/orbax/checkpoint/experimental/emergency/checkpoint_manager.py#L277
        self._tensor_manager = oecp.CheckpointManager(
            self._local_dir,
            persistent_directory=os.path.join(cfg.dir, self._TENSORS_PREFIX),
            global_mesh=thread_resources.env.physical_mesh,
            abstract_state=self._get_abstract_state(state_with_tensors),
            options=oecp.CheckpointManagerOptions(
                local=oecp.LocalCheckpointOptions(
                    should_save_fn=functools.partial(
                        _orbax_save_fn, wrapped_save_policy=local_save_policy
                    ),
                    max_to_keep=cfg.local_keep_last_n,
                ),
                persistent=oecp.PersistentCheckpointOptions(
                    should_save_fn=functools.partial(
                        _orbax_save_fn, wrapped_save_policy=save_policy
                    ),
                    max_to_keep=cfg.keep_last_n,
                ),
                replica_axis_index=cfg.replica_axis_index,
                async_options=oecp.checkpoint_manager.AsyncOptions(
                    timeout_secs=cfg.async_timeout_secs
                ),
                step_name_format=self._name_format,
                cleanup_tmp_directories=True,
                enable_async_checkpointing=True,
            ),
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
        self._non_tensor_manager.save(step=step, state=state)
        # _non_tensor_manager will block for train step to finish. Start the timer here to avoid
        # including step time in total blocking time.
        start_t = time.perf_counter()
        self._get_tensor_manager(state_with_tensors).save(
            step=step, args=ocp.args.PyTreeSave(item=state_with_tensors)
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
            set(self._tensor_manager.all_steps()).intersection(
                set(
                    (
                        parse_step_from_dir(d)
                        for d in self._non_tensor_manager.checkpoint_paths(
                            self._non_tensor_manager.config.dir
                        )
                    )
                )
            )
        )

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """Restores state from either local or persistent checkpoint."""
        start_t = time.perf_counter()
        cfg: OrbaxEmergencyCheckpointer.Config = self.config
        state_with_tensors = jax.tree.map(
            lambda x: x if isinstance(x, (Tensor, TensorSpec)) else None, state
        )
        tensor_manager = self._get_tensor_manager(state_with_tensors)
        if step is None:
            common_steps = self._checkpoint_steps_include_local()
            if not common_steps:
                logging.warning("Could not find any completed checkpoints under %s.", cfg.dir)
                return None, state
            step = max(common_steps)

        restore_step, state = self._non_tensor_manager.restore(step=step, state=state)
        assert step == restore_step

        restored_state_with_tensors = tensor_manager.restore(
            step=step,
            args=ocp.args.PyTreeRestore(item=self._get_abstract_state(state_with_tensors)),
        )
        # Merge non-tensor and tensor states by replacing leaves of the non-tensor Pytree with the
        # not-None leaves of the tensor Pytree.
        restored_state = jax.tree.map(
            lambda non_tensor, tensor: non_tensor if tensor is None else tensor,
            state,
            restored_state_with_tensors,
        )
        time_diff = time.perf_counter() - start_t
        logging.info("Took %ss to restore emergency checkpoint from %s.", time_diff, cfg.dir)
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._non_tensor_manager.wait_until_finished()
        self._tensor_manager.wait_until_finished()

    def stop(self):
        """See `BaseCheckpointer.stop` for details."""
        self._non_tensor_manager.stop()
        self._tensor_manager.close()
