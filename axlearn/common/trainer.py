# Copyright © 2023 Apple Inc.

"""Defines SpmdTrainer, a trainer that supports partitioning of computation and data with GSPMD."""

import contextlib
import itertools
import math
import os.path
import threading
import time
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Set, Tuple, Union

import jax
import tensorflow as tf
from absl import logging
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit

from axlearn.common import measurement, utils
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import Checkpointer
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.learner import ForwardOutputs, Learner, NestedOptParam
from axlearn.common.module import InvocationContext, Module, child_context, clone_context_stack
from axlearn.common.module import functional as F
from axlearn.common.module import install_context_stack, new_output_collection
from axlearn.common.optimizer_base import OptParam
from axlearn.common.param_init import DefaultInitializer
from axlearn.common.state_builder import Builder as TrainerStateBuilder
from axlearn.common.summary_writer import BaseWriter, SummaryWriter
from axlearn.common.utils import (
    HybridMeshShape,
    MeshShape,
    NestedPartitionSpec,
    NestedTensor,
    PartitionSpec,
    Tensor,
    count_model_params,
    flatten_items,
    match_regex_rules,
    thread_stack_traces,
)


class TrainerState(NamedTuple):
    prng_key: Union[Tensor, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Union[NestedTensor, NestedPartitionSpec]


# pylint: disable-next=too-many-instance-attributes
class SpmdTrainer(Module):
    """A trainer implementation that supports partitioning of computation and data with GSPMD."""

    @config_class
    # pylint: disable-next=too-many-instance-attributes
    class Config(Module.Config):
        """Configures SpmdTrainer."""

        # The input source.
        input: Required[InstantiableConfig] = REQUIRED

        # A summary writer to log tagged summary values.
        summary_writer: BaseWriter.Config = SummaryWriter.default_config()

        # The trainer root dir.
        # By default, checkpoints will be written under {dir}/checkpoints/
        # and summaries will be written under {dir}/summaries.
        dir: Required[str] = REQUIRED

        # If not None, initializes trainer states according to the given config.
        # This is only applied if we aren't restoring from an existing checkpoint.
        init_state_builder: Optional[TrainerStateBuilder.Config] = None

        # The maximum number of steps.
        max_step: Union[int, float] = math.inf

        # The default mesh configuration.
        #
        # If specified as a MeshShape, must have the same length as mesh_axis_names. Implicitly,
        # this treats the mesh shape as the ICI mesh shape; we default to a DCN mesh shape that
        # partitions the first non-singleton axis across granules (e.g. TPU slices or GPU nodes).
        # If all axes are singletons, this implies a single-granule environment and therefore an
        # all-1's DCN mesh shape.
        #
        # As an example on 2 H100 nodes, for mesh axes (pipeline, data, model) and a MeshShape of
        # (1, 2, 8), we break the "data" axis across DCN -- this produces a DCN mesh shape (1, 2, 1)
        # and an ICI mesh shape (1, 1, 8), i.e. 2-way data-parallelism across DCN, and 8-way model
        # parallelism within-node (e.g. NVLink). If instead the MeshShape is provided as (2, 1, 8),
        # we break along the "pipeline" axis, producing a DCN mesh shape of (2, 1, 1) and ICI mesh
        # shape (1, 1, 8) for 2-way pipeline-parallelism across DCN and 8-way model parallelism
        # within-node.
        #
        # If specified as a HybridMeshShape, each member must have the same length as
        # mesh_axis_names.
        #
        # Use `mesh_rules` to set different mesh shapes depending on the hardware platform.
        mesh_shape: Required[Union[MeshShape, HybridMeshShape]] = REQUIRED
        # The mesh axis names. The names can be referenced in ParameterSpec.mesh_axes.
        mesh_axis_names: Required[Sequence[str]] = REQUIRED
        # Subset of mesh axis names over which the leaves of the input batch are sharded.
        batch_axis_names: Union[str, Sequence[str]] = "data"

        # An optional list of (regex, MeshShape) pairs to override the default mesh configuration.
        #
        # This is useful when we want to use different mesh shapes depending on the
        # device types (e.g., 'tpu-v4-128' vs. 'gpu-p4de.24xlarge-32').
        #
        # Given a `mesh_selector` string (usually representing the device type and set by user's
        # launch script), the first rule that with a regex that matches the selector will determine
        # the mesh shape.
        #
        # If no rule matches, the default mesh configuration will be used.
        mesh_rules: Optional[Sequence[Tuple[str, Optional[MeshShape]]]] = None

        # The model config.
        model: Required[BaseModel.Config] = REQUIRED
        # The learner config.
        learner: Required[Learner.Config] = REQUIRED
        # The checkpointer config.
        checkpointer: Checkpointer.Config = Checkpointer.default_config()
        # A dict of evaler names to configs, each name must be non-empty.
        evalers: Dict[str, SpmdEvaler.Config] = {}

        # If True, saves the input iterator in checkpoints.
        #
        # It is OK to change this option for existing models, since our checkpoint restoration
        # logic can handle legacy checkpoints with a different value of `save_input_iterator`.
        #
        # WARNING: many input processing ops are stateful and cannot be saved, e.g.,
        # FailedPreconditionError: RandomUniformInt is stateful.
        # FailedPreconditionError: ReduceDataset is stateful.
        # FailedPreconditionError: SentencepieceOp is stateful.
        save_input_iterator: bool = False

        # At which steps to start profiler tracing.
        # Currently each trace will cover 3 consecutive training steps.
        # The start steps must therefore be at least 3 steps apart from each other.
        start_trace_steps: Sequence[int] = []
        # By default, only trace on host 0.
        start_trace_process_indices: Union[Literal["all"], Sequence[int]] = [0]

        # Prune empty state updates.
        # Must be set to True.
        # The configuration option will be removed in a future AXLearn version.
        # It has not been removed yet to prevent the need for a messy regeneration of large numbers
        # of golden configs.
        prune_empty_state_updates: bool = True

        # Cast float inputs and model parameters to this dtype for the train step.
        # If None, we do not cast.
        train_dtype: Optional[jnp.dtype] = None

        # If > 0, run a watchdog thread to print the thread stack traces if step does not
        # increment within this interval.
        watchdog_timeout_seconds: Optional[float] = None

        # An optional recorder for measuring common metrics like step time.
        recorder: Optional[InstantiableConfig[measurement.Recorder]] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        devices: Optional[Sequence[jax.Device]] = None,
    ):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if not cfg.prune_empty_state_updates:
            raise ValueError(
                "Setting prune_empty_state_updates to False is no longer supported.\n"
                "The config option will be removed in a future AXLearn version."
            )

        self._step: int = None
        self._trainer_state: TrainerState = None
        self._jit_train_step: jax.stages.Wrapped = None
        self._watchdog_stopping = None
        self._watchdog_thread = None
        self._recorder = maybe_instantiate(cfg.recorder)

        if cfg.model.dtype is None:
            raise ValueError(f"dtype must be explicitly specified for {self.path()}.model")
        if cfg.model.param_init is None:
            cfg.model.param_init = DefaultInitializer.default_config()
            logging.info(
                "model.param_init is not specified. Default to DefaultInitializer: %s",
                cfg.model.param_init,
            )

        if cfg.train_dtype is not None:
            utils.validate_float_dtype(cfg.train_dtype)

        # Create the device mesh.
        self._step_log(
            "Devices: global=%s local=%s %s",
            jax.device_count(),
            jax.local_device_count(),
            [device.platform for device in jax.local_devices()],
        )
        self._step_log("Mesh shape: %s", cfg.mesh_shape)
        devices = (
            utils.create_device_mesh(mesh_shape=cfg.mesh_shape) if devices is None else devices
        )
        mesh = jax.sharding.Mesh(devices, cfg.mesh_axis_names)
        self._step_log("Global mesh: %s", mesh)
        self._mesh = mesh

        # Create all children within the mesh context so that utils.input_partition_spec() works
        # properly.
        with self.mesh():
            self._add_child("input", cfg.input.set(is_training=True))
            # Start from the beginning of the input dataset by default.
            self._input_iter = iter(self.input.dataset())
            cfg.summary_writer.dir = cfg.summary_writer.dir or os.path.join(
                cfg.dir, "summaries", "train_train"
            )
            self._add_child("summary_writer", cfg.summary_writer)
            self._add_child("model", cfg.model)
            self._add_child("learner", cfg.learner)
            cfg.checkpointer.dir = cfg.checkpointer.dir or os.path.join(cfg.dir, "checkpoints")
            self._add_child("checkpointer", cfg.checkpointer)
            if cfg.init_state_builder is not None:
                self._add_child("init_state_builder", cfg.init_state_builder)

            self._model_param_specs = self.model.create_parameter_specs_recursively()
            model_param_partition_specs = jax.tree_util.tree_map(
                lambda spec: spec.mesh_axes, self._model_param_specs
            )
            for name, spec in utils.flatten_items(self._model_param_specs):
                self._step_log("Model param spec: %s=%s", name, spec)
            self._learner_state_partition_specs = self.learner.create_state_partition_specs(
                self._model_param_specs
            )
            for name, spec in utils.flatten_items(self._learner_state_partition_specs):
                self._step_log("Learner state spec: %s=%s", name, spec)
            self._trainer_state_specs = TrainerState(
                prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
                model=self._model_param_specs,
                learner=self._learner_state_partition_specs,
            )
            self._trainer_state_partition_specs = jax.tree_util.tree_map(
                lambda spec: spec.mesh_axes, self._trainer_state_specs
            )
            # Create evalers, which depend on model_param_partition_specs.
            self._evalers = {}
            for evaler_name, evaler_cfg in cfg.evalers.items():
                evaler_cfg.summary_writer.dir = evaler_cfg.summary_writer.dir or os.path.join(
                    cfg.dir, "summaries", evaler_name
                )
                self._evalers[evaler_name] = self._add_child(
                    evaler_name,
                    evaler_cfg,
                    model=self.model,
                    model_param_partition_specs=model_param_partition_specs,
                )

    @property
    def step(self):
        return self._step

    @property
    def trainer_state(self):
        return self._trainer_state

    @property
    def trainer_state_specs(self):
        return self._trainer_state_specs

    @property
    def trainer_state_partition_specs(self):
        return self._trainer_state_partition_specs

    def _train_step_input_partition_specs(self):
        # By default, each input tensor is fully partitioned along the batch axis.
        return utils.input_partition_spec()

    def model_params_for_eval(self):
        state = self.trainer_state
        if self.config.learner.ema.decay is not None:
            logging.log_first_n(logging.INFO, "Using model parameter EMA for eval", 10)
            return state.learner["ema"].ema
        return state.model

    def _step_log(self, msg, *args, **kwargs):
        logging.info(
            "%s process % 3d step % 8d] " + msg,
            self.path(),
            jax.process_index(),
            -1 if self.step is None else self.step,
            *args,
            **kwargs,
        )

    def mesh(self):
        return jax.sharding.Mesh(self._mesh.devices, self._mesh.axis_names)

    @contextlib.contextmanager
    def _watchdog(self):
        self._start_watchdog()
        try:
            yield
        finally:
            self._stop_watchdog()

    def _start_watchdog(self):
        cfg = self.config
        if cfg.watchdog_timeout_seconds and self._watchdog_thread is None:
            self._watchdog_stopping = threading.Event()
            self._watchdog_thread = threading.Thread(
                name=f"{self.path()}.watchdog",
                target=self._watchdog_loop,
                kwargs=dict(context_stack=clone_context_stack()),
            )
            self._watchdog_thread.start()

    def _stop_watchdog(self):
        """Stops the checkpointer. Waits for async writes and garbage collection loop to finish."""
        logging.info("Waiting for watchdog_thread to finish")
        if self._watchdog_thread is not None:
            self._watchdog_stopping.set()
            self._watchdog_thread.join()
            self._watchdog_thread = None
            logging.info("watchdog_thread finished")

    def _watchdog_loop(self, *, context_stack: List[InvocationContext]):
        cfg = self.config
        install_context_stack(context_stack)
        while True:
            last_step = self.step
            if self._watchdog_stopping.wait(timeout=cfg.watchdog_timeout_seconds):
                break
            current_step = self.step
            if current_step == last_step:
                self._step_log(
                    "Watchdog triggered because step has not incremented in the last %s seconds.\n"
                    "NOTE: this is not an error message, but meant to help debugging "
                    "in case the trainer is stuck.\n"
                    "Threads:\n%s",
                    cfg.watchdog_timeout_seconds,
                    "\n".join(itertools.chain.from_iterable(thread_stack_traces())),
                )
            else:
                self.vlog(1, "Watchdog check passed: %s -> %s", last_step, current_step)
        logging.info("Watchdog loop done")

    def _should_force_run_evals(
        self,
        *,
        return_evaler_summaries: Optional[Union[bool, Set[str]]] = None,
        evalers: Dict[str, SpmdEvaler.Config],
    ) -> Set[str]:
        """Determines which, if any, evalers to force run at the last training step.

        Args:
            return_evaler_summaries: Whether to run evalers at the last training step.
                If None or False, do not force run evalers; if True, force run all
                evalers; if given as a set of strings, force run all the evalers with
                the name in the set.
            evalers: A dict of evaler configs. Only the keys are used to check against
                return_evaler_summaries.

        Returns:
            A set of strings for the evalers to force run at the last training step.
            If empty, no evaler is force run.

        Raises:
            ValueError: If return_evaler_summaries is a set of strings with any not matching
                evaler names; or return_evaler_summaries is an invalid type.
        """
        force_run_evals = set()
        if return_evaler_summaries is True:
            force_run_evals = set(evalers.keys())
        elif isinstance(return_evaler_summaries, Set):
            for evaler_name in return_evaler_summaries:
                if evaler_name not in evalers:
                    raise ValueError(
                        f"{evaler_name} does not match any evaler names: {evalers.keys()}"
                    )
                force_run_evals.add(evaler_name)
        elif not isinstance(return_evaler_summaries, bool) and return_evaler_summaries is not None:
            raise ValueError(
                f"return_evaler_summaries must be bool, None or Set. Got {return_evaler_summaries}"
            )
        return force_run_evals

    def _maybe_record_event(self, event: measurement.Event, *args, **kwargs):
        if self._recorder is not None:
            self._recorder.record(event, *args, **kwargs)

    # pylint: disable-next=too-many-statements,too-many-branches
    def run(
        self, prng_key: Tensor, *, return_evaler_summaries: Optional[Union[bool, Set[str]]] = None
    ) -> Optional[NestedTensor]:
        """Runs training.

        Args:
            prng_key: The pseudo random generator key.
            return_evaler_summaries: Whether to force run evalers and return summaries at the
                last training step. If None or False, do not force run evalers and no evaler
                summaries are returned; if True, force run all evalers at the last training step
                and return summaries; if given as a set of strings, force run all the evalers
                with the name in the set at the last training step and return summaries.

        Returns:
            None if no training is run or a dict otherwise.
            If returned is a dict, it contains the outputs from the last step of training,
            with 'loss' and 'aux' outputs. If return_evaler_summaries is True or a set of strings,
            it also contains 'evaler_summaries', which is a dict containing the evaler summaries
            force run at the last training step. The dict will have evaler names as keys and
            metrics summary dict as values (None for evalers not included in force run).
            The metrics summary dict has string keys for the name of the metrics and can have
            different types of values such as WeightedScalar, Tensor, or string, depending on
            the specific `metric_calculator` config of the evaler.
        """
        with self._watchdog(), self.mesh(), jax.log_compiles(self.vlog_is_on(1)):
            cfg = self.config
            # Check if need to force run evals at the last training step.
            force_run_eval_sets_at_max_step = self._should_force_run_evals(
                return_evaler_summaries=return_evaler_summaries, evalers=cfg.evalers
            )

            # Prepare training.
            if not self._prepare_training(prng_key):
                return None

            with self.checkpointer:
                logging.info("Starting loop...")
                start_time = time.perf_counter()
                num_steps = 0
                output = None
                stop_trace_step = None

                for input_batch in self.input.batches(self._input_iter):
                    self._maybe_record_event(measurement.Event.START_STEP, self._step)
                    logging.log_first_n(
                        logging.INFO, "input_batch=%s", 3, utils.shapes(input_batch)
                    )

                    # Stop or start tracing if necessary.
                    stop_trace_step = self._maybe_stop_or_start_tracing(stop_trace_step, output)

                    self._step = self._step + 1
                    self.vlog(3, "Start step %s", self.step)
                    output = self._run_step(
                        utils.host_to_global_device_array(input_batch),
                        force_run_evals=force_run_eval_sets_at_max_step
                        if self.step >= cfg.max_step
                        else None,
                    )
                    self.vlog(3, "Done step %s", self.step)
                    num_steps += 1
                    if num_steps % 100 == 0:
                        now = time.perf_counter()
                        average_step_time = (now - start_time) / num_steps
                        self._step_log("Average step time: %s seconds", average_step_time)
                        self.summary_writer(self.step, {"average_step_time": average_step_time})
                        num_steps = 0
                        start_time = now
                    if self.step >= cfg.max_step:
                        self._step_log("Reached max_step=%s. Stopping", cfg.max_step)
                        break
                if self.step < cfg.max_step:
                    self._step_log("Reached end of inputs. Stopping")
            self._step_log("Checkpointer flushed.")
            return output

    def _opt_params(self, model_params: NestedTensor) -> NestedOptParam:
        """Returns a tree of OptParam for Learner.{init,update}."""
        # self._model_param_specs can be incomplete. Complete it first.
        specs = utils.complete_partition_spec_tree(
            jax.tree_util.tree_structure(model_params), self._model_param_specs
        )
        return jax.tree_util.tree_map(
            lambda param, spec: OptParam(
                value=param,
                factorization_spec=spec.factorization if spec is not None else None,
                weight_decay_scale=spec.weight_decay_scale if spec is not None else 1.0,
            ),
            model_params,
            specs,
        )

    def init(self, prng_key: Tensor):
        """Initializes self._step and self._trainer_state.

        Args:
            prng_key: The initialization key.
        """
        if "init_state_builder" not in self.children:
            self._init_with_prebuilt_state(prng_key, prebuilt_state=None)
            return
        input_state_type = self.init_state_builder.input_state_type()
        if input_state_type == TrainerStateBuilder.StateType.TENSOR_SPECS:
            logging.info("Creating state from init_state_builder before initialization.")
            built_state = self._restore_from_builder()
            self._init_with_prebuilt_state(prng_key, prebuilt_state=built_state)
        else:
            assert input_state_type == TrainerStateBuilder.StateType.TENSORS, input_state_type
            logging.info("Creating state from init_state_builder after initialization.")
            self._init_with_prebuilt_state(prng_key, prebuilt_state=None)
            built_state = self._restore_from_builder()
            self._step = built_state.step
            self._trainer_state = built_state.trainer_state

    def _init_with_prebuilt_state(
        self,
        prng_key: Tensor,
        *,
        prebuilt_state: Optional[TrainerStateBuilder.State],
    ):
        """Initializes `self._step` and `self._trainer_state`, optionally from `prebuilt_state`.

        If `prebuilt_state` contains the complete trainer state, sets it as `self._trainer_state`.

        Otherwise `prebuilt_state` is expected to contain a subset of the model parameters and
        none of the learner state. Initializes model parameters by copying from `prebuilt_state`
        if available, otherwise with `prng_key`.

        Args:
            prebuilt_state: None or a TrainerStateBuilder.State constructed by
                `self.init_state_builder`.

        Raises:
            ValueError: if `prebuilt_state.trainer_state` contains non-Tensor leaf nodes or a
                Tensor of unexpected shape.
            NotImplementedError: if `prebuilt_state.trainer_state` is not complete, but contains
                state other than model parameters.
        """
        if prebuilt_state is None:
            prebuilt_state = TrainerStateBuilder.State(
                step=None,
                trainer_state=self.trainer_state_specs,
                built_keys=set(),
            )
        self._step = prebuilt_state.step
        all_trainer_state_keys = set(
            key for key, _ in utils.flatten_items(self.trainer_state_specs)
        )
        if prebuilt_state.built_keys == all_trainer_state_keys:
            logging.info(
                "Prebuilt state has the complete trainer state.",
            )
            for key, value in utils.flatten_items(prebuilt_state.trainer_state):
                if not isinstance(value, Tensor):
                    raise ValueError(f"{key}={value} is not a Tensor")
            # All keys are already built.
            self._trainer_state = prebuilt_state.trainer_state
            return

        for key in prebuilt_state.built_keys:
            if not key.startswith("model/"):
                raise NotImplementedError(f"Partial initialization is not supported for: {key}")

        prebuilt_model_state_partition_spec = jax.tree_util.tree_map(
            lambda value: value.sharding if isinstance(value, Tensor) else None,
            prebuilt_state.trainer_state.model,
        )
        prebuilt_model_state = jax.tree_util.tree_map(
            lambda value: value if isinstance(value, Tensor) else None,
            prebuilt_state.trainer_state.model,
        )

        def _init_state(prng_key: Tensor, prebuilt_model_state: NestedTensor):
            prng_key, init_key = jax.random.split(prng_key)
            logging.info("prebuilt_model_state: %s", utils.shapes(prebuilt_model_state))
            model_params = self.model.initialize_parameters_recursively(
                init_key,
                prebuilt=prebuilt_model_state,
            )
            self.vlog(
                1, "tree_structure(model_params)=%s", jax.tree_util.tree_structure(model_params)
            )
            learner_params = self.learner.init(self._opt_params(model_params))
            return TrainerState(
                prng_key=prng_key,
                model=model_params,
                learner=learner_params,
            )

        logging.info("prebuilt_model_state_partition_spec: %s", prebuilt_model_state_partition_spec)
        logging.info("trainer_state_partition_specs: %s", self._trainer_state_partition_specs)
        init_computation = pjit(
            _init_state,
            in_shardings=(None, prebuilt_model_state_partition_spec),
            out_shardings=self._trainer_state_partition_specs,
        )
        self._step_log("Initializing trainer state.")
        with self.mesh():
            self._trainer_state = init_computation(prng_key, prebuilt_model_state)

    def _log_trainer_state_stats(self):
        total_num_params = count_model_params(self._trainer_state.model)
        self._step_log("Total number of model params: %s", f"{total_num_params:,}")
        self.summary_writer(0, {"num_model_params": total_num_params})

        # Training state size.
        total_state_bytes = 0
        total_sharded_state_bytes = 0
        state_spec_map = dict(utils.flatten_items(self.trainer_state_specs))
        for path, value in utils.flatten_items(self._trainer_state):
            self._step_log(
                "State: %s=%s(%s) mesh_axes=%s",
                path,
                value.dtype,
                value.shape,
                state_spec_map.get(path),
            )
            total_state_bytes += value.size * value.dtype.itemsize
            shard_shape = value.sharding.shard_shape(value.shape)
            total_sharded_state_bytes += math.prod(shard_shape) * value.dtype.itemsize

        total_sharded_state_gb = total_sharded_state_bytes / 1024**3
        if jax.process_count() > 1:
            max_sharded_state_gb = multihost_utils.process_allgather(total_sharded_state_gb).max()
        else:
            max_sharded_state_gb = total_sharded_state_gb

        self._step_log(
            "Training state size: %.2f GiB\n"
            "Training state size (partitioned): %.2f GiB\n"
            "Max training state size (partitioned): %.2f GiB",
            total_state_bytes / 1024**3,
            total_sharded_state_gb,
            max_sharded_state_gb,
        )

    def _prepare_training(self, prng_key: Tensor) -> bool:
        """Prepares training.

        This function does the following to prepare the training procedure:
        1. Restores trainer state from checkpoint.
        2. Initializes step to zero if it's not in the checkpoint.
        3. Returns early if max_steps has been reached.
        4. Otherwise Jits self._train_step.

        Args:
            prng_key: The PRNG key of the `run` method.

        Returns:
            A boolean indicating whether the model training should start. If not, return
                None from the `run` function.
        """
        cfg = self.config

        # Attempt to restore the latest checkpoint, which may contain a saved `_input_iter`.
        self.restore_checkpoint(restore_step=None)

        if self.step is None:
            # If we didn't restore from checkpoint, attempt to build initial state according
            # to `cfg.init_state_builder` and initialize the remaining parameters.
            self.init(prng_key)
            self._step = 0

            # Note the default checkpointer and evaler do nothing at step 0 with min_step=1.
            self.save_checkpoint(self._run_eval())

            # Log trainer state tree.
            if jax.process_index() == 0:
                with tf.io.gfile.GFile(os.path.join(cfg.dir, "trainer_state_tree.txt"), "w") as f:
                    f.write(str(jax.tree_util.tree_structure(self._trainer_state)))

        self._log_trainer_state_stats()
        # Log config.
        self.summary_writer.log_config(cfg, step=self.step)

        if self.step >= cfg.max_step:
            self._step_log("Already reached max_step=%s. Stopping", cfg.max_step)
            return False

        self._jit_train_step = self._pjit_train_step()
        return True

    def restore_checkpoint(self, restore_step: Optional[int] = None) -> Optional[int]:
        """Restores trainer state from checkpoint.

        If successful, sets self._step and self._trainer_state to the restored step and state,
        respectively.

        Args:
            restore_step: If an integer, restore from the specified step
                (or throw an exception if the restoration fails).
                If None, try to restore from the latest to the earliest available checkpoint.

        Returns:
            The restored step or None (if restore_step is None and no checkpoint is found).
        """
        cfg: SpmdTrainer.Config = self.config
        # Try to restore the checkpoint at `restore_step`.
        with self.mesh():
            for path, spec in utils.flatten_items(self._trainer_state_specs):
                self.vlog(1, "restore spec: %s=%s", path, spec)
            ckpt_state_spec = self._trainer_state_specs._asdict()
            ckpt_state_spec_with_input_iter = dict(
                **ckpt_state_spec, input_iter=iter(self.input.dataset())
            )
            restore_input_iter = cfg.save_input_iterator
            try:
                # Try to restore with `input_iter`.
                step, ckpt_state = self.checkpointer.restore(
                    step=restore_step,
                    state=(
                        ckpt_state_spec_with_input_iter if restore_input_iter else ckpt_state_spec
                    ),
                )
                if step is not None:
                    self.vlog(
                        0,
                        "Restored checkpoint at %s with restore_input_iter=%s",
                        step,
                        restore_input_iter,
                    )
            except ValueError as e:
                logging.warning(
                    "Attempt to restore checkpoint with restore_input_iter=%s failed: %s",
                    restore_input_iter,
                    e,
                )
                # Restore with a different restore_input_iter setting.
                restore_input_iter = not restore_input_iter
                step, ckpt_state = self.checkpointer.restore(
                    step=restore_step,
                    state=(
                        ckpt_state_spec_with_input_iter if restore_input_iter else ckpt_state_spec
                    ),
                )
                if step is not None:
                    self.vlog(
                        0,
                        "Restored checkpoint at %s with restore_input_iter=%s",
                        step,
                        restore_input_iter,
                    )
            if step is not None:
                self._step = step
                self._trainer_state = TrainerState(
                    **{k: v for k, v in ckpt_state.items() if k in TrainerState._fields}
                )
                if cfg.save_input_iterator and "input_iter" in ckpt_state:
                    self._input_iter = ckpt_state["input_iter"]
            return step

    def save_checkpoint(self, evaler_summaries: Optional[Dict[str, Any]]) -> Optional[int]:
        """Saves a checkpoint (subject to checkpointer policy)."""
        cfg: SpmdTrainer.Config = self.config
        with self.mesh():
            ckpt_state = self._trainer_state._asdict()
            if cfg.save_input_iterator:
                ckpt_state["input_iter"] = self._input_iter
            self.checkpointer.save(
                step=self.step, state=ckpt_state, evaler_summaries=evaler_summaries
            )

    def _restore_from_builder(self) -> Optional[TrainerStateBuilder.State]:
        """Restores trainer state by building it with init_state_builder."""
        logging.info("Initializing trainer state with init_state_builder")
        with self.mesh():
            input_state_type = self.init_state_builder.input_state_type()
            if input_state_type == TrainerStateBuilder.StateType.TENSOR_SPECS:
                input_trainer_state = self._trainer_state_specs
            else:
                assert input_state_type == TrainerStateBuilder.StateType.TENSORS, input_state_type
                input_trainer_state = self._trainer_state
            built_state: TrainerStateBuilder.State = self.init_state_builder(
                TrainerStateBuilder.State(
                    step=self._step, trainer_state=input_trainer_state, built_keys=set()
                )
            )
        logging.info(
            "Successfully built trainer state: step=%s built_keys=%s",
            built_state.step,
            built_state.built_keys,
        )
        return built_state

    def _run_step(
        self, input_batch: NestedTensor, *, force_run_evals: Optional[Set[str]] = None
    ) -> NestedTensor:
        """Runs a single training step.

        Args:
            input_batch: a NestedTensor containing global arrays.
            force_run_evals: Whether to force run evalers and return summaries.
                If None, do not force run evalers and no evaler summaries are returned;
                if given as a set of strings, force run all the evalers with the name in
                the set and return summaries.

        Returns:
            A dict containing 'loss' and 'aux' outputs. If force_run_evals is a set,
            force run the evalers in the set and return 'evaler_summaries' output.
        """
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            # Note(Jan 2022):
            # pjit currently requires all parameters to be specified as positional args.
            self._trainer_state, outputs = self._jit_train_step(self._trainer_state, input_batch)

        if self.step % 100 == 0 or 0 <= self.step <= 5:
            self._step_log(
                "loss=%s aux=%s",
                outputs["loss"],
                jax.tree_util.tree_map(
                    lambda x: x.item() if x.ndim == 0 else f"T{x.shape}", outputs["aux"]
                ),
            )

        self.summary_writer(self.step, {"loss": outputs["loss"], **outputs["summaries"]})

        # Aggregate summaries across evalers.
        evaler_summaries = self._run_eval(
            train_summaries=outputs["summaries"], force_runs=force_run_evals
        )

        # Checkpointer policy will decide if we should save.
        self.save_checkpoint(evaler_summaries=evaler_summaries)

        return_dict = {"loss": outputs["loss"], "aux": outputs["aux"]}
        # Returns evaler_summaries if force_run_evals is not None or empty set.
        if force_run_evals:
            return_dict["evaler_summaries"] = evaler_summaries

        return return_dict

    def _run_eval(
        self,
        *,
        train_summaries: Optional[NestedTensor] = None,
        force_runs: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluations and returns the corresponding summaries."""
        evaler_summaries = {}
        # Note: we will use the same eval key as the training keys of the future step,
        # which should be okay.
        prng_key = self._trainer_state.prng_key
        for evaler_name, evaler in self._evalers.items():
            prng_key, summaries, _ = evaler.eval_step(
                self.step,
                prng_key=prng_key,
                model_params=self.model_params_for_eval(),
                train_summaries=train_summaries,
                force_run=bool(force_runs is not None and evaler_name in force_runs),
            )
            evaler_summaries[evaler_name] = summaries
        return evaler_summaries

    def _pjit_train_step(self) -> jax.stages.Wrapped:
        return pjit(
            self._train_step,
            in_shardings=(
                self._trainer_state_partition_specs,
                self._train_step_input_partition_specs(),
            ),
            out_shardings=(
                self._trainer_state_partition_specs,
                dict(
                    summaries=None,
                    loss=None,
                    aux=None,
                ),
            ),
            donate_argnums=(0,),  # donate the state
        )

    def compile_train_step(self) -> jax.stages.Compiled:
        with self.mesh():
            # Do not run init(), which require real devices.
            trainer_state_specs = jax.tree_util.tree_map(
                lambda spec: jax.ShapeDtypeStruct(shape=spec.shape, dtype=spec.dtype),
                self.trainer_state_specs,
            )
            input_batch_specs = jax.tree_util.tree_map(
                lambda tf_spec: jax.ShapeDtypeStruct(
                    shape=tf_spec.shape, dtype=tf_spec.dtype.as_numpy_dtype
                ),
                self.input.dataset().element_spec,
            )
            jit_train_step = self._pjit_train_step()
            lowered_train_step = jit_train_step.lower(trainer_state_specs, input_batch_specs)
            return lowered_train_step.compile()

    def _train_step(
        self,
        state: TrainerState,
        input_batch: Dict[str, Any],
    ) -> Tuple[TrainerState, NestedTensor]:
        cfg = self.config
        # Shard and (possibly) dispatch the input batch.
        if hasattr(self.input, "dispatch_global_batch"):
            input_batch = self.input.dispatch_global_batch(
                input_batch, batch_axis_names=cfg.batch_axis_names
            )

        new_prng_key, param_noise_key, forward_key, learner_key = jax.random.split(
            state.prng_key, 4
        )

        def train_cast(in_tree):
            return utils.cast_floats(in_tree, to_dtype=cfg.train_dtype)

        # A nested tree of booleans.
        should_compute_gradients = self.learner.should_update_with_optimizers(state.model)
        for path, value in flatten_items(should_compute_gradients):
            if not value:
                self.vlog(1, "Skipping gradients on %s", path)

        def _forward(*, inputs: NestedTensor, model_params: NestedTensor) -> ForwardOutputs:
            params = train_cast(model_params)
            params = self.model.apply_parameter_noise_recursively(param_noise_key, params)
            model_output_collection = new_output_collection()
            with child_context(
                "model",
                module=self.model,
                state=params,
                prng_key=forward_key,
                output_collection=model_output_collection,
            ):
                loss, aux = self.model(input_batch=train_cast(inputs))
            return ForwardOutputs(loss=loss, aux=aux, output_collection=model_output_collection)

        # `grads` are computed for `model_parameters_grad`.
        opt_params = self._opt_params(state.model)
        fwd_bwd_outputs, learner_output_collection = F(
            self.learner,
            method="forward_and_backward",
            state=state.learner,
            is_training=True,
            prng_key=learner_key,
            inputs=dict(fn=_forward, opt_params=opt_params, inputs=input_batch),
        )
        forward_outputs: ForwardOutputs = fwd_bwd_outputs.forward_outputs
        updated_model_params = fwd_bwd_outputs.backward_outputs.updated_params
        updated_state = TrainerState(
            prng_key=new_prng_key,
            model=updated_model_params,
            learner=learner_output_collection.state_updates,
        )
        # TODO(ruoming): only retrieve summaries when necessary.
        summaries = dict(
            model=forward_outputs.output_collection.summaries,
            learner=learner_output_collection.summaries,
        )
        return updated_state, dict(
            summaries=summaries,
            loss=forward_outputs.loss,
            aux=forward_outputs.aux,
        )

    def _maybe_stop_or_start_tracing(
        self, stop_trace_step: Optional[int], output: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Stops or starts jax profiler tracing if necessary.

        Args:
            stop_trace_step: The step at which we should stop tracing.
            output: The output of run_step.

        Returns:
            The updated value for `stop_trace_step`.
        """
        updated_stop_trace_step = stop_trace_step
        # Check if we should stop tracing.
        if self.step == stop_trace_step:
            assert output is not None
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
            jax.profiler.stop_trace()
            self._step_log("Stopped profiler tracing")
            updated_stop_trace_step = None

        # Check if we should start tracing.
        cfg = self.config
        if self.step not in cfg.start_trace_steps:
            should_start_tracing = False
        elif updated_stop_trace_step is not None:
            logging.warning(
                "Skipping trace at step %s, since it is too close to the previous one: %s",
                self.step,
                cfg.start_trace_steps,
            )
            should_start_tracing = False
        else:
            should_start_tracing = (
                cfg.start_trace_process_indices == "all"
                or jax.process_index() in cfg.start_trace_process_indices
            )
        if should_start_tracing:
            self._step_log("Start profiler tracing")
            jax.profiler.start_trace(self.summary_writer.config.dir)
            updated_stop_trace_step = self.step + 3
        return updated_stop_trace_step


def select_mesh_config(trainer_config: SpmdTrainer.Config, *, mesh_selector: str):
    """Selects a mesh rule (if one matches `mesh_selector` to override mesh config.

    If any of `trainer_config.mesh_rules` matches `mesh_selector`, modifies
    `trainer_config.mesh_shape` according to the rule.

    Args:
        trainer_config: The trainer config. Will be modified if any mesh rule matches.
        mesh_selector: A string used to select the mesh rule to apply.
    """
    if trainer_config.mesh_rules:
        mesh = match_regex_rules(
            mesh_selector, rules=trainer_config.mesh_rules, default_value=REQUIRED
        )
        logging.info("Mesh selector %s matches mesh rule %s", mesh_selector, mesh)
        if mesh is not REQUIRED:
            trainer_config.mesh_shape = mesh
