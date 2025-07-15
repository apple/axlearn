# Copyright © 2023 Apple Inc.

"""Defines SpmdTrainer, a trainer that supports partitioning of computation and data with GSPMD."""

import contextlib
import itertools
import math
import os.path
import signal
import threading
import time
from collections.abc import Sequence
from typing import Any, Callable, ContextManager, Literal, NamedTuple, Optional, Union

import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit

from axlearn.common import file_system as fs
from axlearn.common import measurement, utils
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import BaseCheckpointer, Checkpointer
from axlearn.common.compiler_options import infer_xla_performance_flags, infer_xsc_compiler_options
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
    maybe_set_config,
)
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.input_base import Input
from axlearn.common.learner import Learner
from axlearn.common.module import InvocationContext, Module, child_context, clone_context_stack
from axlearn.common.module import functional as F
from axlearn.common.module import install_context_stack, new_output_collection
from axlearn.common.monitoring.device_monitor import DeviceMonitor
from axlearn.common.optimizer_base import NestedOptParam, OptParam
from axlearn.common.param_init import DefaultInitializer
from axlearn.common.state_builder import Builder as TrainerStateBuilder
from axlearn.common.summary_writer import BaseWriter, SummaryWriter
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import (
    HybridMeshShape,
    MeshShape,
    Nested,
    NestedTensor,
    PartitionSpec,
    PerParamFn,
    Tensor,
    TensorSpec,
    canonicalize_per_param_dtype,
    count_model_params,
    flatten_items,
    host_to_global_specs,
    match_regex_rules,
    thread_stack_traces,
)


class TrainerState(NamedTuple):
    prng_key: Union[Tensor, TensorSpec, jax.sharding.NamedSharding]
    model: Union[NestedTensor, Nested[TensorSpec], Nested[jax.sharding.NamedSharding]]
    learner: Union[NestedTensor, Nested[TensorSpec], Nested[jax.sharding.NamedSharding]]


# pylint: disable-next=too-many-instance-attributes
class SpmdTrainer(Module):
    """A trainer implementation that supports partitioning of computation and data with GSPMD."""

    @config_class
    # pylint: disable-next=too-many-instance-attributes
    class Config(Module.Config):
        """Configures SpmdTrainer."""

        # The input source.
        input: Required[Input.Config] = REQUIRED

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
        # TODO(markblee): Deprecate this field in favor of `input.input_partitioner`.
        batch_axis_names: Optional[Union[str, Sequence[str]]] = "data"

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
        mesh_rules: Optional[Sequence[tuple[str, Optional[MeshShape]]]] = None

        # The model config.
        model: Required[BaseModel.Config] = REQUIRED
        # The learner config.

        learner: Required[Learner.Config] = REQUIRED
        # The checkpointer config.
        checkpointer: BaseCheckpointer.Config = Checkpointer.default_config()
        # A dict of evaler names to configs, each name must be non-empty.
        evalers: dict[str, SpmdEvaler.Config] = {}

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

        # Determines whether to run the XLA Silent-data-corruption Checker (XSC) for a given step.
        # If None, never run the checker.
        # N.B. if provided on backends other than TPU this will be a no-op with warning logs.
        # N.B. XSC will not detect repeatable defects in TPU SparseCores.
        xsc_check_policy: Optional[ConfigOr[Callable[[int], bool]]] = None

        # Prune empty state updates.
        # Must be set to True.
        # The configuration option will be removed in a future AXLearn version.
        # It has not been removed yet to prevent the need for a messy regeneration of large numbers
        # of golden configs.
        prune_empty_state_updates: bool = True

        # Cast float inputs and model parameters to dtype for the train step.
        # This parameter can either be:
        # 1. A `jnp.dtype`, where both float inputs and model parameters will
        #    be cast to this dtype.
        # 2. A `ConfigOr[PerParamFn[jnp.dtype]]`, allowing different dtypes to be applied to
        #    different parameters during training.
        # If not provided, the default value is `None`, no casting applied.
        train_dtype: Optional[Union[jnp.dtype, ConfigOr[PerParamFn[jnp.dtype]]]] = None

        # If > 0, run a watchdog thread to print the thread stack traces if step does not
        # increment within this interval.
        watchdog_timeout_seconds: Optional[float] = None

        # If > 0, crash the program if the watchdog thread suspect a hanging
        # after this interval.
        # The crash is only trigggered after the trainer is initialized:
        # (1) device_monitor enabled and the device monitor detects the host idleness or
        # (2) watchdog_timeout_seconds is triggered without a device_monitor,
        # both are indications of system hanging.
        crash_on_hang_timeout_seconds: Optional[float] = None

        # Device monitor to check if the devices are idle.
        # TODO(kelvin-zou): integrate with watchdog function.
        device_monitor: Optional[DeviceMonitor.Config] = None

        # An optional recorder for measuring common metrics like step time.
        recorder: Optional[InstantiableConfig[measurement.Recorder]] = None

        # An additional context manager to run the training loop and initialization inside of.
        # The provided config should instantiate to a thunk that returns the context manager.
        context_manager: Optional[ConfigOr[Callable[[], ContextManager]]] = None

        # If False, assumes the train_step may need to be recompiled and go through the lowering
        # and compilation process every train step and rely on compilation cache to prevent
        # excessive recompilations. Note: this could introduce overhead to training due to
        # pre-compilation checks (such as sharding check) that increases the step time for some
        # models. Note that this cache is always disabled at steps when xsc is enabled.
        # Defaults to None which is interpreted as True.
        cache_compiled_train_step: Optional[bool] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        devices: Optional[np.ndarray] = None,
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
        self._device_monitor = maybe_instantiate(cfg.device_monitor)
        self._recorder = maybe_instantiate(cfg.recorder)
        self._is_initialized: bool = False
        # Accelerator initialization.
        with self._record_event(measurement.Event.ACCELERATOR_INIT):
            self._device_init(devices)

        # Create all children within the mesh context so that utils.input_partition_spec() works
        # properly.
        with self.mesh():
            if cfg.batch_axis_names is not None:
                cfg.input = maybe_set_config(
                    cfg.input, partition_spec=PartitionSpec(cfg.batch_axis_names)
                )
            self.input: Input = self._add_child(
                "input", maybe_set_config(cfg.input, is_training=True)
            )
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
            model_param_partition_specs = jax.tree.map(
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
            self._trainer_state_partition_specs: TrainerState = jax.tree.map(
                lambda spec: spec.sharding, self._trainer_state_specs
            )
            # Create evalers, which depend on model_param_partition_specs.
            self._evalers = {}
            for evaler_name, evaler_cfg in cfg.evalers.items():
                evaler_cfg.summary_writer.dir = evaler_cfg.summary_writer.dir or os.path.join(
                    cfg.dir, "summaries", evaler_name
                )
                if cfg.batch_axis_names is not None:
                    maybe_set_config(
                        evaler_cfg.input, partition_spec=PartitionSpec(cfg.batch_axis_names)
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

    @contextlib.contextmanager
    def _record_event(self, event: measurement.Event, *args, **kwargs):
        """A helper to record an event if a recorder is configured."""
        if self._recorder:
            with self._recorder.record_event(event, *args, **kwargs) as event_manager:
                yield event_manager
        else:
            yield

    def _device_init(self, devices: Optional[np.ndarray] = None):
        """Initializes the device mesh and other device-dependent configurations."""
        cfg = self.config
        if cfg.model.dtype is None:
            raise ValueError(f"dtype must be explicitly specified for {self.path()}.model")
        if cfg.model.param_init is None:
            cfg.model.param_init = DefaultInitializer.default_config()
            logging.info(
                "model.param_init is not specified. Default to DefaultInitializer: %s",
                cfg.model.param_init,
            )

        self._per_param_train_dtype = maybe_instantiate(
            canonicalize_per_param_dtype(cfg.train_dtype)
        )

        # Create the device mesh.
        if devices is None:
            self._step_log(
                "Devices: global=%s local=%s %s",
                jax.device_count(),
                jax.local_device_count(),
                [device.platform for device in jax.local_devices()],
            )
        else:
            local_devices = [d for d in devices.flatten() if d.process_index == jax.process_index()]
            self._step_log(
                "Devices: global=%s local=%s %s",
                len(devices),
                len(local_devices),
                [device.platform for device in local_devices],
            )
        self._step_log("Mesh shape: %s", cfg.mesh_shape)
        devices = (
            utils.create_device_mesh(mesh_shape=cfg.mesh_shape) if devices is None else devices
        )
        mesh = jax.sharding.Mesh(devices, cfg.mesh_axis_names)
        self._step_log("Global mesh: %s", mesh)
        self._mesh = mesh
        self._context_manager: Callable[[], ContextManager] = (
            maybe_instantiate(cfg.context_manager) or contextlib.nullcontext
        )
        xsc_check_policy = None
        if cfg.xsc_check_policy:
            if jax.default_backend() != "tpu":
                # XSC is currently only supported on TPU XLA backend.
                logging.warning(
                    "xsc_check_policy was set for non-TPU XLA backend. Running without XSC."
                )
            else:
                xsc_check_policy = maybe_instantiate(cfg.xsc_check_policy)
        self._xsc_check_policy: Optional[Callable[[int], bool]] = xsc_check_policy
        self._compiled_train_step: Optional[jax.stages.Compiled] = None

    def _train_step_input_partition_specs(self):
        # Note that subclasses may override this method to set a partition spec for pjit which is
        # different from that of the input partition spec.
        return self.input.partition_spec

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
        if (
            cfg.watchdog_timeout_seconds or cfg.crash_on_hang_timeout_seconds
        ) and self._watchdog_thread is None:
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

    def _watchdog_loop(self, *, context_stack: list[InvocationContext]):
        cfg: SpmdTrainer.Config = self.config
        install_context_stack(context_stack)
        time_elapsed_in_sec_since_last_check: float = 0.0
        # Set a scanning time to 10 mins or the watchdog_timeout_seconds, whichever is smaller.
        health_check_in_sec = 600
        if cfg.watchdog_timeout_seconds is not None:
            health_check_in_sec = min(cfg.watchdog_timeout_seconds, health_check_in_sec)
        job_hang_suspected = False
        while True:
            last_step = self.step
            if self._watchdog_stopping.wait(health_check_in_sec):
                break
            current_step = self.step
            if current_step == last_step:
                time_elapsed_in_sec_since_last_check += health_check_in_sec
                # When device_monitor is enabled, we can check if the host is idle
                # and trigger the watchdog proactively.
                if self._device_monitor is not None:
                    if self._device_monitor.is_host_idle():
                        self._step_log(
                            "Watchdog triggered because step has not incremented in the last %s "
                            "seconds and the host is idle.\n"
                            "NOTE: this is not an error message, but meant to help debugging "
                            "in case the trainer is stuck.\n"
                            "Threads:\n%s",
                            time_elapsed_in_sec_since_last_check,
                            "\n".join(
                                itertools.chain.from_iterable(thread_stack_traces()),
                            ),
                        )
                        job_hang_suspected = True
                # Without device_monitor, we still want to log the thread stack traces
                # when the trainer is stuck at cfg.watchdog_timeout_seconds.
                elif (
                    cfg.watchdog_timeout_seconds is not None
                    and time_elapsed_in_sec_since_last_check >= cfg.watchdog_timeout_seconds
                ):
                    self._step_log(
                        "Watchdog triggered because step has not incremented in the last %s "
                        "seconds.\n NOTE: this is not an error message, but meant to help "
                        "debugging in case the trainer is stuck.\n"
                        "Threads:\n%s",
                        time_elapsed_in_sec_since_last_check,
                        "\n".join(itertools.chain.from_iterable(thread_stack_traces())),
                    )
                    job_hang_suspected = True
                # Crash the program here to trigger a job restart outside.
                # Crash after crash_on_hang_timeout_seconds after initialization.
                if (
                    cfg.crash_on_hang_timeout_seconds is not None
                    and time_elapsed_in_sec_since_last_check >= cfg.crash_on_hang_timeout_seconds
                    and job_hang_suspected
                    and self._is_initialized
                ):
                    logging.error("Exit due to no progress during training.")
                    os.kill(os.getpid(), signal.SIGKILL)
            else:
                self.vlog(1, "Watchdog check passed: %s -> %s", last_step, current_step)
                time_elapsed_in_sec_since_last_check = 0
        logging.info("Watchdog loop done")

    def _should_force_run_evals(
        self,
        *,
        return_evaler_summaries: Optional[Union[bool, set[str]]] = None,
        evalers: dict[str, SpmdEvaler.Config],
    ) -> set[str]:
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
        elif isinstance(return_evaler_summaries, set):
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

    # pylint: disable-next=too-many-statements,too-many-branches
    def run(
        self, prng_key: Tensor, *, return_evaler_summaries: Optional[Union[bool, set[str]]] = None
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
        goodput_monitor_manager = (
            self._recorder.maybe_monitor_goodput()
            if hasattr(self._recorder, "maybe_monitor_goodput")
            else contextlib.nullcontext()
        )
        rolling_goodput_monitor_manager = (
            self._recorder.maybe_monitor_rolling_window_goodput()
            if hasattr(self._recorder, "maybe_monitor_rolling_window_goodput")
            else contextlib.nullcontext()
        )

        with (
            (
                self._device_monitor.start_monitoring()
                if self._device_monitor is not None
                else contextlib.nullcontext()
            ),
            self._watchdog(),
            self.mesh(),
            jax.log_compiles(self.vlog_is_on(1)),
            self._context_manager(),
            goodput_monitor_manager,
            rolling_goodput_monitor_manager,
        ):
            cfg = self.config
            # Check if need to force run evals at the last training step.
            force_run_eval_sets_at_max_step = self._should_force_run_evals(
                return_evaler_summaries=return_evaler_summaries, evalers=cfg.evalers
            )

            # Prepare training.
            with self._record_event(measurement.Event.TRAINING_PREPARATION):
                if not self._prepare_training(prng_key):
                    return None

            self._is_initialized = True

            with self.checkpointer:
                logging.info("Starting loop...")
                start_time = time.perf_counter()
                num_steps = 0
                output = None
                stop_trace_step = None

                input_iterator = self.input.batches(self._input_iter)
                while True:
                    try:
                        with self._record_event(measurement.Event.DATA_LOADING):
                            input_batch = next(input_iterator)

                        logging.log_first_n(
                            logging.INFO, "host_input_batch=%s", 3, utils.shapes(input_batch)
                        )

                        # Stop or start tracing if necessary.
                        stop_trace_step = self._maybe_stop_or_start_tracing(stop_trace_step, output)

                        self._step = self._step + 1
                        self.vlog(3, "Start step %s", self.step)
                        with self._record_event(measurement.Event.STEP):
                            output = self._run_step(
                                utils.host_to_global_array(
                                    input_batch,
                                    partition=self._train_step_input_partition_specs(),
                                ),
                                force_run_evals=(
                                    force_run_eval_sets_at_max_step
                                    if self.step >= cfg.max_step
                                    else None
                                ),
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
                    except StopIteration:
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
        return jax.tree.map(
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
            self._trainer_state = jax.tree.map(
                lambda state, spec: jax.device_put(state, spec.sharding),
                built_state.trainer_state,
                self._trainer_state_specs,
            )

    def _init_with_prebuilt_state(
        self,
        prng_key: Tensor,
        *,
        prebuilt_state: Optional[TrainerStateBuilder.State],
    ):
        """Initializes `self._step` and `self._trainer_state`, optionally from `prebuilt_state`.

        If `prebuilt_state` contains the complete trainer state, sets it as `self._trainer_state`.

        Otherwise initializes model parameters by copying from `prebuilt_state` if available,
        otherwise with `prng_key`. `prebuilt_state` is expected to contain a subset of the model
        parameters and none of the learner state. Specifically, `prebuilt_state.trainer_state.model`
        should have the same structure as the model params and each leaf node is either a Tensor
        (a prebuilt param) or a ParameterSpec (a param to be initialized).

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
        all_trainer_state_keys = {key for key, _ in utils.flatten_items(self.trainer_state_specs)}
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

        # A tree where a leaf is a ParameterSpec for a prebuilt param, None otherwise.
        # This is used for `initialize_parameters_recursively` inside `_init_state`.
        prebuilt_model_param_specs = jax.tree.map(
            lambda value, spec: spec if isinstance(value, Tensor) else None,
            prebuilt_state.trainer_state.model,
            self._trainer_state_specs.model,
        )
        self.vlog(1, "prebuilt_model_state: %s", utils.shapes(prebuilt_model_param_specs))
        # Partition specs for parameters that are *not* prebuilt and therefore to be initialized.
        #
        # While `prebuilt_state.trainer_state.model` also contain ParameterSpec's, we use
        # `self._trainer_state_partition_specs.model` to ensure that the partition spec matches
        # the model's partition config (rather than coming from `init_state_builder`).
        model_initialization_partition_specs = jax.tree.map(
            lambda value, spec: None if isinstance(value, Tensor) else spec,
            prebuilt_state.trainer_state.model,
            self._trainer_state_partition_specs.model,
        )
        # The output sharding for `_init_state`.
        out_shardings = self._trainer_state_partition_specs._replace(
            model=model_initialization_partition_specs,
        )

        def merge_model_states(
            prebuilt_model_params: Nested[Union[Tensor, ParameterSpec]],
            initialized_model_params: Nested[Optional[NestedTensor]],
        ) -> Nested[Tensor]:
            """Merges prebuilt and initialized params to a single tree."""
            if prebuilt_model_params is None:
                return initialized_model_params
            return jax.tree.map(
                lambda prebuilt, initialized: (
                    prebuilt if isinstance(prebuilt, Tensor) else initialized
                ),
                prebuilt_model_params,
                initialized_model_params,
            )

        def _init_state(prng_key: Tensor) -> TrainerState:
            prng_key, init_key = jax.random.split(prng_key)
            model_params = self.model.initialize_parameters_recursively(
                init_key,
                prebuilt=prebuilt_model_param_specs,
            )
            self.vlog(1, "initialized_model_state: %s", utils.shapes(model_params))
            learner_params = self.learner.init(
                self._opt_params(
                    # Initialize learner with union(prebuilt + initialized).
                    merge_model_states(prebuilt_state.trainer_state.model, model_params)
                )
            )
            return TrainerState(prng_key=prng_key, model=model_params, learner=learner_params)

        init_computation = pjit(
            _init_state,
            in_shardings=(None,),
            out_shardings=out_shardings,
        )
        self._step_log("Initializing trainer state.")
        with self.mesh():
            initialized_trainer_state: TrainerState = init_computation(prng_key)
        # Merge prebuilt and initialized model params.
        merged_model_state = merge_model_states(
            prebuilt_state.trainer_state.model, initialized_trainer_state.model
        )
        self.vlog(1, "merged_model_state: %s", utils.shapes(merged_model_state))
        self._trainer_state = TrainerState(
            prng_key=initialized_trainer_state.prng_key,
            model=merged_model_state,
            learner=initialized_trainer_state.learner,
        )

    def _log_trainer_state_stats(self) -> str:
        total_num_params = count_model_params(self._trainer_state.model)
        analysis_logs = []

        def _step_log(msg, *args, **kwargs):
            self._step_log(msg, *args, **kwargs)
            analysis_logs.append(msg % args)

        _step_log("##################### Model analysis #####################\n")
        _step_log("## Parameters:")
        fmt = "%10d %-20s %s"
        flatten_name_and_spec = flatten_items(self._model_param_specs)
        for name, spec in flatten_name_and_spec:
            spec_size = np.prod(spec.shape)
            _step_log(fmt, spec_size, spec.shape, name)

        _step_log("Total number of model params: %s", f"{total_num_params:,}")
        self.summary_writer(0, {"num_model_params": total_num_params})

        _step_log("\n## Trainer States:")
        # Training state size.
        total_state_bytes = 0
        total_sharded_state_bytes = 0
        state_spec_map = dict(utils.flatten_items(self.trainer_state_specs))
        for path, value in utils.flatten_items(self._trainer_state):
            _step_log(
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

        _step_log(
            "Training state size: %.2f GiB\n"
            "Training state size (partitioned): %.2f GiB\n"
            "Max training state size (partitioned): %.2f GiB",
            total_state_bytes / 1024**3,
            total_sharded_state_gb,
            max_sharded_state_gb,
        )

        _step_log("\n##########################################################")
        return "\n".join(analysis_logs)

    def _prepare_training(self, prng_key: Tensor) -> bool:
        """Prepares training.

        This function does the following to prepare the training procedure:
        1. Restores the trainer state from a checkpoint. If no checkpoint exists,
           initializes a new trainer state using the provided prng_key.
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

        model_analysis = self._log_trainer_state_stats()

        # Log trainer state tree.
        if not self.step and jax.process_index() == 0:
            with fs.open(os.path.join(cfg.dir, "trainer_state_tree.txt"), "w") as f:
                f.write(str(jax.tree_util.tree_structure(self._trainer_state)))

            with fs.open(os.path.join(cfg.dir, "model_analysis.txt"), "w") as f:
                f.write(model_analysis)

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

    def save_checkpoint(self, evaler_summaries: Optional[dict[str, Any]]) -> Optional[int]:
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

    def _get_compiled_train_step_fn(
        self, *, trainer_state: TrainerState, input_batch: NestedTensor, with_xsc: bool = False
    ) -> Callable[[TrainerState, NestedTensor], tuple[TrainerState, NestedTensor]]:
        """Build a fully compiled train step function.

        Relies on the JAX pjit cache to avoid recompilation when with_xsc=True or
        cache_compiled_train_step=False.

        Args:
            train_state: A TrainerState instance.
            input_batch: A NestedTensor containing global arrays.
            with_xsc: Compile the train step with the XLA SDC Checker enabled.

        Returns:
            A train step function with signature matching that of self._jit_train_step's return.

        Raises:
            RuntimeError: If `with_xsc` is requested on heterogenous device kinds.
        """
        if (
            self.config.cache_compiled_train_step is not False
            and not with_xsc
            and self._compiled_train_step is not None
        ):
            return self._compiled_train_step
        cfg: SpmdTrainer.Config = self.config
        # Get device kinds and assert that they are homogenous.
        # TODO(markblee): Get devices from self._mesh.devices.
        device_kinds = set(d.device_kind for d in jax.devices())
        if len(device_kinds) != 1:
            raise RuntimeError(f"Heterogenous device kinds ({device_kinds}) are not supported.")
        device_kind = device_kinds.pop()
        options = infer_xla_performance_flags(
            mesh_shape=cfg.mesh_shape, mesh_axis_names=cfg.mesh_axis_names, device_kind=device_kind
        )
        if not with_xsc:
            with self._record_event(
                measurement.Event.CUSTOM_BADPUT_EVENT,
                custom_badput_event_type="COMPILATION_NO_XSC",
            ):
                self._compiled_train_step = self.compile_train_step(
                    trainer_state=trainer_state, input_batch=input_batch, compiler_options=options
                )
            return self._compiled_train_step

        logging.log_first_n(logging.INFO, "Compiling XSC train step.", 1)

        with self._record_event(
            measurement.Event.CUSTOM_BADPUT_EVENT,
            custom_badput_event_type="COMPILATION_WITH_XSC",
        ):
            compiled_jit_train_step_fn = self.compile_train_step(
                trainer_state=trainer_state,
                input_batch=input_batch,
                compiler_options=options
                | infer_xsc_compiler_options(
                    halt_on_detection=True, repeat_count=1, device_kind=device_kind
                ),
            )
        return compiled_jit_train_step_fn

    def _run_step(
        self, input_batch: NestedTensor, *, force_run_evals: Optional[set[str]] = None
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
        logging.log_first_n(logging.INFO, "global_input_batch=%s", 3, utils.shapes(input_batch))
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            run_with_xsc = self._xsc_check_policy and self._xsc_check_policy(self.step)
            compiled_train_step_fn = self._get_compiled_train_step_fn(
                trainer_state=self.trainer_state, input_batch=input_batch, with_xsc=run_with_xsc
            )
            # Run the compiled function.
            self._trainer_state, outputs = compiled_train_step_fn(self.trainer_state, input_batch)

        if self.step % 100 == 0 or 0 <= self.step <= 5:
            self._step_log(
                "loss=%s aux=%s",
                outputs["loss"],
                jax.tree.map(lambda x: x.item() if x.ndim == 0 else f"T{x.shape}", outputs["aux"]),
            )

        with self._record_event(
            measurement.Event.CUSTOM_BADPUT_EVENT, custom_badput_event_type="SUMMARY_WRITER"
        ):
            self.summary_writer(self.step, {"loss": outputs["loss"], **outputs["summaries"]})
        # Aggregate summaries across evalers.
        evaler_summaries = self._run_eval(
            train_summaries=outputs["summaries"], force_runs=force_run_evals
        )

        # Checkpointer policy will decide if we should save.
        with self._record_event(
            measurement.Event.CUSTOM_BADPUT_EVENT, custom_badput_event_type="CHECKPOINT_SAVE"
        ):
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
        force_runs: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Runs evaluations and returns the corresponding summaries."""
        with self._record_event(
            measurement.Event.CUSTOM_BADPUT_EVENT, custom_badput_event_type="EVAL"
        ):
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

    def compile_train_step(
        self,
        *,
        trainer_state: Optional[TrainerState] = None,
        input_batch: Optional[dict[str, Any]] = None,
        compiler_options: Optional[dict[str, Union[str, bool]]] = None,
    ) -> jax.stages.Compiled:
        """Produce a lowered and compiled training step.

        Args:
            trainer_state: The global trainer state (or state specs).
                If None, infer from self.trainer_state_specs.
            input_batch: An input batch (or specs for the global input batch).
                If None, attempt to infer from the (host-local) input element spec.
            compiler_options: Options passed to the XLA compiler, selectively overwriting
                any settings already provided by environment variables for this compilation.

        Returns:
            A compiled training step, with signature matching self._pjit_train_step's return.
        """
        with self.mesh(), self._context_manager():
            if trainer_state is None:
                # Do not run init(), which requires real devices.
                trainer_state = jax.tree.map(
                    lambda spec: jax.ShapeDtypeStruct(shape=spec.shape, dtype=spec.dtype),
                    self.trainer_state_specs,
                )
            if input_batch is None:
                # Infer global input batch shapes from input element spec.
                host_batch = self.input.element_spec()
                if "input_dispatcher" in self.input.children:
                    host_batch = self.input.input_dispatcher.logical_to_physical_shapes(host_batch)
                input_batch = host_to_global_specs(
                    host_batch, partition=self._train_step_input_partition_specs()
                )

            # Rely on the instance handle to ensure that we hit the compilation cache if possible.
            jit_train_step = self._jit_train_step or self._pjit_train_step()
            # Note(Jan 2022):
            # pjit currently requires all parameters to be specified as positional args.
            lowered_train_step = jit_train_step.lower(trainer_state, input_batch)
            compiled = lowered_train_step.compile(compiler_options=compiler_options)
            logging.log_first_n(logging.INFO, aot_model_analysis(compiled), 1)
            return compiled

    def _train_step(
        self,
        state: TrainerState,
        input_batch: dict[str, Any],
    ) -> tuple[TrainerState, NestedTensor]:
        # Shard and (possibly) dispatch the input batch.
        input_batch = self.input.dispatch_global_batch(input_batch)
        new_prng_key, param_noise_key, forward_key, learner_key = jax.random.split(
            state.prng_key, 4
        )

        def train_cast(in_tree):
            per_param_train_dtype = self._per_param_train_dtype(in_tree)
            return utils.cast_floats_per_param(in_tree, per_param_train_dtype)

        # A nested tree of booleans.
        should_compute_gradients = self.learner.should_update_with_optimizers(state.model)
        for path, value in flatten_items(should_compute_gradients):
            if not value:
                self.vlog(1, "Skipping gradients on %s", path)

        def _forward(*, inputs: NestedTensor, model_params: NestedTensor) -> ForwardOutputs:
            params = train_cast(model_params)
            params = self.model.apply_parameter_noise_recursively(inputs["param_noise_key"], params)
            model_output_collection = new_output_collection()
            with child_context(
                "model",
                module=self.model,
                state=params,
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                loss, aux = self.model(input_batch=train_cast(inputs["input_batch"]))
            return ForwardOutputs(loss=loss, aux=aux, output_collection=model_output_collection)

        # `grads` are computed for `model_parameters_grad`.
        opt_params = self._opt_params(state.model)
        fwd_bwd_outputs, learner_output_collection = F(
            self.learner,
            method="forward_and_backward",
            state=state.learner,
            is_training=True,
            prng_key=learner_key,
            inputs=dict(
                fn=_forward,
                opt_params=opt_params,
                inputs=dict(
                    input_batch=input_batch,
                    forward_key=forward_key,
                    param_noise_key=param_noise_key,
                ),
            ),
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
        self, stop_trace_step: Optional[int], output: Optional[dict[str, Any]]
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
            jax.tree.map(lambda x: x.block_until_ready(), output)
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
        mesh_rule = match_regex_rules(
            mesh_selector, rules=trainer_config.mesh_rules, default_value=REQUIRED
        )
        logging.info("Mesh selector %s matches mesh rule %s", mesh_selector, mesh_rule)
        if mesh_rule is not REQUIRED:
            # Mesh config is just mesh rule or hybrid mesh rule.
            if isinstance(mesh_rule, (tuple, HybridMeshShape)) or mesh_rule is None:
                trainer_config.mesh_shape = mesh_rule
            else:
                # Override configs from ConfigModifier.
                mesh_rule_fn = maybe_instantiate(mesh_rule)
                trainer_config = mesh_rule_fn(trainer_config)


def aot_model_analysis(compiled: jax.stages.Compiled) -> str:
    """Performs the model analysis on the AOT compiled JAX program.

    Refer to https://docs.jax.dev/en/latest/jax.stages.html#jax.stages.Compiled

    Note: memory_analysis() and cost_analysis() are internal statistics used by the XLA compiler,
    and there is no official documentation for them.
    The human-readable interpretation provided here is based on best guesses from reviewing
    the XLA source code. If there are any inaccuracies, please update accordingly.
    * memory_analysis:
    https://github.com/openxla/xla/blob/101045ad079d17701986060666feda0e70d6c4cf/xla/pjrt/pjrt_executable.h#L284
    * cost_analysis:
    https://github.com/openxla/xla/blob/101045ad079d17701986060666feda0e70d6c4cf/xla/service/hlo_cost_analysis.h#L41

    Args:
        compiled: The compiled JAX program.

    Returns:
        memory_analysis: String, model analysis results.
    """
    # e.g. _CheckifyCompiledFnWrapper doesn't have memory_analysis attribute.
    if not hasattr(compiled, "memory_analysis"):
        return ""

    def m_or_g(x, suffix=""):
        if x is None:
            return None
        m = 1024**2
        g = 1024**3
        if x > g:
            return f"{x / g:.1f}G{suffix}"
        else:
            return f"{x / m:.1f}M{suffix}"

    mb_or_gb = lambda x: m_or_g(x, "B")
    analysis_results = ""
    mem_stats = compiled.memory_analysis()
    # According to the doc, some platforms may not support it.
    if mem_stats is not None:
        analysis_results += "======= Memory Analysis ==================================\n"
        try:
            total_hbm = (
                mem_stats.argument_size_in_bytes
                + mem_stats.output_size_in_bytes
                + mem_stats.temp_size_in_bytes
                + mem_stats.generated_code_size_in_bytes
            )
            analysis_results += (
                f"Input memory: {mb_or_gb(mem_stats.argument_size_in_bytes)}\n"
                + f"Output memory: {mb_or_gb(mem_stats.output_size_in_bytes)}\n"
                + f"Temp memory: {mb_or_gb(mem_stats.temp_size_in_bytes)}\n"
                + f"Code memory: {mb_or_gb(mem_stats.generated_code_size_in_bytes)}\n"
                + f"Total HBM memory: {mb_or_gb(total_hbm)}\n"
            )
        except AttributeError as e:
            # Some platforms may return different format.
            analysis_results += f"{mem_stats}\n"
            logging.warning("Attempt to parse mem_stats=%s failed: %s", mem_stats, e)

    cost_stats = compiled.cost_analysis()
    analysis_results += "======= Cost Analysis ====================================\n"
    # According to the doc, some platforms may not support it.
    if cost_stats and isinstance(cost_stats, list):
        cost_stats = cost_stats[0]
    if cost_stats and isinstance(cost_stats, dict):
        analysis_results += (
            f"FLOPS: {m_or_g(cost_stats.get('flops'))}\n"
            + f"The number of exp/log/sin/cos ops: {m_or_g(cost_stats.get('transcendentals'))}\n"
            + f"The total memory traffic: {mb_or_gb(cost_stats.get('bytes accessed'))}\n"
            + f"  HBM access: {mb_or_gb(cost_stats.get('bytes accessed0{}'))}\n"
            + f"  L2 cache access: {mb_or_gb(cost_stats.get('bytes accessed1{}'))}\n"
            + f"  Register usage: {mb_or_gb(cost_stats.get('bytes accessed2{}'))}\n"
            + f"  Output data transferred: {mb_or_gb(cost_stats.get('bytes accessedout{}'))}\n"
            + "Hardware utilization scores\n"
            + f"  Tensor Cores / MatMul units: {cost_stats.get('utilization0{}')}\n"
            + f"  ALU (Arithmetic Logic Unit): {cost_stats.get('utilization1{}')}\n"
            + f"  Memory Load/Store Units: {cost_stats.get('utilization2{}')}\n"
            + f"  L1 Cache Operations: {cost_stats.get('utilization3{}')}\n"
            + f"  L2 Cache Operations: {cost_stats.get('utilization4{}')}\n"
            + f"  Special Function Units (exp/log/sin/cos): {cost_stats.get('utilization5{}')}\n"
            + f"  Integer Units (for indexing, loop counters): {cost_stats.get('utilization6{}')}\n"
            + f"  Branch Divergence (Control Flow Processing): {cost_stats.get('utilization7{}')}\n"
            + f"  Load Balancing / Dispatch): {cost_stats.get('utilization8{}')}\n"
            + f"  Texture Units (or Rarely Used Compute Units): {cost_stats.get('utilization9{}')}"
        )
    else:
        # Some platforms may return different format unlike CPU, TPU (v5p) and GPU (H100).
        analysis_results += f"{cost_stats}\n"
        logging.warning("Attempt to parse cost_stats=%s but failed.", cost_stats)

    return analysis_results
