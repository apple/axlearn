# Copyright © 2023 Apple Inc.

"""Evaler and base metric calculators."""

import functools
import graphlib
import os.path
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec

from axlearn.common import flax_struct, input_base, summary_writer, utils
from axlearn.common.base_model import BaseModel
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_set_config,
)
from axlearn.common.inference_output import BaseOutputWriter
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import Module, OutputCollection
from axlearn.common.module import functional as F
from axlearn.common.utils import (
    NestedPartitionSpec,
    NestedTensor,
    Tensor,
    input_partition_spec,
    replicate_to_local_data,
    with_sharding_constraint,
)


class BaseMetricCalculator(Module):
    """The base class of classes to calculate evaluation metrics.

    Pseudo-code for using a metric_calculator.

    During evaler `__init__` (will be called within the trainer mesh context):
        self._add_child(
            "metric_calculator", cfg.metric_calculator,
            model=model, model_param_partition_specs=model_param_partition_specs)

    For each eval step:
        state = self.metric_calculator.init_state(prng_key=prng_key, model_params=model_params)
        outputs = []

        for input_batch in eval_inputs:
            # Compute outputs for `input_batch`.
            forward_output = self.metric_calculator.forward(
                input_batch, model_params=model_params, state=state)
            # Update `state` and save the per-batch outputs.
            state = forward_output["state"]
            outputs.append(forward_output["output"])

        # Finalize the outputs to get the summaries.
        summaries = self.metric_calculator.get_summaries(
            model_params=model_params, state=state, all_forward_outputs=outputs)
    """

    @config_class
    class Config(Module.Config):
        """Configures BaseMetricCalculator."""

        # Cast float inputs and parameters to this dtype for the evaluation step.
        # If None, do not cast.
        eval_dtype: Optional[jnp.dtype] = None

        # Optional prefix for formatting metric name on tensorboard. User can call
        # `self.formatted_metric_name` to format the metric names to group them in the same tab.
        # Note it is not always useful to set prefix. E.g. when evaluating accuracy@1 with multiple
        # evalers, not setting prefix will show the accuracies on the same plot for comparison
        # across evalers.
        prefix: Optional[str] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(cfg, parent=parent)
        self._model = model
        self._model_param_partition_specs = model_param_partition_specs
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        if mesh.empty:
            raise RuntimeError("MetricCalculator should be created within the context of a mesh")

    def init_state(self, *, prng_key: Tensor, model_params: NestedTensor) -> NestedTensor:
        """Initializes the state.

        Will be called at the beginning of an evaluation step.

        Args:
            prng_key: The random key.
            model_params: The model parameters, containing Tensors.

        Returns:
            The initial state, to be passed to `forward`.
        """
        raise NotImplementedError(type(self))

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Handles an input batch.

        Will be called repeatedly during an evaluation step, once per evaluation input batch.

        All tensors in args and return values will/should be Tensors (especially for
        per-example logits or beam search outputs) or host arrays (only if fully replicated, such
        as scalars).

        Consumers can use utils.replicate_to_local_data(x) to convert Tensors to
        host arrays, if necessary.

        Args:
            input_batch: The evaluation input batch.
            model_params: The model parameters.
            state: As returned by `init_state` or by the previous invocation of `forward`.

        Returns:
            A dict containing:
            - "output": The per-batch outputs.
            - "state": The updated state.
        """
        raise NotImplementedError(type(self))

    def get_summaries(
        self,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
        all_forward_outputs: list[NestedTensor],
    ) -> dict[str, WeightedScalar]:
        """Computes summaries.

        Will be called at the end of an evaluation step.

        Args:
            model_params: The model parameters.
            state: As returned by the last invocation of `forward`.
            all_forward_outputs: A list of forward(...)["output"].

        Returns:
            A dict of summaries.
        """
        raise NotImplementedError(type(self))

    def _pjit(self, fn: Callable) -> Callable:
        """Compiles `fn` to run on the device mesh.

        _pjit can be used for functions with a commonly used signature and partitioning scheme.
        Subclasses can also call pjit() directly to customize signature or partition specs.

        Args:
            fn: A function that takes (model_params, replicated_inputs, per_example_inputs) as args,
                where:
                - `model_params` will be partitioned according to self._model_param_partition_specs;
                - `replicated_inputs` (e.g., prng_key) will be replicated across devices;
                - `per_example_inputs` contains tensors with a leading dimension of `batch_size`
                  and will be partitioned with input_partition_spec.
                `fn` should return a Dict containing keys "replicated" and "per_example", where:
                - `replicated` contains Tensors that have been replicated;
                - `per_example` contains Tensors with a leading dimension of `batch_size`
                  and will be partitioned with input_partition_spec.

        Returns:
            A function with the same signature as `fn`.
        """
        return pjit(
            fn,
            in_shardings=(
                self._model_param_partition_specs,  # model_params.
                None,  # replicated_inputs (e.g., prng_key).
                self._input_partition_spec(),  # per_example_inputs.
            ),
            out_shardings=dict(
                replicated=None,
                per_example=self._input_partition_spec(),
            ),
        )

    def _eval_cast(self, in_tree: NestedTensor) -> NestedTensor:
        # Cast valid input values if an eval dtype is specified.
        return utils.cast_floats(in_tree, to_dtype=self.config.eval_dtype)

    def _call_model(
        self,
        *,
        method: str,
        prng_key: Tensor,
        model_params: NestedTensor,
        input_batch: NestedTensor,
        **kwargs,
    ) -> tuple[NestedTensor, OutputCollection]:
        """Computes self._model.method(input_batch).

        Should be called inside pjit().

        Args:
            method: The model method.
            prng_key: The random key.
            model_params: The model parameters, containing Tensors.
            input_batch: The evaluation input batch, containing Tensors.
            **kwargs: kwargs to be forwarded to the model method.

        Returns
            (outputs, output_collection), where `outputs` are the return value of
            self._model.method(...).
        """
        # Shard and (possibly) dispatch the input batch.
        input_batch = self._dispatch_global_batch(input_batch)
        model_inputs = dict(
            input_batch=self._eval_cast(input_batch),
            **kwargs,
        )
        # Run model forward pass to compute outputs.
        return F(
            self._model,
            method=method,
            prng_key=prng_key,
            state=self._eval_cast(model_params),
            inputs=model_inputs,
            is_training=False,
        )

    def _input_partition_spec(self) -> PartitionSpec:
        module = self.parent
        while module is not None and not isinstance(module, SpmdEvaler):
            module = module.parent
        if module is not None and hasattr(module.input, "partition_spec"):
            return module.input.partition_spec
        return utils.input_partition_spec()

    def _dispatch_global_batch(self, input_batch: NestedTensor) -> NestedTensor:
        module = self.parent
        while module is not None and not isinstance(module, SpmdEvaler):
            module = module.parent
        if module is not None and hasattr(module.input, "dispatch_global_batch"):
            input_batch = module.input.dispatch_global_batch(input_batch)
        return input_batch

    def formatted_metric_name(self, metric_name):
        """Prepend the prefix to the metric_name."""
        if self.config.prefix is not None:
            return f"{self.config.prefix}/{metric_name}"
        else:
            return metric_name


class ModelSummaryAccumulator(BaseMetricCalculator):
    """Accumulates model summaries over evaluation batches.

    Currently only accumulates WeightedScalar summaries.
    """

    @config_class
    class Config(BaseMetricCalculator.Config):
        """Configures ModelSummaryAccumulator."""

        # Model method to call.
        model_method: str = "forward"
        # kwargs passed to model_method along with input batch.
        model_method_kwargs: dict[str, Any] = {}
        # The metric accumulator.
        metric_accumulator: InstantiableConfig = MetricAccumulator.default_config()

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(
            cfg, parent=parent, model=model, model_param_partition_specs=model_param_partition_specs
        )
        self._metric_accumulator = None
        self._jit_forward = self._pjit(self._forward_in_pjit)

    def init_state(self, *, prng_key: Tensor, model_params: NestedTensor) -> NestedTensor:
        cfg = self.config
        self._metric_accumulator = cfg.metric_accumulator.instantiate()
        return dict(prng_key=prng_key)

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        outputs = self._jit_forward(model_params, state["prng_key"], input_batch)
        self._process_summaries(outputs["replicated"]["summaries"])
        return dict(
            state=dict(prng_key=outputs["replicated"]["prng_key"]), output=outputs["per_example"]
        )

    def _forward_in_pjit(
        self,
        model_params: NestedTensor,
        prng_key: Tensor,
        input_batch: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Calls `self._model` and returns summaries."""
        cfg = self.config
        next_key, forward_prng = jax.random.split(prng_key)
        model_outputs, model_output_collection = self._call_model(
            method=cfg.model_method,
            prng_key=forward_prng,
            model_params=model_params,
            input_batch=input_batch,
            **cfg.model_method_kwargs,
        )
        return dict(
            replicated=dict(
                prng_key=next_key,
                summaries=model_output_collection.summaries,
            ),
            per_example=self._per_example_outputs(model_outputs),
        )

    # pylint: disable-next=no-self-use
    def _per_example_outputs(self, model_outputs: NestedTensor) -> NestedTensor:
        # Outputs nothing by default. Subclass can override this class to output something.
        del model_outputs
        return {}

    def _process_summaries(self, summaries: dict[str, Any]):
        self._metric_accumulator.update(summaries)

    def get_summaries(
        self,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
        all_forward_outputs: list[NestedTensor],
    ) -> dict[str, WeightedScalar]:
        return self._metric_accumulator.summaries()


class CompositeMetricCalculator(BaseMetricCalculator):
    """Runs multiple metric calculators over evaluation batches.

    This calculator supports propagating outputs from certain (sub)calculators to others via
    `dependencies`. Propagated outputs appear as new keys in the input batch to the receiving
    calculator. It is up to the caller to ensure that calculators receiving augmented inputs
    actually read the new keys.
    """

    class Dependency(flax_struct.PyTreeNode):
        # Source calculator name.
        src: str
        # Destination calculator name.
        dst: str
        # Destination input batch key. If None, defaults to `src`.
        dst_key: Optional[str] = None

    @config_class
    class Config(BaseMetricCalculator.Config):
        """Configures CompositeMetricCalculator."""

        # A mapping from unique names to metric calculators. Names must be valid module names.
        # If `dependencies` is left unspecified, calculators will be invoked in the given order.
        metric_calculators: Required[Mapping[str, BaseMetricCalculator.Config]] = REQUIRED
        # Optionally specify outputs to be routed from one calculator to another.
        # Dependencies are specified as (src_calculator, dst_calculator, dst_key), indicating that
        # the outputs from `src_calculator` will be provided as inputs to `dst_calculator` as
        # `input_batch[dst_key]`.
        # The routes must form a DAG, and calculators will be invoked in the topologically sorted
        # order. `dst_calculator` can be a regex, to be full-matched against
        # `cfg.metric_calculators`. To avoid confusion about which `src` produces which `dst_key`,
        # each `src` is required to produce a disjoint set of `dst_key`s.
        dependencies: Optional[Sequence["CompositeMetricCalculator.Dependency"]] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(
            cfg, parent=parent, model=model, model_param_partition_specs=model_param_partition_specs
        )
        # Maps dst to (one or more) src calculators, forming a DAG.
        self._calculator_dag: dict[str, set[str]] = defaultdict(set)
        # Each edge (src, dst) corresponds to a dst_key.
        self._edge_names: dict[tuple[str, str], str] = {}
        # Maps dst_key to src.
        dst_key_src: dict[str, str] = {}

        # Given `dependencies` in the form of (src, dst), build the DAG.
        for src, dst, dst_key in self._dependencies():
            if src in self._calculator_dag[dst]:
                raise ValueError(f"Encountered duplicate edge ({src}, {dst}).")
            self._calculator_dag[dst].add(src)
            self._edge_names[(src, dst)] = dst_key

            # Make sure we don't have duplicate keys across different src.
            if dst_key_src.get(dst_key, src) != src:
                raise ValueError(f"Both {dst_key_src[dst_key]} and {src} produce key {dst_key}.")
            dst_key_src[dst_key] = src

        # Calculators not in `dependencies` appear as nodes with no dependencies.
        for name in cfg.metric_calculators:
            if name not in self._calculator_dag:
                self._calculator_dag[name] = set()

        # Instantiate calculators in topologically sorted order.
        # Raises graphlib.CycleError if a cycle is detected.
        self._calculators: dict[str, BaseMetricCalculator] = {}
        for name in graphlib.TopologicalSorter(self._calculator_dag).static_order():
            if name not in cfg.metric_calculators:
                raise ValueError(f"Encountered unknown calculator name {name}.")
            self._calculators[name] = self._add_child(
                name,
                cfg.metric_calculators[name],
                model=model,
                model_param_partition_specs=model_param_partition_specs,
            )
        assert len(self._calculators) == len(cfg.metric_calculators)

    def _dependencies(self):
        """Expands regex patterns from `cfg.dependencies` and yields concrete tuples of
        (src_calculator_name, dst_calculator_name, dst_key).
        """
        cfg: CompositeMetricCalculator.Config = self.config

        @functools.cache
        def resolve_name(pattern: str) -> Sequence[str]:
            matches = []
            for name in cfg.metric_calculators:
                if re.fullmatch(pattern, name):
                    matches.append(name)
            return matches

        for dep in cfg.dependencies or []:
            yield from ((dep.src, dst, dep.dst_key or dep.src) for dst in resolve_name(dep.dst))

    def init_state(self, *, prng_key: Tensor, model_params: NestedTensor) -> NestedTensor:
        states = {}
        for name, calculator in self._calculators.items():
            states[name] = calculator.init_state(prng_key=prng_key, model_params=model_params)
        return states

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        composite_outputs = dict(output={}, state={})

        for name, calculator in self._calculators.items():
            # Augment the current calculator's input batch by retrieving outputs of its source
            # calculator(s). Since self._calculators is topologically sorted, the outputs should
            # have already been computed.
            input_batch_i = {**input_batch}
            for src in self._calculator_dag[name]:
                assert (src, name) in self._edge_names, "Each edge must have an associated key."
                assert src in composite_outputs["output"], "Source calculator must have run before."
                key = self._edge_names[(src, name)]
                if key in input_batch_i:
                    raise ValueError(f"Input batch for calculator {name} already has key {key}.")
                input_batch_i[key] = composite_outputs["output"][src]

            forward_outputs = calculator.forward(
                input_batch_i,
                model_params=model_params,
                state=state[name],
            )
            composite_outputs["output"][name] = forward_outputs["output"]
            composite_outputs["state"][name] = forward_outputs["state"]

        return composite_outputs

    def get_summaries(
        self,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
        all_forward_outputs: list[NestedTensor],
    ) -> dict[str, WeightedScalar]:
        all_forward_outputs_grouped_by_name: dict[str, list[NestedTensor]] = defaultdict(list)
        for d in all_forward_outputs:
            for name in self._calculators:
                all_forward_outputs_grouped_by_name[name].append(d[name])

        composite_summaries = {}
        for calculator_name, calculator in self._calculators.items():
            summaries = calculator.get_summaries(
                model_params=model_params,
                state=state[calculator_name],
                all_forward_outputs=all_forward_outputs_grouped_by_name[calculator_name],
            )
            for summary_name, summary in summaries.items():
                composite_summaries[f"{calculator_name}/{summary_name}"] = summary

        return composite_summaries


class EvalPolicy(Protocol):
    """Decides whether evaler should run eval at the given step."""

    def __call__(self, *, step: int, train_summaries: dict[str, Any]) -> bool:
        """Implements the policy.

        Args:
            step: Current step.
            train_summaries: A collection of summaries from the most recent train step. Can be an
                empty dict, e.g. if no summaries exist.

        Returns:
            True iff we should eval at the current step.
        """
        raise NotImplementedError(type(self))


def every_n_steps_policy(
    n: int = 1, *, min_step: int = 1, max_step: Optional[int] = None
) -> EvalPolicy:
    """Evals every n steps, but not before `min_step`."""

    if max_step is not None and max_step < min_step:
        raise ValueError(f"max_step {max_step} cannot be smaller than min_step {min_step}.")

    def fn(*, step: int, train_summaries: dict[str, Any]) -> bool:
        del train_summaries
        if step < min_step:
            logging.log_first_n(
                logging.INFO, "Skipping eval, as step (%s) < min_step (%s).", 10, step, min_step
            )
            return False
        return step % n == 0 or (max_step is not None and step >= max_step)

    return fn


class SpmdEvaler(Module):
    """An evaler implementation that supports partitioning of computation and data with GSPMD."""

    @config_class
    class Config(Module.Config):
        """Configures SpmdEvaler."""

        # The input source.
        input: Required[input_base.Input.Config] = REQUIRED
        # A summary writer to log tagged summary values.
        summary_writer: InstantiableConfig = summary_writer.SummaryWriter.default_config()
        # Run this evaler according to this policy.
        eval_policy: InstantiableConfig = config_for_function(every_n_steps_policy)
        # Which evaluation iters to trace with the profiler each time the evaler is run.
        # Each trace will cover one full evaluation batch.
        # Traces will run for at most 3 unique steps.
        trace_at_iters: Sequence[int] = []
        # Cast float inputs and parameters to this dtype for the evaluation step.
        # If None, do not cast.
        eval_dtype: Optional[jnp.dtype] = None
        # The evaler metric_calculator to compute summaries.
        metric_calculator: BaseMetricCalculator.Config = ModelSummaryAccumulator.default_config()
        # If not None, writes input batches and `metric_calculator` forward outputs.
        output_writer: Optional[BaseOutputWriter.Config] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        cfg.summary_writer.dir = cfg.summary_writer.dir or os.path.join(
            cfg.dir, "summaries", "eval"
        )

        if cfg.eval_dtype is not None:
            utils.validate_float_dtype(cfg.eval_dtype)

        self.input: input_base.Input = self._add_child(  # pytype: disable=annotation-type-mismatch
            "input", maybe_set_config(cfg.input, is_training=False)
        )
        self._add_child(
            "metric_calculator",
            cfg.metric_calculator.set(eval_dtype=cfg.eval_dtype),
            model=model,
            model_param_partition_specs=model_param_partition_specs,
        )
        self._add_child("summary_writer", cfg.summary_writer)
        if cfg.output_writer is not None:
            self._add_child("output_writer", cfg.output_writer)

        self._trace_steps = set()
        self._eval_policy: EvalPolicy = cfg.eval_policy.instantiate()

    def eval_step(
        self,
        step: int,
        *,
        prng_key: Tensor,
        model_params: NestedTensor,
        return_aux: bool = False,
        train_summaries: Optional[NestedTensor] = None,
        force_run: bool = False,
    ) -> tuple[Tensor, Optional[dict[str, Any]], Optional[list[NestedTensor]]]:
        """Runs eval for the given step.

        Args:
            step: Current step.
            prng_key: PRNG key.
            model_params: Model parameters.
            return_aux: Boolean to determine whether outputs are returned.
            train_summaries: Summaries from the most recent training step. Can be used in the
                `evaler_policy`.
            force_run: If True, force run the eval for the given step.

        Returns:
            A tuple (prng_key, summaries, outputs), where
            prng_key can be used for a future training step,
            summaries contains replicated eval summaries, or None if eval did not run this step,
            and outputs contains an optional list of evaler outputs for all of the input.

        Raises:
            RuntimeError: If attempting to nest profilers.
        """
        cfg = self.config

        if not force_run and not self._eval_policy(
            step=step, train_summaries=(train_summaries or {})
        ):
            return prng_key, None, None

        self.vlog(
            2,
            "%s: Process % 3d step % 8d: starting",
            self.path(),
            jax.process_index(),
            step,
        )

        start_time = time.perf_counter()
        metric_calculator_state = self.metric_calculator.init_state(
            prng_key=prng_key, model_params=model_params
        )
        all_metric_calculator_outputs = []

        forward_outputs = None
        stop_trace_iter = None
        eval_input_iter = iter(self.input.dataset())
        for batch_ix, input_batch in enumerate(self.input.batches(eval_input_iter)):
            logging.log_first_n(logging.INFO, "Evaler input_batch=%s", 3, utils.shapes(input_batch))

            if batch_ix == stop_trace_iter:
                assert (
                    forward_outputs is not None
                ), "output was None at the end of a trace, not expected."
                jax.tree.map(lambda x: x.block_until_ready(), forward_outputs)
                jax.profiler.stop_trace()
                self.vlog(2, "Stopped profiler tracing for evaler %s.", cfg.name)
                stop_trace_iter = None
                self._trace_steps.add(step)

            if batch_ix in cfg.trace_at_iters and len(self._trace_steps) <= 3:
                try:
                    jax.profiler.start_trace(self.summary_writer.config.dir)
                except RuntimeError as e:
                    if "Only one profile may be run at a time." in str(e):
                        # https://github.com/google/jax/blob/260f1d8b/jax/_src/profiler.py#L110-L111
                        # No functionality is currently exposed to check this robustly.
                        raise RuntimeError(
                            "Nesting evaler profiling within a higher "
                            "level profile session is not currently supported. "
                        ) from e
                    # Else profiler is already running.
                finally:
                    stop_trace_iter = batch_ix + 1  # We only look at one batch.
                self.vlog(2, "Start profiling evaler %s", cfg.name)

            with jax.profiler.StepTraceAnnotation(cfg.name, step_num=step):
                with jax.profiler.TraceAnnotation(f"{cfg.name}.forward"):
                    global_input_batch = utils.host_to_global_device_array(
                        input_batch,
                        partition=self.input.partition_spec,
                    )
                    forward_outputs = self.metric_calculator.forward(
                        global_input_batch,
                        model_params=model_params,
                        state=metric_calculator_state,
                    )
                metric_calculator_state = forward_outputs["state"]
                all_metric_calculator_outputs.append(forward_outputs["output"])
                if "output_writer" in self.children:
                    self.output_writer.write(
                        input_batch=global_input_batch, output_batch=forward_outputs["output"]
                    )

            self.vlog(
                3,
                "%s: Process % 3d step % 8d batch % 8d done",
                self.path(),
                jax.process_index(),
                step,
                batch_ix,
            )

        summaries = self.metric_calculator.get_summaries(
            model_params=model_params,
            state=metric_calculator_state,
            all_forward_outputs=all_metric_calculator_outputs,
        )
        self.vlog(
            1,
            "%s: Process % 3d step % 8d: metrics=%s",
            self.path(),
            jax.process_index(),
            step,
            utils.flatten_items(summaries),
        )
        elapsed = time.perf_counter() - start_time
        self.summary_writer(step, {"eval_time_secs": elapsed, **summaries})

        outputs = all_metric_calculator_outputs if return_aux else None
        return prng_key, summaries, outputs


class PredictionOutputs(NamedTuple):
    input_batch: NestedTensor
    predict_outputs: NestedTensor


class GlobalMetricCalculator(BaseMetricCalculator):
    """A metric calculator for tasks require evaluation on entire datasets.

    This calculator fits for the scenario where model predictions for the whole evaluation set need
    to be collected before computing metrics.
    """

    @config_class
    class Config(BaseMetricCalculator.Config):
        # Method used for getting model's prediction.
        predict_method: str = "predict"

        # Field inside input batch used during prediction. '/' is used for nesting.
        predict_input_field: Optional[str] = None

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
        use_jit_for_metric_calculation: bool = True,
    ):
        super().__init__(
            cfg, parent=parent, model=model, model_param_partition_specs=model_param_partition_specs
        )
        self._use_jit_for_metric_calculation = use_jit_for_metric_calculation
        self._jit_predict = self._pjit(self._predict_in_pjit)
        if self._use_jit_for_metric_calculation:
            self._jit_compute_metrics = self._pjit(self._compute_metrics_in_pjit)
        self._metric_accumulator: MetricAccumulator = None

    def init_state(  # pylint: disable=duplicate-code
        self, *, prng_key: Tensor, model_params: NestedTensor
    ) -> NestedTensor:
        self._metric_accumulator = MetricAccumulator.default_config().instantiate()
        return dict(prng_key=prng_key)

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Calls predict method of the model and returns input_batch and per-batch model outputs.

        Will be called repeatedly during an evaluation step, once per evaluation input batch.

        All tensors in args and return values will/should be Tensors (especially for
        per-example logits or beam search outputs) or host arrays (only if fully replicated, such
        as scalars).

        Args:
            input_batch: The evaluation input batch.
            model_params: The model parameters.
            state: As returned by `init_state` or by the previous invocation of `forward`.

        Returns:
            A dict containing:
            - "state": A dict containing prng_key.
            - "output": A dict containing input_batch and per-batch model outputs.
        """
        outputs = self._jit_predict(model_params, state["prng_key"], input_batch)
        predict_outputs = outputs["per_example"]
        summaries = outputs["replicated"]["summaries"]
        self._metric_accumulator.update(summaries)
        return dict(
            state=dict(prng_key=outputs["replicated"]["prng_key"]),
            output=PredictionOutputs(input_batch, predict_outputs),
        )

    def _predict_in_pjit(
        self,
        model_params: NestedTensor,
        prng_key: Tensor,
        input_batch: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Core function that calls model's predict() method for each batch and will be pjit-ed."""
        predict_key, next_key = jax.random.split(prng_key)
        cfg = self.config
        if cfg.predict_input_field:
            input_batch = utils.get_recursively(input_batch, cfg.predict_input_field)
        model_outputs, model_output_collection = self._call_model(
            method=cfg.predict_method,
            model_params=model_params,
            prng_key=predict_key,
            input_batch=input_batch,
        )
        return dict(
            replicated=dict(prng_key=next_key, summaries=model_output_collection.summaries),
            per_example=model_outputs,
        )

    def _calculate_metrics(self, outputs: PredictionOutputs) -> dict[str, Tensor]:
        """Calculates metrics from ``concatenated_outputs`` of the whole evaluation set.

        Args:
            outputs: A PredictionOutputs with input field name as key and a tensor of shape
                [num_examples, ...] representing the concatenated input across the whole evaluation
                set for metrics calculation.

        Returns:
            A dict containing all metrics for current task.
        """
        raise NotImplementedError(type(self))

    def _compute_metrics_in_pjit(
        self,
        model_params: NestedTensor,
        prng_key: Tensor,
        outputs: list[PredictionOutputs],
    ) -> dict[str, NestedTensor]:
        """Computes metrics and returns them in "replicated"."""
        del model_params, prng_key

        # WARNING: Directly concatenating leads to incorrect XLA sharding.
        # A nested tree where each leaf tensor has shape [num_eval_batches, batch_size, ...]
        stacked_outputs = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *outputs)
        # Each concatenated leaf tensor has shape [num_eval_batches * batch_size, ...] when all
        # other dimensions of the stacked output are positive. Otherwise, that concatenated leaf
        # node will be set as None.
        concatenated_outputs = jax.tree.map(
            lambda xs: (
                with_sharding_constraint(
                    jnp.reshape(xs, (-1, *xs.shape[2:])), input_partition_spec()
                )
                if all(dim > 0 for dim in xs.shape[2:])
                else None
            ),
            stacked_outputs,
        )

        return dict(
            replicated=self._calculate_metrics(concatenated_outputs),
            per_example={},
        )

    def get_summaries(
        self,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
        all_forward_outputs: list[PredictionOutputs],
    ) -> dict[str, Union[WeightedScalar, Tensor]]:
        if self._use_jit_for_metric_calculation:
            metrics = self._jit_compute_metrics(
                model_params,
                state["prng_key"],
                all_forward_outputs,
            )["replicated"]
            return metrics

        outputs = replicate_to_local_data(all_forward_outputs)
        concatenated_outputs = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)
        return self._calculate_metrics(concatenated_outputs)
