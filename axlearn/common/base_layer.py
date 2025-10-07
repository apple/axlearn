# Copyright © 2023 Apple Inc.

"""Base layer infrastructure for neural network implementations in AXLearn.

This module provides the foundational classes for building neural network layers with
support for distributed training, memory optimization, and debugging utilities.

Design Philosophy:

- Declarative Configuration: Parameters and behaviors are specified through configs
  rather than imperative code, enabling better composition and experimentation.
- Hierarchical Defaults: Settings cascade down the layer tree, allowing model-wide
  defaults with local overrides where needed.
- Separation of Concerns: Layers declare what they need (parameters, monitoring points),
  while the framework handles how (initialization, sharding, statistics computation).

Key Components:

1. Parameter Specification Classes:
   - `ParameterSpec`: Comprehensive parameter metadata including shape, dtype, initialization,
     sharding, and optimizer-specific configurations (factorization, weight decay).
   - `FactorizationSpec`: Configuration for memory-efficient gradient factorization used by
     AdaFactor optimizer to reduce memory footprint from O(n×m) to O(n+m).

2. BaseLayer:
   The foundation class for all neural network layers, providing:
   - Declarative parameter management with automatic initialization
   - Hierarchical configuration inheritance (dtype, initialization cascade from parent)
   - Built-in support for model/data parallelism via parameter sharding
   - Memory optimization through rematerialization (checkpoint/remat)
   - Debugging utilities via TensorStats integration

3. Memory Optimization:
   - `RematSpec`: Configuration for selective recomputation during backprop to trade
     compute for memory, with customizable policies for fine-grained control.

4. Debugging & Monitoring:
   - `TensorStats`: Abstract interface for computing statistics on tensors
   - `TensorRMSNorm`: Computes root mean square norm for gradient monitoring
   - `TensorMaxAbs`: Tracks maximum absolute values for overflow detection
   - `CompositeTensorStats`: Combines multiple statistics for comprehensive monitoring
   - `DefaultTensorStats`: Pre-configured combination of common statistics

5. Training Utilities:
   - `ParameterNoise`: Interface for parameter perturbation during training

Usage Examples:
    ```python
    # Pattern 1: Direct parameter creation
    class Linear(BaseLayer):
        def _create_layer_parameter_specs(self):
            return {
                "weight": ParameterSpec(
                    shape=[self.config.input_dim, self.config.output_dim],
                    mesh_axes=("data", "model"),
                )
            }

        def forward(self, x):
            return jnp.dot(x, self.parameters["weight"])

    # Pattern 2: Composing child layers (more common)
    class FFN(BaseLayer):
        @config_class
        class Config(BaseLayer.Config):
            input_dim: Required[int] = REQUIRED
            hidden_dim: Required[int] = REQUIRED
            output_dim: Required[int] = REQUIRED
            linear1: BaseLayer.Config = Linear.default_config()
            linear2: BaseLayer.Config = Linear.default_config()
            activation: BaseLayer.Config = ReLU.default_config()

        def __init__(self, cfg: Config, *, parent: Optional[Module]):
            super().__init__(cfg, parent=parent)
            cfg = self.config
            # Add child layers using _add_child
            self._add_child("linear1", cfg.linear1.set(
                input_dim=cfg.input_dim, output_dim=cfg.hidden_dim
            ))
            self._add_child("activation", cfg.activation)
            self._add_child("linear2", cfg.linear2.set(
                input_dim=cfg.hidden_dim, output_dim=cfg.output_dim
            ))

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
    ```

See individual class docstrings for detailed usage and examples.
"""

import dataclasses
import functools
import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import jax
import jax.ad_checkpoint
from absl import logging
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.config import ConfigOr, Configurable, config_class, maybe_instantiate
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, child_context, current_context, new_output_collection
from axlearn.common.param_init import DefaultInitializer, FanAxes
from axlearn.common.traceback_util import no_stack_summary
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    PartitionSpec,
    Tensor,
    TensorSpec,
    check_jax_type,
    flatten_items,
    get_or_none,
)


@dataclasses.dataclass
class FactorizationSpec:
    """A FactorizationSpec describes how to factorize a parameter's gradient.

    Used by AdaFactor optimizer for memory-efficient second-moment estimation by factorizing
    the second-moment matrix into row and column statistics instead of storing the full matrix.

    Attributes:
        axes: A list of None/str corresponding to the axes of the parameter shape.
            Each element is either:
            - None: no factorization along this axis.
            - str: the factorization axis name (typically "row" or "col").

            For AdaFactor, either:
            - All axes are None (no factorization, used for small parameters)
            - Exactly two axes are "row" and "col" (factorized, used for large matrices)

            Example:
                For a weight matrix of shape [1024, 4096]:
                axes=["row", "col"] enables factorization

                For a 3D tensor of shape [8, 1024, 4096]:
                axes=[None, "row", "col"] factorizes only the last two dimensions
    """

    axes: Sequence[Optional[str]]


# Ideally this would be a recursive type:
# Dict[str, Union[FactorizationSpec, "NestedFactorizationSpec"]]
# but we use Any to avoid issues with recursive type definitions in Python's type system.
NestedFactorizationSpec = dict[str, Union[FactorizationSpec, Any]]


@dataclasses.dataclass
class ParameterSpec(TensorSpec):
    """Specification of a layer parameter.

    This is a dataclass so that jax.tree.map does not expand it, treating it as a leaf node
    in pytrees. This ensures that ParameterSpec instances are preserved as whole objects
    during tree transformations rather than being decomposed into their fields.

    Inherits from TensorSpec:
        shape: The shape of the parameter tensor.
        dtype: The data type of the param. If None, uses the layer's default dtype.
        mesh_axes: Partitioning specification for the parameter.
            - If None, the parameter will not be partitioned and will be replicated.
            - If a sequence, it should have exactly len(shape) elements.
            - mesh_axes[i] describes partitioning for shape[i], where each value can be:
              None(do not partition along this axis), 'model', 'data', 'fsdp', 'track', or etc.
        memory_kind: Optional memory location ('device' or 'pinned_host').

    Attributes:
        initializer: The initializer of the param. If None, uses the layer's default initializer.
        factorization: Factorization spec for the parameter. Used by AdaFactor optimizer to
            determine which dimensions can be factorized for memory-efficient second-moment
            estimation.
        fan_axes: Fan axes information for the parameter. Used to compute fan-in/fan-out
            for initialization schemes like Xavier/He initialization.
        weight_decay_scale: Per-parameter weight decay / l2 regularization scale.
            The effective weight decay rate will be:
                global_weight_decay_rate * param_weight_decay_scale

            Ways to configure:
            - Layer implementation can override _create_layer_parameter_specs() or
              create_parameter_specs_recursively() to set weight_decay_scale.
            - Set to 0 to disable weight decay on specific parameters.
            - Users can also use `per_param_scale` arg of `add_decayed_weights` or
              `weight_decay_per_param_scale` arg of optimizers with custom logic.

            Note: ParameterSpec.weight_decay_scale takes precedence over `per_param_scale`.
    """

    initializer: Optional[param_init.Initializer] = None
    factorization: Optional[FactorizationSpec] = None
    fan_axes: Optional[FanAxes] = None
    weight_decay_scale: Optional[float] = None

    def fans(self) -> dict[str, float]:
        """Returns a dictionary with keys 'fan_in', 'fan_out', and 'fan_avg' containing
        the fan values for this parameter.

        The calculation is consistent with jax's initializers: Indices without
        an explicit axis type specified are treated as both in and out axes.
        Batch axes are ignored.
        """
        sizes = {}
        for axis_type in self.fan_axes._fields:  # pylint: disable=protected-access
            axes = getattr(self.fan_axes, axis_type)
            if isinstance(axes, int):
                axes = [axes]
            sizes[axis_type] = math.prod(self.shape[axis] for axis in axes)
        unbatched_size = math.prod(self.shape) / sizes["batch_axis"]
        result = dict(
            fan_in=unbatched_size / sizes["out_axis"], fan_out=unbatched_size / sizes["in_axis"]
        )
        result["fan_avg"] = (result["fan_in"] + result["fan_out"]) / 2
        return result


# Legacy type alias. For new code, use Nested[ParameterSpec] from axlearn.common.utils.
NestedParameterSpec = Optional[Union[ParameterSpec, dict[str, Any]]]


@dataclasses.dataclass
class RematSpec:
    """Configuration for rematerialization (remat) / checkpointing of layer computations.

    Rematerialization (also called checkpointing) is a memory-saving technique where
    intermediate values are recomputed during the backward pass instead of being saved
    during the forward pass. This trades computation for memory.

    In JAX terminology, "remat" and "checkpoint" are interchangeable names for the same
    feature - jax.remat() and jax.checkpoint() are aliases. The term "remat" is short
    for "rematerialization", which refers to recomputing (rematerializing) values when
    needed rather than storing them.

    Reference:
        https://github.com/google/jax/blob/1b79caa/jax/_src/ad_checkpoint.py#L99
        https://docs.jax.dev/en/latest/jep/11830-new-remat-checkpoint.html

    Attributes:
        prevent_cse: If True, prevents common subexpression elimination optimizations
            that might defeat the purpose of rematerialization.
        policy: Optional rematerialization policy that controls which values to save
            vs. recompute. Can be a custom policy or one from jax.checkpoint_policies
            (e.g., checkpoint_dots to save only matrix multiplication results).
    """

    prevent_cse: bool = True
    policy: Optional[ConfigOr[Callable[..., bool]]] = None


class ParameterNoise(Configurable):
    """An interface for applying parameter noise."""

    def apply(self, prng_key: Tensor, params: NestedTensor) -> NestedTensor:
        """To be implemented by subclasses."""
        raise NotImplementedError(self)


class TensorStats(Module):
    """An abstract Module to add summaries about the given Tensors.

    TensorStats provides a flexible way to compute and log statistics (e.g., RMS norm, max abs)
    about tensors at various points in a model. This is useful for debugging, monitoring training
    stability, and understanding model behavior.

    Usage pattern:
    1. Configure a TensorStats implementation in BaseLayer.Config.tensor_stats
    2. The layer calls self._add_tensor_stats(name, tensor) at points of interest
    3. The TensorStats implementation computes and logs the requested statistics

    Example:
        # In layer configuration:
        layer_cfg.tensor_stats = DefaultTensorStats.default_config()

        # In layer forward method:
        self._add_tensor_stats("attention_output", attn_output)

        # This logs stats like "layer_name/tensor_stats/attention_output/rms_norm"

    Common implementations:
        - TensorRMSNorm: Computes RMS norm of tensors
        - TensorMaxAbs: Computes maximum absolute value
        - DefaultTensorStats: Combines RMS norm and max abs
        - CompositeTensorStats: Combines multiple TensorStats implementations
    """

    def add_stats(self, name: str, value: Nested[Tensor]):
        """Computes and adds summaries for the given tensor.

        Args:
            name: Name identifier for the tensor (e.g., "attention_output", "linear1_outputs").
            value: The tensor or nested structure of tensors to compute stats for.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(type(self))


class CompositeTensorStats(TensorStats):
    """A TensorStats that combines multiple child TensorStats implementations.

    This class implements the composition pattern, allowing you to compute multiple
    statistics on the same tensors without modifying existing implementations.
    It's useful when you want different views of your tensors for debugging or monitoring.

    Example usage:
        # Compute both RMS norm and max absolute value:
        tensor_stats = CompositeTensorStats.default_config().set(
            tensor_stats={
                "norm": TensorRMSNorm.default_config(),
                "max": TensorMaxAbs.default_config(),
            }
        )

        # This produces summaries like:
        # - With inline_child_summaries=False (default):
        #   "layer/tensor_stats/attention/norm/rms_norm": 0.5
        #   "layer/tensor_stats/attention/max/max_abs": 1.2
        # - With inline_child_summaries=True:
        #   "layer/tensor_stats/attention/rms_norm": 0.5
        #   "layer/tensor_stats/attention/max_abs": 1.2

    Common use cases:
        - Debugging gradient explosions: Combine max_abs with percentile stats
        - Monitoring training stability: Combine RMS norm with mean/variance
        - Custom analysis: Mix any TensorStats implementations you need

    See DefaultTensorStats for a pre-configured version with common statistics.
    """

    @config_class
    class Config(TensorStats.Config):
        # Dictionary of child TensorStats configs to apply. Keys become prefixes in summary paths
        # unless inline_child_summaries is True.
        tensor_stats: dict[str, TensorStats.Config] = {}
        # Whether to inline child summaries into the parent's summary.
        #  If False (default), child stats are nested under their keys:
        #      {"foo": {"rms_norm": value}}
        #  If True, child summaries are flattened into parent namespace:
        #      {"rms_norm": value}
        inline_child_summaries: bool = False

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._child_stats = {}
        for child_stats_name, child_stats_cfg in cfg.tensor_stats.items():
            self._child_stats[child_stats_name] = self._add_child(child_stats_name, child_stats_cfg)

    def add_stats(self, name: str, value: Nested[Tensor]):
        cfg = self.config
        for child_name, child_stats in self._child_stats.items():
            output_collection = new_output_collection()
            with child_context(child_name, output_collection=output_collection):
                child_stats.add_stats(name, value)
            if cfg.inline_child_summaries:
                self.get_invocation_context().output_collection.summaries.update(
                    output_collection.summaries
                )
            else:
                self.get_invocation_context().output_collection.summaries[
                    child_name
                ] = output_collection.summaries


class TensorRMSNorm(TensorStats):
    """Computes and logs the root mean square norm of tensors."""

    def add_stats(self, name: str, value: Nested[Tensor]):
        self.add_summary("rms_norm", (value**2.0).mean().astype(jnp.float32) ** 0.5)


class TensorMaxAbs(TensorStats):
    """Computes and logs the maximum absolute value of tensors."""

    def add_stats(self, name: str, value: Nested[Tensor]):
        self.add_summary("max_abs", jnp.abs(value).max().astype(jnp.float32))


class DefaultTensorStats(CompositeTensorStats):
    """Default tensor stats that compute RMS norm and max value.

    This is a pre-configured CompositeTensorStats that combines the two most
    commonly used statistics for monitoring neural networks:
    - RMS norm: Helps detect gradient explosion/vanishing
    - Max absolute value: Shows the range of tensor values

    The summaries are inlined by default, producing paths like:
        "layer/tensor_stats/attention/rms_norm": 0.5
        "layer/tensor_stats/attention/max_abs": 1.2

    This is the recommended starting point for adding tensor monitoring to layers.
    You can override the configuration if you need different statistics.
    """

    @config_class
    class Config(CompositeTensorStats.Config):
        tensor_stats: dict[str, TensorStats.Config] = {
            "norm": TensorRMSNorm.default_config(),
            "max": TensorMaxAbs.default_config(),
        }
        inline_child_summaries: bool = True


class BaseLayer(Module):
    """Base class for all neural network layers in AXLearn.

    BaseLayer extends Module to provide neural network-specific functionality including
    parameter management, initialization, sharding, and various training utilities.
    It serves as the foundation for all layer implementations (Linear, Conv, Attention, etc.).

    Key features:
        1. Parameter Management: Declarative parameter specification with shape, dtype,
           initialization, and sharding configuration through ParameterSpec.

        2. Hierarchical Configuration: Inherits dtype and initialization from parent layers,
           allowing model-wide defaults with local overrides.

        3. Distributed Training Support: Built-in parameter sharding via mesh_axes for
           model parallelism and FSDP.

        4. Memory Optimization: Optional rematerialization (remat_spec) to trade compute
           for memory during backpropagation.

        5. Debugging & Monitoring: TensorStats integration for tracking layer activations
           and gradients during training.

        6. Training Utilities: Parameter noise injection for regularization and exploration.

    Layer implementation patterns:

        Pattern 1: Direct parameters (for primitive layers like Linear, Conv):
            - Override _create_layer_parameter_specs() to declare weight/bias parameters
            - Access parameters via self.parameters["name"] in forward()

        Pattern 2: Composing child layers (more common for compound layers):
            - Define child configs in Config class
            - Use self._add_child() in __init__ to instantiate child layers
            - Call child layers directly in forward() - context management is automatic

        Both patterns can be mixed in a single layer. See module docstring for examples.

    Subclass implementation guide:
        1. Define Config class with layer-specific configuration fields
        2. For direct parameters: Override _create_layer_parameter_specs()
        3. For child layers: Override __init__ and use _add_child()
        4. Implement forward() method for the layer's computation
        5. Optionally use _add_tensor_stats() to monitor intermediate values
    """

    @config_class
    class Config(Module.Config):
        """Configuration for BaseLayer. These settings cascade down the layer hierarchy - child
        layers inherit from parents unless explicitly overridden.
        """

        # Default parameter dtype. If None, inherits from parent module.  Common values:
        # jnp.float32, jnp.bfloat16 for mixed precision training.
        dtype: Optional[jnp.dtype] = None
        # Parameter initialization configuration. If None, inherits from parent. Controls weight
        # initialization schemes (Xavier, He, etc.).
        param_init: Optional[DefaultInitializer.Config] = None
        # Sharding specification for distributed training.  Defines how parameters are partitioned
        # across the device mesh.  For layers with weight and bias, typically only weight spec is
        # provided; bias spec is derived automatically.
        param_partition_spec: NestedParameterSpec = None
        # Rematerialization (checkpointing) configuration for memory optimization.  When set, wraps
        # layer methods with jax.checkpoint to trade compute for memory.  If None, XLA handles
        # rematerialization automatically.
        remat_spec: Optional[RematSpec] = None
        # Parameter noise configuration for regularization/exploration.  Note:
        # apply_parameter_noise_recursively() must be called explicitly by trainer, not
        # automatically invoked during forward pass.  Child layer noise is applied before parent
        # layer noise.
        param_noise: Optional[ParameterNoise.Config] = None
        # Tensor statistics collection for debugging and monitoring.  When configured, enables
        # _add_tensor_stats() calls within the layer.  The layer implementation controls WHAT to
        # monitor (via _add_tensor_stats calls), while this config controls HOW to compute stats
        # (RMS norm, max, etc.).
        tensor_stats: Optional[TensorStats.Config] = None

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg: BaseLayer.Config = self.config
        if cfg.param_init is not None:
            init = cfg.param_init.instantiate()
        elif parent is None:
            init = DefaultInitializer.default_config().instantiate()
        else:
            init = None
        self._param_init = init
        if cfg.param_noise is not None:
            self._param_noise = cfg.param_noise.instantiate()
        else:
            self._param_noise = None
        if cfg.tensor_stats is not None:
            self._add_child("tensor_stats", cfg.tensor_stats)
        self._remat_methods = []  # List[str]. Used for testing.

    def _methods_to_wrap_for_auto_child_context(self) -> dict[str, Callable]:
        return {
            method: method_fn
            for method, method_fn in super()._methods_to_wrap_for_auto_child_context().items()
            if not hasattr(BaseLayer, method)
        }

    def _wrap_methods_with_auto_child_context(
        self, methods: dict[str, Callable]
    ) -> dict[str, Callable]:
        cfg: Module.Config = self.config

        if cfg.remat_spec is not None:
            for method_name, method_fn in dict(methods).items():
                methods[method_name] = self._maybe_wrap_with_remat(method_fn)

        return super()._wrap_methods_with_auto_child_context(methods)

    def _maybe_wrap_with_remat(self, method_fn: Callable) -> Callable:
        """Maybe wrap `method_fn` with jax remat.

        This is called from `BaseLayer._wrap_methods_with_auto_child_context`.

        Args:
            method_fn: A function that takes a module as its first argument.

        Returns:
            A possibly wrapped version of `method_fn`.
        """
        cfg: Module.Config = self.config
        if getattr(method_fn, "_no_remat", False):
            return method_fn

        @no_stack_summary
        @functools.wraps(method_fn)
        def maybe_call_with_remat(module: Module, *args, **kwargs):
            if current_context() is None or not module.is_training:
                return method_fn(module, *args, **kwargs)

            static_kwargs = {}
            tracer_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Tensor):
                    tracer_kwargs[k] = v
                else:
                    static_kwargs[k] = v
            module.vlog(
                3,
                "BaseLayer.maybe_call_with_remat %s.%s: static_kwargs=%s",
                module.path(),
                method_fn,
                set(static_kwargs.keys()),
            )

            # Remat always uses abstract tracers even if concrete information is available.
            # This means that all inputs and outputs to a remat function need to be JAX types.
            # We print a nice error if the inputs are not.
            check_jax_type(
                args=args,
                kwargs=tracer_kwargs,
                msg=f"Attempt to use remat on {module}.{method_fn} "
                "failed. Consider decorating with @no_remat.",
            )

            def fn(*args, **kwargs):
                """Unlike module.method, fn returns (outputs, output_collection)."""
                output_collection = new_output_collection()
                # We override output_collection to avoid leaking tracers.
                with child_context("remat", module=module, output_collection=output_collection):
                    outputs = method_fn(module, *args, **kwargs, **static_kwargs)
                return outputs, output_collection

            logging.info("Applying remat on %s.%s: %s", module.path(), method_fn, cfg.remat_spec)
            # pylint: disable-next=protected-access
            module._remat_methods.append(method_fn.__name__)
            # Pass both outputs and output_collection through remat(...) to avoid leaking tracers.
            outputs, output_collection = jax.checkpoint(
                fn,
                **{k: maybe_instantiate(v) for k, v in dataclasses.asdict(cfg.remat_spec).items()},
            )(*args, **tracer_kwargs)
            module.get_invocation_context().output_collection.update(output_collection)
            return outputs

        return maybe_call_with_remat

    def dtype(self):
        if self.config.dtype is not None:
            return self.config.dtype
        if self.parent is not None:
            return self.parent.dtype()
        return jnp.float32

    def param_init(self) -> param_init.Initializer:
        init = getattr(self, "_param_init", None)
        if init is not None:
            return init
        return self.parent.param_init()

    # pylint: disable-next=no-self-use
    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        """Subclasses can override this method to add layer parameters."""
        return {}

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs: dict[str, NestedParameterSpec] = {}
        param_specs = self._create_layer_parameter_specs()
        for name, param_spec in param_specs.items():
            partition_spec = param_spec.mesh_axes
            if partition_spec is None:
                # Replicate along all axes.
                partition_spec = [None] * len(param_spec.shape)
            if len(partition_spec) != len(param_spec.shape):
                raise ValueError(
                    f"partition_spec {partition_spec} must have the same length as "
                    f"shape {param_spec.shape})"
                )
            param_spec = dataclasses.replace(
                param_spec,
                mesh_axes=PartitionSpec(*partition_spec),
                fan_axes=(
                    param_spec.fan_axes
                    if param_spec.fan_axes is not None
                    else self._compute_fan_axes(name=name, parameter_spec=param_spec)
                ),
            )
            if param_spec.dtype is None:
                param_spec = dataclasses.replace(param_spec, dtype=self.dtype())
            specs[name] = param_spec
        for name, child in self._children.items():
            if not isinstance(child, BaseLayer):
                # `child` is not a BaseLayer and does not have parameters, e.g., it can be an
                # instance of TensorStats.
                continue
            assert name not in specs
            specs[name] = child.create_parameter_specs_recursively()
        return specs

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        """Initializes parameters with given ParameterSpecs for the prebuilt params.

        Note that the returned tree contains Tensors for initialized parameters and None for the
        prebuilt parameters. This ensures that the return value contain only JAX types and
        therefore can be wrapped inside JAX function transformations.

        Use `test_utils.initialize_parameters_with_prebuilt` in testing to get a merged tree of
        prebuilt and initialized parameters.

        Args:
            prng_key: The random key.
            prebuilt: A Nested tree with the same structure as the layer parameters, whose leaf
                nodes are ParameterSpecs if the parameters are prebuilt, None if the parameters
                should be initialized.

        Returns:
            A Nested Tree with Tensors (if initialized) and None (if prebuilt) as leaf nodes.
        """
        params = {}
        param_specs = self._create_layer_parameter_specs()
        for name, spec in param_specs.items():
            # Note: we split the key even if value is prebuilt.
            prng_key, child_key = jax.random.split(prng_key)
            value = get_or_none(prebuilt, name)
            if value is not None:
                # Note that we cannot set `params[name]` to `value`, since `value` is an instance
                # of ParameterSpec and not a JAX type.
                params[name] = None
            else:
                if spec.dtype is None:
                    spec.dtype = self.dtype()
                if spec.initializer is None:
                    spec.initializer = self.param_init()
                params[name] = self._initialize_parameter(
                    name,
                    prng_key=child_key,
                    parameter_spec=spec,
                )
        for name, child in self._children.items():
            if not isinstance(child, BaseLayer):
                # `child` is not a BaseLayer and does not have parameters, e.g., it can be an
                # instance of TensorStats.
                continue
            assert name not in params
            prng_key, child_key = jax.random.split(prng_key)
            params[name] = child.initialize_parameters_recursively(
                prng_key=child_key,
                prebuilt=get_or_none(prebuilt, name),
            )
        return params

    def _use_prebuilt_params(self, prebuilt: Optional[Nested[Optional[ParameterSpec]]]) -> bool:
        prebuilt_keys = {key for key, value in flatten_items(prebuilt) if value is not None}
        if not prebuilt_keys:
            return False
        param_keys = {key for key, _ in flatten_items(self.create_parameter_specs_recursively())}
        if prebuilt_keys != param_keys:
            raise NotImplementedError(
                f"Partial prebuilt params are not supported: {param_keys - prebuilt_keys}"
            )
        return True

    def _initialize_parameter(
        self, name: str, *, prng_key: Tensor, parameter_spec: ParameterSpec
    ) -> Tensor:
        """Adds a parameter with the given name and shape.

        Args:
            name: The parameter name.
            prng_key: The pseudo random generator key.
            parameter_spec: The parameter specification.

        Returns:
            The created parameter.

        Raises:
            ValueError: If child with name already exists.
        """
        if name in self._children:
            raise ValueError(f"Child {name} already exists.")
        param = parameter_spec.initializer.initialize(
            name,
            prng_key=prng_key,
            shape=parameter_spec.shape,
            dtype=parameter_spec.dtype,
            axes=(
                parameter_spec.fan_axes
                if parameter_spec.fan_axes is not None
                else self._compute_fan_axes(name=name, parameter_spec=parameter_spec)
            ),
        )
        return param

    def apply_parameter_noise_recursively(
        self, prng_key: Tensor, params: NestedTensor
    ) -> NestedTensor:
        """Applies parameter noise recursively on `params`.

        Args:
            prng_key: The random key.
            params: The parameters of this layer. apply_parameter_noise_recursively should not
                mutate the nested structure in place but should make a copy.

        Returns:
            A NestedTensor with the same structure as the given `params`.
        """
        cfg = self.config
        params = type(params)(**params)  # Make a copy of `params`.
        for name, child in self._children.items():
            if not isinstance(child, BaseLayer):
                # `child` is not a BaseLayer and does not have parameters, e.g., it can be an
                # instance of TensorStats.
                continue
            prng_key, child_key = jax.random.split(prng_key)
            params[name] = child.apply_parameter_noise_recursively(child_key, params[name])
        if cfg.param_noise is not None:
            params = self._param_noise.apply(prng_key, params)
        return params

    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if not name.endswith("weight"):
            return None
        if len(parameter_spec.shape) != 2:
            raise NotImplementedError(
                f"{type(self)} uses the default _compute_fan_axes, which requires weight "
                f"parameters to have exactly 2 axes: shape({name}) = {parameter_spec.shape}"
            )
        return FanAxes(in_axis=-2, out_axis=-1)

    def _remat_name(self, x: Tensor, name: str) -> Tensor:
        """Tags 'x' with 'name' using a custom jax.extend.core.Primitive, which
        is otherwise a no-op.

        This is useful for custom activation rematerialization policies, as it allows
        us to filter on tagged points in the jaxpr.

        Args:
            x: The tensor to tag.
            name: The name to tag 'x' with, after prefixing with the class name for 'self'.
        Returns:
            Tagged x.
        """
        full_name = f"{type(self).__name__}.{name}"
        return jax.ad_checkpoint.checkpoint_name(x, full_name)

    @property
    def parameters(self):
        return self.state

    def _add_tensor_stats(self, name: str, value: Nested[Tensor]):
        """Adds tensor stats about `value`.

        Suppose `self.tensor_stats` adds some summaries about `value`, e.g.,
            `self._add_tensor_stats("mean", jnp.mean(value))`.

        The "mean" summary will show up under path f"{self.path()}/tensor_stats/{name}/mean".

        Args:
            name: The name for the `value`. E.g., "inputs".
            value: A Tensor or a Nested[Tensor].
        """
        if "tensor_stats" in self.children:
            output_collection = new_output_collection()
            with child_context("tensor_stats", output_collection=output_collection):
                with child_context(name, module=self.tensor_stats):
                    self.tensor_stats.add_stats(name, value)
            layer_summaries = self.get_invocation_context().output_collection.summaries
            if "tensor_stats" not in layer_summaries:
                layer_summaries["tensor_stats"] = {}
            layer_summaries["tensor_stats"][name] = output_collection.summaries[name]

    def _add_activation_summary(
        self, *, name: str, activations: Tensor, activation_paddings: Optional[Tensor] = None
    ):
        """Add activation summaries.

        TODO(ruoming): use cfg.tensor_stats to represent activation summaries.

        Args:
            name: Activation name.
            activations: Layer activations tensor of shape [batch_size, ...].
            activation_paddings: Optional 0/1 tensor of shape [batch_size, seq_len].
                Paddings of the activations.
        """
        if activation_paddings is None:
            weights = activations.shape[0]
            reduction_axis = tuple(range(1, activations.ndim))
            d = math.prod(activations.shape[1:])
            activations_mean = jnp.mean(activations)
            # [batch_size, ...].
            sum_x2 = (activations * activations).sum(axis=reduction_axis, keepdims=True)
            activations_norm_mean = jnp.mean(jax.lax.sqrt(sum_x2) / jnp.sqrt(float(d)))
        else:
            reduction_axis = tuple(range(2, activations.ndim))
            d = math.prod(activations.shape[2:])
            expanded_shape = list(activations.shape[:2]) + [1] * (activations.ndim - 2)
            weights = jnp.sum(1 - activation_paddings)
            activations_sum = jnp.sum(
                activations * jnp.reshape(1 - activation_paddings, expanded_shape)
            )
            activations_mean = activations_sum / jnp.maximum(1, weights) / d

            # [batch_size, seq_len, ...].
            x2_sum = (activations * activations).sum(axis=reduction_axis, keepdims=True)
            activations_norm = jax.lax.sqrt(x2_sum) / jnp.sqrt(float(d))
            activations_norm_mean = jnp.sum(
                activations_norm * jnp.reshape(1 - activation_paddings, expanded_shape)
            ) / jnp.maximum(1, weights)

        # All hidden units average.
        self.add_summary(f"activations/{name}_mean", WeightedScalar(activations_mean, weights))
        # Average of per hidden unit norm.
        self.add_summary(f"activations/{name}_norm", WeightedScalar(activations_norm_mean, weights))


def no_remat(fn: Callable) -> Callable:
    """Decorator to exclude a method from rematerialization (remat/checkpoint).

    When a layer has remat_spec configured, ALL its methods are wrapped with jax.checkpoint
    by default, which requires all inputs and outputs to be JAX types (arrays). This causes
    problems for helper methods that work with Python values like strings or configuration
    objects.

    Use @no_remat to exclude such helper methods from rematerialization.

    When to use @no_remat:
    - Methods that process non-JAX types (strings, configs, Python objects)
    - Methods that perform compile-time logic (if/else on string values)
    - Pure utility methods that don't involve tensor computations
    - Methods that would fail with "abstract tracer" errors under remat

    Example problematic method that needs @no_remat:
        ```python
        @no_remat  # Required because method uses string comparison
        def get_activation(self, name: str):
            if name == 'relu':
                return jax.nn.relu
            elif name == 'gelu':
                return jax.nn.gelu
        ```

    Without @no_remat, the above would fail because remat converts the string
    argument to an abstract tracer, making string comparison impossible.

    Implementation note: This sets a _no_remat attribute that _maybe_wrap_with_remat
    checks to skip wrapping the method with jax.checkpoint.

    Args:
        fn: The method to exclude from rematerialization.

    Returns:
        The same method marked to skip rematerialization.
    """
    # pylint: disable=protected-access
    fn._no_remat = True
    return fn
