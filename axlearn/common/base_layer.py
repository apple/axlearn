# Copyright © 2023 Apple Inc.

"""Defines BaseLayer, the base class for layer implementations."""
import dataclasses
import math
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax
from absl import logging
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.config import ConfigOr, Configurable, config_class, maybe_instantiate
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, child_context, current_context, new_output_collection
from axlearn.common.param_init import DefaultInitializer, FanAxes
from axlearn.common.utils import (
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
    """A FactorizationSpec describes how to factorize a parameter's gradient. Used for Adafactor."""

    # A list of None/str corresponding to the axes of the parameter shape. Each element is either:
    # None:  no factorization along this axis.
    # str: the factorization axis name.
    #
    # For adafactor, either all axes are None or exactly two of the axes are "row" and "col",
    # respectively.
    axes: Sequence[Optional[str]]


# NestedFactorizationSpec = Dict[str, Union[FactorizationSpec, "NestedFactorizationSpec"]]
NestedFactorizationSpec = Dict[str, Union[FactorizationSpec, Any]]


# ParameterSpec is a dataclass so that jax.tree_util.tree_map does not expand it.
@dataclasses.dataclass
class ParameterSpec(TensorSpec):
    """Specification of a layer parameter."""

    # dtype (defined in TensorSpec):
    # The data type of the param. If None, uses the layer's default dtype.
    #
    # mesh_axes (defined in TensorSpec):
    # If mesh_axes is None, the parameter will not be partitioned and will be replicated.
    # If mesh_axes is a sequence, it should have exactly len(shape) elements.
    # mesh_axes[i] describes partitioning for shape[i], where each value can be:
    # - None: do not partition along this axis, or
    # - 'model': partition along this axis across the 'model' dim of the device mesh.
    # - 'data': partition along this axis across the 'data' dim of the device mesh.

    # The initializer of the param. If None, uses the layer's default initializer.
    initializer: Optional[param_init.Initializer] = None
    # Factorization spec for the parameter.
    factorization: Optional[FactorizationSpec] = None

    # Per-parameter weight decay / l2 regularization scale.
    #
    # The effective weight decay rate will be global_weight_decay_rate * param_weight_decay_scale.
    #
    # A layer implementation can override _create_layer_parameter_specs() or
    # create_parameter_specs_recursively() to set the weight_decay_scale of its parameters.
    # Specifically, set the scale to 0 to disable weight decay on the parameter.
    #
    # User can also set the `per_param_scale` arg of `add_decayed_weights` or the
    # `weight_decay_per_param_scale` arg of optimizers with custom logic to compute per-parameter
    # scales given a parameter tree, e.g., by matching parameter paths to regex-based rules.
    #
    # Note that ParameterSpec.weight_decay_scale takes precedence over `per_param_scale`.
    weight_decay_scale: Optional[float] = None


# When pytype supports recursive types, switch to:
# Optional[Union[ParameterSpec, Dict[str, "NestedParameterSpec"]]]
NestedParameterSpec = Optional[Union[ParameterSpec, Dict[str, Any]]]


@dataclasses.dataclass
class RematSpec:
    """RematSpec captures the configurable arguments for 'jax.remat'.

    https://github.com/google/jax/blob/1b79caa/jax/_src/ad_checkpoint.py#L99
    """

    prevent_cse: bool = True
    policy: Optional[ConfigOr[Callable[..., bool]]] = None


class ParameterNoise(Configurable):
    """An interface for applying parameter noise."""

    def apply(self, prng_key: jax.random.KeyArray, params: NestedTensor) -> NestedTensor:
        """To be implemented by subclasses."""
        raise NotImplementedError(self)


class BaseLayer(Module):
    """A base class for layer implementations."""

    @config_class
    class Config(Module.Config):
        """Configures BaseLayer."""

        # If not None, the default parameter dtype.
        # If None, inherits from the parent module.
        dtype: Optional[jnp.dtype] = None

        # If not None, parameter initialization config of this module.
        # If None, inherits from the parent module.
        param_init: Optional[DefaultInitializer.Config] = None

        # The partition spec for the layer parameters.
        # When the layer contains a weight parameter and a bias parameter,
        # the partition spec will be defined in terms of the weight parameter,
        # while the partition spec of the bias parameter can be derived accordingly.
        param_partition_spec: NestedParameterSpec = None

        # A RematSpec containing kwargs used by jax.remat as it wraps this layer.
        # If None, leaves XLA to figure out how to handle rematerialization without guidance.
        remat_spec: Optional[RematSpec] = None

        # If not None, BaseLayer.apply_parameter_noise_recursively() will apply noise to the given
        # parameters.
        #
        # `apply_parameter_noise_recursively` is not called by BaseLayer.forward() by default and
        # should be called by the trainer explicitly.
        #
        # `apply_parameter_noise_recursively` calls the child layers to apply noise (if any)
        # before applying the parent layer's noise (if any).
        param_noise: Optional[ParameterNoise.Config] = None

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
        self._remat_methods = []  # List[str]. Used for testing.

    def _methods_to_wrap_for_auto_child_context(self) -> Dict[str, Callable]:
        return {
            method: method_fn
            for method, method_fn in super()._methods_to_wrap_for_auto_child_context().items()
            if not hasattr(BaseLayer, method)
        }

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
    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Subclasses can override this method to add layer parameters."""
        return {}

    def _call_thunk(self, *args, method_fn, **kwargs) -> Callable[[], Any]:
        cfg: Module.Config = self.config
        self.vlog(
            3, "BaseLayer._call_thunk %s.%s: remat=%s", self.path(), method_fn, cfg.remat_spec
        )

        has_no_remat = getattr(method_fn, "_no_remat", False)

        nullary = super()._call_thunk(*args, method_fn=method_fn, **kwargs)
        if (
            current_context() is None
            or cfg.remat_spec is None
            or not self.is_training
            or has_no_remat
        ):
            return nullary

        # Remat always uses abstract tracers even if concrete information is available.
        # This means that all inputs and outputs to a remat function need to be JAX types.
        # We print a nice error if the inputs are not.
        check_jax_type(args=args, kwargs=kwargs, msg=f"Call to {self}.{method_fn} failed.")

        def nullary_with_remat():
            def fn(*args, **kwargs):
                """Unlike self.method, fn returns (outputs, output_collection)."""
                output_collection = new_output_collection()
                # We override output_collection to avoid leaking tracers.
                with child_context("remat", module=self, output_collection=output_collection):
                    outputs = method_fn(self, *args, **kwargs)
                return outputs, output_collection

            logging.info("Applying remat on %s.%s: %s", self.path(), method_fn, cfg.remat_spec)
            self._remat_methods.append(method_fn.__name__)
            # Pass both outputs and output_collection through remat(...) to avoid leaking tracers.
            outputs, output_collection = jax.ad_checkpoint.remat(
                fn,
                **{k: maybe_instantiate(v) for k, v in dataclasses.asdict(cfg.remat_spec).items()},
            )(*args, **kwargs)
            self.get_invocation_context().output_collection.update(output_collection)
            return outputs

        return nullary_with_remat

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs: Dict[str, NestedParameterSpec] = {}
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
            param_spec = dataclasses.replace(param_spec, mesh_axes=PartitionSpec(*partition_spec))
            if param_spec.dtype is None:
                param_spec = dataclasses.replace(param_spec, dtype=self.dtype())
            specs[name] = param_spec
        for name, child in self._children.items():
            assert name not in specs
            specs[name] = child.create_parameter_specs_recursively()
        return specs

    def initialize_parameters_recursively(
        self,
        prng_key: jax.random.KeyArray,
        *,
        prebuilt: Optional[NestedTensor] = None,
    ) -> NestedTensor:
        params = {}
        param_specs = self._create_layer_parameter_specs()
        for name, spec in param_specs.items():
            # Note: we split the key even if value is prebuilt.
            prng_key, child_key = jax.random.split(prng_key)
            value = get_or_none(prebuilt, name)
            if value is not None:
                params[name] = value
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
            assert name not in params
            prng_key, child_key = jax.random.split(prng_key)
            params[name] = child.initialize_parameters_recursively(
                prng_key=child_key,
                prebuilt=get_or_none(prebuilt, name),
            )
        return params

    def _use_prebuilt_params(self, prebuilt: Optional[NestedTensor]) -> bool:
        prebuilt_keys = set(
            key for key, value in flatten_items(prebuilt) if isinstance(value, Tensor)
        )
        if not prebuilt_keys:
            return False
        param_keys = set(key for key, _ in flatten_items(self.create_parameter_specs_recursively()))
        if prebuilt_keys != param_keys:
            raise NotImplementedError(
                f"Partial prebuilt params are not supported: {param_keys - prebuilt_keys}"
            )
        return True

    def _initialize_parameter(
        self, name: str, *, prng_key: jax.random.KeyArray, parameter_spec: ParameterSpec
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
            axes=self._compute_fan_axes(name, parameter_spec),
        )
        return param

    def apply_parameter_noise_recursively(
        self, prng_key: jax.random.KeyArray, params: NestedTensor
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
            prng_key, child_key = jax.random.split(prng_key)
            params[name] = child.apply_parameter_noise_recursively(child_key, params[name])
        if cfg.param_noise is not None:
            params = self._param_noise.apply(prng_key, params)
        return params

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if not name.endswith("weight"):
            return None
        if len(parameter_spec.shape) < 2:
            raise NotImplementedError(
                "Default _compute_fan_axes requires weight parameters to have at least 2 axes "
                f"shape({name}) = {parameter_spec.shape}"
            )
        return FanAxes(in_axis=-2, out_axis=-1)

    def _remat_name(self, x: Tensor, name: str) -> Tensor:
        """Tags 'x' with 'name' using a custom jax.core.Primitive, which is otherwise a no-op.

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

    def _add_activation_summary(
        self, *, name: str, activations: Tensor, activation_paddings: Optional[Tensor] = None
    ):
        """Add activation summaries.

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
    """Annotates fn so that remat will not be applied to it.

    This can be used to prevent tracers from leaking into helper methods that depend
    only on data available at compile time when using `remat_spec`. For example, the following
    method cannot be used in a class that uses remat_spec without using @no_remat:

    ```
    def fn(self, st: str):
        if st=='three':
            return 3
    ```

    Args:
        fn: The method to annotate.

    Returns:
        The input `fn` after having been annotated.
    """
    # pylint: disable=protected-access
    fn._no_remat = True
    return fn
