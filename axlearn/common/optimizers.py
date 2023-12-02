# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# deepmind/optax:
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/lingvo:
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/automl:
# Copyright 2023 Google Research. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

# pylint: disable=too-many-lines
"""Optimization modules."""
import dataclasses
import re
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import chex
import jax
import optax
from absl import logging
from jax import numpy as jnp
from optax._src import numerics

from axlearn.common import schedule
from axlearn.common.base_layer import NestedParameterSpec, ParameterSpec, PartitionSpec
from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.factorized_rms import scale_by_factored_rms
from axlearn.common.module import current_context
from axlearn.common.optimizer_base import (
    NestedOptParam,
    OptParam,
    OptStateSpec,
    PartitionedGradientTransformation,
    TransformPartitionSpecFn,
)
from axlearn.common.utils import (
    NestedPartitionSpec,
    NestedTensor,
    NestedTree,
    Tensor,
    TensorSpec,
    register_per_param_settings,
    tree_paths,
    vectorized_tree_map,
)


def _to_partitioned_transformation(transformation: ConfigOr[PartitionedGradientTransformation]):
    transformation = maybe_instantiate(transformation)
    if not isinstance(transformation, PartitionedGradientTransformation):
        raise ValueError(
            "Expected PartitionedGradientTransformation. "
            f"Got {type(transformation)}: {transformation}"
        )
    return transformation


def chain(*args):
    args = [_to_partitioned_transformation(e) for e in args]
    base = optax.chain(*[optax.GradientTransformation(init=e.init, update=e.update) for e in args])

    def partition(param_spec):
        return tuple(e.partition(param_spec) for e in args)

    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=partition
    )


@dataclasses.dataclass
class SubOptimizerRule:
    param_regex: str  # Parameter path regex.
    optimizer: ConfigOr[PartitionedGradientTransformation]


def _no_op():
    def update_fn(
        updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam
    ) -> Tuple[NestedTensor, optax.EmptyState]:
        del params
        return updates, state

    return PartitionedGradientTransformation(
        init=lambda params: optax.EmptyState(),
        update=update_fn,
        partition=lambda param_specs: optax.EmptyState(),
    )


def opt_param_values(params: NestedOptParam) -> NestedTensor:
    return jax.tree_util.tree_map(lambda opt_param: opt_param.value, params)


def with_partition_fn(
    base: optax.GradientTransformation, partition_fn: TransformPartitionSpecFn
) -> PartitionedGradientTransformation:
    def init_fn(params: NestedOptParam) -> NestedTensor:
        return base.init(opt_param_values(params))

    def update_fn(
        updates: optax.Updates, state: optax.OptState, params: NestedOptParam
    ) -> Tuple[optax.Updates, optax.OptState]:
        return base.update(updates, state, opt_param_values(params))

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def copy_partition(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
    return jax.tree_util.tree_map(
        lambda param_spec: OptStateSpec(
            dtype=param_spec.dtype, shape=param_spec.shape, mesh_axes=param_spec.mesh_axes
        ),
        param_specs,
    )


def trace_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        return optax.TraceState(trace=copy_partition(param_specs))

    return with_partition_fn(base, partition_fn)


def adam_partition(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    state: optax.ScaleByAdamState = base.init({})

    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        return optax.ScaleByAdamState(
            count=OptStateSpec(
                dtype=state.count.dtype, shape=state.count.shape, mesh_axes=PartitionSpec()
            ),
            mu=copy_partition(param_specs),
            nu=copy_partition(param_specs),
        )

    return with_partition_fn(base, partition_fn)


def scale(step_size: float) -> PartitionedGradientTransformation:
    return with_partition_fn(optax.scale(step_size), lambda _: optax.ScaleState())


def scale_by_schedule(step_size_fn: schedule.Schedule) -> PartitionedGradientTransformation:
    return with_partition_fn(
        optax.scale_by_schedule(schedule.as_schedule_fn(step_size_fn)),
        lambda _: optax.ScaleByScheduleState(
            count=OptStateSpec(shape=[], dtype=jnp.int32, mesh_axes=PartitionSpec())
        ),
    )


def scale_from_learning_rate(
    learning_rate: schedule.Schedule, *, flip_sign=True
) -> schedule.ScheduleFn:
    learning_rate_fn = schedule.as_schedule_fn(learning_rate)

    def scale_fn(step):
        lr = learning_rate_fn(step)
        context = current_context()
        if context:
            context.add_summary("lr_schedule_step", step)
            context.add_summary("learning_rate", lr)
        return -lr if flip_sign else lr

    return scale_fn


def per_param_scale_by_path(
    *, scale_by_path: Sequence[Tuple[str, float]], description: str, default_scale: float = 1.0
) -> Callable[[NestedOptParam], Any]:
    """Computes per-parameter scales with regex-based rules.

    Args:
        scale_by_path: a list of (regex, scale) pairs. The first regex pattern fully matching the
            parameter path determines the weight decay scale for the parameter.
        description: a string of what the per-parameter scale is used for in the model.
        default_scale: The scale to use if none of the regex patterns matches the parameter path.

    Returns:
        A function that computes per-parameter scales given a NestedOptParam tree. The returned
        scales will be a tree with the same structure, but with float numbers instead of OptParam
        as leaves.
    """

    def fn(params: NestedOptParam) -> Any:
        def per_param_fn(param_path: str):
            # TODO(ruoming): use `match_regex_rules`.
            for path_regex, path_scale in scale_by_path:
                if re.fullmatch(path_regex, param_path):
                    logging.info(
                        "%s per_param_scale_by_path: %s matches %s: scale=%s",
                        description,
                        path_regex,
                        param_path,
                        path_scale,
                    )
                    return path_scale
            logging.info(
                "%s per_param_scale_by_path: using default scale for %s: scale=%s",
                description,
                param_path,
                default_scale,
            )
            return default_scale

        return jax.tree_util.tree_map(per_param_fn, tree_paths(params, separator="/"))

    return fn


def per_param_scale_by_rms(*, min_scale: float = 1e-4) -> Callable[[NestedOptParam], NestedTree]:
    """Computes per-parameter scales with its Root-Mean-Square (RMS).

    Args:
        min_scale: The minimum scale for each paramter.

    Returns:
        A function that computes per-parameter scales given a NestedOptParam tree. The returned
        scales will be a tree with the same structure, but with float numbers instead of OptParam
        as leaves.
    """

    def fn(params: NestedOptParam) -> Any:
        return jax.tree_util.tree_map(
            lambda p: optax.safe_root_mean_squares(p.value, min_scale), params
        )

    return fn


def scale_by_trust_ratio(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.0,
    eps: float = 0.0,
) -> PartitionedGradientTransformation:
    """Scale updates by trust ratio`.

    Based on:
    <https://github.com/deepmind/optax/blob/5f0f5da11477b9321baad719cae1f46b7758b203/optax/_src/transform.py#L818>

    Args:
      min_norm: minimum norm for params and gradient norms; by default is zero.
      trust_coefficient: a multiplier for the trust ratio.
      eps: additive constant added to the denominator for numerical stability.

    Returns:
      A corresponding `PartitionedGradientTransformation`.
    """

    def init_fn(params):
        del params
        return optax.ScaleByTrustRatioState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("Params must not be None for trust ratio scaling!")

        def _scale_update(update, param):
            # Clip norms to minimum value, by default no clipping.
            param_norm = numerics.safe_norm(param, min_norm)
            update_norm = numerics.safe_norm(update, min_norm)
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

            # If no minimum norm clipping is used
            # Set trust_ratio to 1 in case where parameters would never be updated.
            zero_norm = jnp.logical_or(param_norm == 0.0, update_norm == 0.0)
            safe_trust_ratio = jnp.where(zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

            return update * safe_trust_ratio

        # The only difference from the optax implementation:
        # vectorized_tree_map vs. jax.tree_util.tree_map.
        updates = vectorized_tree_map(_scale_update, updates, params)
        return updates, state

    return with_partition_fn(
        optax.GradientTransformation(init_fn, update_fn), lambda _: optax.ScaleByTrustRatioState()
    )


class AddDecayedWeightsState(NamedTuple):
    count: Optional[Tensor]  # Number of steps.


def scale_update_per_param(
    per_param_scale: Callable[[NestedOptParam], Any],
) -> PartitionedGradientTransformation:
    """Scales updates based on `per_param_scale`.

    Args:
        per_param_scale: a Callable that returns a tree with same structure as
            the params PyTree, where each leaf is a float scalar, indicating the
            scaling factor for the parameter update. Specifically a scale of 0
            will disable updates to the parameter and can be used when we want
            to freeze a subset of parameters. per_param_scale can be computed
            by `per_param_scale_by_path`.

    Returns:
        A PartitionedGradientTransformation.
    """

    def init_fn(params):
        del params

        return optax.EmptyState()

    def update_fn(
        updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam
    ) -> Tuple[NestedTensor, optax.EmptyState]:
        if params is None:
            raise ValueError(optax.NO_PARAMS_MSG)  # pylint: disable=no-member

        param_scales = maybe_instantiate(per_param_scale)(params)
        register_per_param_settings(
            param_scales,
            description=getattr(per_param_scale, "description", "update_scale"),
        )

        updates = jax.tree_map(
            # Apply the scaling to each update.
            lambda g, m: g * m,
            updates,
            param_scales,
        )

        return updates, state

    return PartitionedGradientTransformation(
        init=init_fn,
        update=update_fn,
        partition=lambda param_specs: optax.EmptyState(),
    )


def _weight_decay_scales(
    params: NestedOptParam,
    *,
    per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
) -> NestedTree:
    """Returns a nested tree with float leaf nodes."""
    if per_param_scale is None:
        param_scales = jax.tree_util.tree_map(lambda _: 1, params)
    else:
        param_scales = maybe_instantiate(per_param_scale)(params)

    # `param.weight_decay_scale` takes precedence over `per_param_scale`.
    def maybe_override_scale(path: str, param: OptParam, curr_scale: float) -> float:
        if param.weight_decay_scale is not None and param.weight_decay_scale != curr_scale:
            logging.info(
                "Overriding the weight decay scale of %s, "
                "according to ParameterSpec.weight_decay_scale, from %s to %s",
                path,
                curr_scale,
                param.weight_decay_scale,
            )
            return param.weight_decay_scale
        return curr_scale

    scales = jax.tree_util.tree_map(maybe_override_scale, tree_paths(params), params, param_scales)
    return register_per_param_settings(
        scales,
        description=getattr(per_param_scale, "description", "weight_decay_scale"),
    )


def add_decayed_weights(
    weight_decay: float,
    *,
    learning_rate_exponent: Optional[float] = None,
    learning_rate: Optional[schedule.Schedule] = None,
    per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
) -> PartitionedGradientTransformation:
    """Add parameter scaled by `weight_decay * (learning_rate ** learning_rate_exponent)`.

    Unlike optax.add_decayed_weights, this supports scaling weight decay by the learning rate
    and per-parameter scaling.

    Reference:
    https://github.com/google/praxis/blob/f352ccdffa438e6bf7cbaa30e23961b00df28f4b/praxis/optimizers.py#L1666

    Args:
        weight_decay: the base weight decay rate (before learning rate and per-param scaling).
        learning_rate_exponent: the exponent on learning_rate to scale the weight decay.
            If None, do not scale by the learning rate.
        learning_rate: the learning rate schedule. Required if learning_rate_exponent is not None.
        per_param_scale: (optional) a Callable that returns a tree with same structure as
            the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
            Note that `per_param_scale` does not override weight decay scales for the parameters
            that have `weight_decay_scale` explicitly specified in the ParameterSpec.
            See base_layer.ParameterSpec.weight_decay_scale for details.

    Returns:
        A PartitionedGradientTransformation.
    """

    def init_fn(params):
        del params
        if learning_rate_exponent is None:
            count = None
        else:
            count = jnp.zeros([], jnp.int32)
        return AddDecayedWeightsState(count=count)

    def update_fn(updates: NestedTensor, state: AddDecayedWeightsState, params: NestedOptParam):
        if params is None:
            raise ValueError(optax.NO_PARAMS_MSG)  # pylint: disable=no-member

        if not learning_rate_exponent:
            lr_scale = 1.0
        else:
            learning_rate_fn = schedule.as_schedule_fn(learning_rate)
            lr = learning_rate_fn(state.count)
            lr_scale = lr**learning_rate_exponent

        param_scales = _weight_decay_scales(params, per_param_scale=per_param_scale)
        updates = jax.tree_util.tree_map(
            lambda g, p, s: g + weight_decay * lr_scale * p.value * s,
            updates,
            params,
            param_scales,
        )
        if learning_rate_exponent is None:
            updated_state = state
        else:
            updated_state = AddDecayedWeightsState(optax.safe_int32_increment(state.count))
        return updates, updated_state

    def partition_fn(param_specs):
        del param_specs
        if learning_rate_exponent is None:
            count = None
        else:
            count = OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec())
        return AddDecayedWeightsState(count=count)

    return PartitionedGradientTransformation(
        init=init_fn,
        update=update_fn,
        partition=partition_fn,
    )


def l2_regularizer(
    regularizer_weight: Optional[float] = 0.0,
    per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
) -> PartitionedGradientTransformation:
    """Adds l2_regularization scaled by `regularizer_weight` to gradients.

    This is the L2 regularization in paper Decoupled Weight Decay Regularization.
    https://arxiv.org/pdf/1711.05101.pdf. Note at most one of L2 regularizer and weight decay
    should be applied.

    Args:
        regularizer_weight: the l2 regularizer weight.
        per_param_scale: (optional) a Callable that returns a tree with same structure as
            the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global regularizer_weight. If None, all leaves
            will have a scale of 1.
            Note that `per_param_scale` does not override weight decay scales for the parameters
            that have `weight_decay_scale` explicitly specified in the ParameterSpec.
            See base_layer.ParameterSpec.weight_decay_scale for details.

    Returns:
        A PartitionedGradientTransformation.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam):
        if regularizer_weight:
            if params is None:
                raise ValueError(optax.NO_PARAMS_MSG)  # pylint: disable=no-member
            param_scales = _weight_decay_scales(params, per_param_scale=per_param_scale)
            updates = jax.tree_util.tree_map(
                lambda g, p, s: g + regularizer_weight * p.value * s,
                updates,
                params,
                param_scales,
            )

        return updates, state

    return PartitionedGradientTransformation(
        init=init_fn,
        update=update_fn,
        partition=lambda _: optax.EmptyState(),
    )


def sgd_optimizer(
    learning_rate: schedule.Schedule,
    *,
    decouple_weight_decay: bool,
    momentum: float = 0,
    weight_decay: float = 0,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
) -> PartitionedGradientTransformation:
    """SGD optimizer implementation.

    Args:
        learning_rate: the learning rate schedule.
        decouple_weight_decay: Decouples weight decay so that it is not
            part of the gradient and thus do not affect the gradient
            accumulators. A brief guidance:
            - If you are trying to reproduce an existing model trained
                with SGD, you probably want to set it to `False` to get the
                same behavior as in Torch or TF;
            - If you are tuning a new model and plan to tune the weight
                decay rate, you may want to set it to `True` to get a
                simpler behavior.
            Reference:
            https://arxiv.org/abs/1711.05101
            https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw
        momentum: the momentum for SGD update.
        weight_decay: the weight decay rate.
        weight_decay_per_param_scale: the per-param decay scale. The scale
            will be applied on top of the global decay rate.

    Returns:
        A corresponding `PartitionedGradientTransformation`.
    """
    if decouple_weight_decay:
        return chain(
            trace_partition(optax.trace(decay=momentum)),
            add_decayed_weights(
                weight_decay=weight_decay,
                # Weight decay updates will already be scaled by the learning rate below.
                learning_rate_exponent=None,
                per_param_scale=weight_decay_per_param_scale,
            ),
            scale_by_schedule(scale_from_learning_rate(learning_rate)),
        )
    else:
        return chain(
            add_decayed_weights(
                weight_decay=weight_decay,
                # Weight decay updates will already be scaled by the learning rate below.
                learning_rate_exponent=None,
                per_param_scale=weight_decay_per_param_scale,
            ),
            trace_partition(optax.trace(decay=momentum)),
            scale_by_schedule(scale_from_learning_rate(learning_rate)),
        )


def adamw_optimizer(
    learning_rate: schedule.Schedule,
    *,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float = 0,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
    mu_dtype: Optional[jnp.dtype] = None,
    multiply_by_parameter_scale: bool = False,
) -> PartitionedGradientTransformation:
    """AdamW optimizer with parameter scaling.

    N.B. The default weight-decay implementation is consistent with
        those in e.g. PyTorch & Optax, but inconsistent with the "decoupled"
        adamw weight decay formulation in <https://arxiv.org/abs/1711.05101> Algorithm 2.
        To faithfully replicate Algorithm 2, use `adamw_decoupled_optimizer`.

    Args:
        learning_rate: the learning rate schedule.
        b1: the exponential decay rate for the 1st moment estimates.
        b2: the exponential decay rate for the 2nd moment estimates.
        eps: a small constant for numerical stability.
        weight_decay: optional rate at which to decay weights.
        weight_decay_per_param_scale: a Callable that returns a tree with same structure
            as the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
        mu_dtype: optional `dtype` to be used for the first order accumulator;
            if `None` then the dtype is inferred from params and updates.
        multiply_by_parameter_scale: if `True`, then scale learning_rate by
            parameter RMS. if `False`, provided learning_rate is absolute step size.
            Usually this should be left as False.

    Returns:
        A PartitionedGradientTransformation representing an AdamW optimizer with parameter scaling.
    """
    tx = [adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype))]
    # Add the per-parameter scaling (PPS) according to the adafactor optimizer.
    if multiply_by_parameter_scale:
        tx.append(scale_by_param_block_rms())
    tx.extend(
        [
            add_decayed_weights(
                weight_decay=weight_decay,
                # Weight decay updates will already be scaled by the learning rate below.
                learning_rate_exponent=None,
                per_param_scale=weight_decay_per_param_scale,
            ),
            scale_by_schedule(scale_from_learning_rate(learning_rate)),
        ]
    )

    return chain(*tx)


def adamw_decoupled_optimizer(
    learning_rate: float,
    *,
    b1: float,
    b2: float,
    eps: float,
    update_schedule: schedule.Schedule,
    weight_decay: float = 0,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
    mu_dtype: Optional[jnp.dtype] = None,
    multiply_by_parameter_scale: bool = False,
) -> PartitionedGradientTransformation:
    """A "decoupled" version of the AdamW optimizer, with optional parameter scaling.

    Farthfully replicates Adam with "decoupled" weight decay from
    <https://arxiv.org/abs/1711.05101> Algorithm 2.
    Specifically, `learning_rate`, `weight_decay`, and `update_schedule` correspond to
    `alpha`, `lambda`, and `eta` in Algorithm 2, respectively.

    Args:
        learning_rate: the learning rate (will be scaled by the update_schedule).
        b1: the exponential decay rate for the 1st moment estimates.
        b2: the exponential decay rate for the 2nd moment estimates.
        eps: a small constant for numerical stability.
        update_schedule: an update schedule, which is applied to scale both the learning rate
            and the weight decay.
        weight_decay: optional rate at which to decay weights (will be scaled by update_schedule).
        weight_decay_per_param_scale: a Callable that returns a tree with same structure
            as the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
        mu_dtype: optional `dtype` to be used for the first order accumulator;
            if `None` then the dtype is inferred from params and updates.
        multiply_by_parameter_scale: if `True`, then scale learning_rate by
            parameter RMS. if `False`, provided learning_rate is absolute step size.
            Usually this should be left as False.

    Returns:
        A PartitionedGradientTransformation representing a decoupled AdamW optimizer with
            parameter scaling.
    """
    tx = [adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype))]
    # Add the per-parameter scaling (PPS) according to the adafactor optimizer.
    if multiply_by_parameter_scale:
        tx.append(scale_by_param_block_rms())
    tx.extend(
        [
            # Scale the update by the fixed learning rate.
            scale_by_schedule(scale_from_learning_rate(learning_rate, flip_sign=False)),
            add_decayed_weights(
                weight_decay=weight_decay,
                learning_rate_exponent=None,
                per_param_scale=weight_decay_per_param_scale,
            ),
            # Scale the overall update by the update schedule.
            scale_by_schedule(update_schedule),
            # Invert the sign.
            scale(-1.0),
        ]
    )

    return chain(*tx)


def adam_optimizer(
    learning_rate: schedule.Schedule,
    *,
    b1: float,
    b2: float,
    eps: float,
    l2_regularizer_weight: float = 0,
    l2_regularizer_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
    mu_dtype: Optional[jnp.dtype] = None,
) -> PartitionedGradientTransformation:
    """Adam optimizer with l2 regularization."""
    tx = [
        l2_regularizer(
            regularizer_weight=l2_regularizer_weight,
            per_param_scale=l2_regularizer_per_param_scale,
        ),
        adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype)),
        scale_by_schedule(scale_from_learning_rate(learning_rate)),
    ]
    return chain(*tx)


class EmaState(NamedTuple):
    # Captures an exponential moving average for all params.
    count: Tensor  # Number of times the EMA has been updated.
    ema: NestedTree  # Exponential moving averages.
    scale: NestedTree  # Scale applied when/if reversing EMA quantization.


def ema(
    decay: schedule.Schedule,
    debias: bool = True,
    accumulator_dtype: Optional[jnp.dtype] = jnp.float32,
) -> PartitionedGradientTransformation:
    """Compute an exponential moving average of updates.

    Based on:
    <https://github.com/deepmind/optax/blob/252d15/optax/_src/transform.py#L120-L158>
    <https://github.com/tensorflow/lingvo/blob/4c0252f/lingvo/jax/optimizers.py#L1780-L1858>

    Args:
        decay: the decay rate schedule for the exponential moving average.
        debias: whether to debias the transformed gradient.
        accumulator_dtype: optional `dtype` to use for the accumulator; if `None`
            or if the parameter is a scalar then the `dtype` is inferred.
            Supports: float32, bfloat16, int16, int8.

    Returns:
        A corresponding `PartitionedGradientTransformation`.

    Raises:
        ValueError: If accumulator_dtype is invalid.
    """
    decay_fn = schedule.as_schedule_fn(decay)
    # Validate accumulator_dtype.
    float_dtypes = [jnp.float32, jnp.bfloat16]
    int_dtypes = [jnp.int16, jnp.int8]
    if accumulator_dtype is not None:
        valid_dtypes = float_dtypes + int_dtypes
        accumulator_dtype = jax.dtypes.canonicalize_dtype(accumulator_dtype)
        if accumulator_dtype not in valid_dtypes:
            raise ValueError(f"accumulator_dtype must be one of {valid_dtypes} if set.")

    def _should_quantize(t_shape: Sequence[int]):
        return t_shape and accumulator_dtype in int_dtypes

    @dataclasses.dataclass
    class _TensorEma:
        # The exponential moving average state and quantization scaling factor for a tensor.
        value: Tensor  # Current value of the momentum estimate for the tensor.
        qstep_size: Tensor  # Scaling factor, for converting 'value' to float if quantized, else 0.

    def _to_state(count: Tensor, ema_tree: NestedTree):
        return EmaState(
            count=count,
            ema=jax.tree_util.tree_map(lambda ema: ema.value, ema_tree),
            scale=jax.tree_util.tree_map(lambda ema: ema.qstep_size, ema_tree),
        )

    def init_fn(params):
        def _init(t):
            # Store momentum in accumulator_dtype if it is set and p is not scalar.
            if t.shape and accumulator_dtype is not None:
                value = jnp.zeros(t.shape, dtype=accumulator_dtype)
            else:
                value = jnp.zeros(t.shape, dtype=t.dtype)

            # Momentum scaling required if momentum quantized to int.
            if _should_quantize(t.shape):
                qstep_size = jnp.zeros(t.shape[1:], dtype=jnp.float32)
            else:
                qstep_size = jnp.zeros((1,), dtype=jnp.float32)
            return _TensorEma(value=value, qstep_size=qstep_size)

        return _to_state(jnp.zeros([], jnp.int32), jax.tree_util.tree_map(_init, params))

    @dataclasses.dataclass
    class _UpdateResult:
        # Captures an update and associated EMA value.
        update: Tensor  # The transformed update value.
        tensor_ema: _TensorEma  # The transformed tensor EMA.

    def update_fn(updates, state, params=None):
        del params
        decay_t = decay_fn(state.count)

        def _to_qint_tensor_ema(value: Tensor) -> _TensorEma:
            # Map value to integer with a symmetric quantization scheme.
            # E.g. [-0.5, 0.5] * step_size -> 0; (0.5, 1.5) * step_size -> 1.
            num_steps = jnp.array(jnp.iinfo(accumulator_dtype).max, dtype=value.dtype)
            # Use half the int* range for each side of zero.
            qstep_size = jnp.max(jnp.abs(value), axis=0, keepdims=True) / num_steps
            # Handle zero step sizes for any type of float input.
            divisor_qstep_size = jnp.where(qstep_size > 0.0, qstep_size, jnp.ones_like(qstep_size))
            quantized_value = jnp.round(value / divisor_qstep_size).astype(accumulator_dtype)
            return _TensorEma(value=quantized_value, qstep_size=qstep_size.squeeze(0))

        def _to_tensor_ema(value: Tensor) -> _TensorEma:
            # Convert tensor back to _TensorEma, quantizing first if required.
            if _should_quantize(value.shape):
                return _to_qint_tensor_ema(value)
            return _TensorEma(value=value.astype(accumulator_dtype), qstep_size=jnp.zeros((1,)))

        def _to_float(value: Tensor, qstep_size: Tensor) -> Tensor:
            # If necessary map quantized value back to float using qstep_size scale factor.
            if value.dtype in float_dtypes:
                return value
            return value.astype(qstep_size.dtype) * jnp.expand_dims(qstep_size, axis=0)

        # pylint: disable-next=redefined-outer-name
        def _update(value: Tensor, ema: Tensor, qstep_size: Tensor, count: Tensor) -> _UpdateResult:
            update = new_ema = (1 - decay_t) * value + decay_t * _to_float(ema, qstep_size)
            if debias:
                bias_correction = 1 - decay_t**count
                update = new_ema / bias_correction.astype(new_ema.dtype)
            return _UpdateResult(update=update, tensor_ema=_to_tensor_ema(new_ema))

        # Transform updates and compute new per-tensor EMA.
        count_inc = optax.safe_int32_increment(state.count)
        update_results = jax.tree_util.tree_map(
            lambda update, ema, scale: _update(update, ema=ema, qstep_size=scale, count=count_inc),
            updates,
            state.ema,
            state.scale,
        )

        # Unpack update, and pack state into EmaState.
        updates = jax.tree_util.tree_map(lambda ur: ur.update, update_results)
        new_state = _to_state(
            count=count_inc,
            ema_tree=jax.tree_util.tree_map(lambda ur: ur.tensor_ema, update_results),
        )
        return updates, new_state

    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        def get_ema_partition(param_spec: ParameterSpec) -> OptStateSpec:
            # Store momentum in accumulator_dtype if it is set and p is not scalar.
            if param_spec.shape and accumulator_dtype is not None:
                return OptStateSpec(
                    dtype=accumulator_dtype, shape=param_spec.shape, mesh_axes=param_spec.mesh_axes
                )
            else:
                return OptStateSpec(
                    dtype=param_spec.dtype, shape=param_spec.shape, mesh_axes=param_spec.mesh_axes
                )

        def get_scale_partition(param_spec: ParameterSpec) -> OptStateSpec:
            shape = param_spec.shape
            partition = param_spec.mesh_axes
            # This condition must be consistent with the one in init_fn().
            if _should_quantize(shape):
                # Copy the appropriate part of the parameter partition spec.
                # We took the max over each parameters first dimension, so flatten it for scale.
                return OptStateSpec(
                    dtype=jnp.float32,
                    shape=shape[1:],
                    mesh_axes=(
                        PartitionSpec(*partition[1:])
                        if partition is not None
                        else [None] * (len(shape) - 1)
                    ),
                )
            return OptStateSpec(
                dtype=jnp.float32,
                shape=(1,),
                mesh_axes=PartitionSpec(
                    None,
                ),
            )

        return EmaState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            ema=jax.tree_util.tree_map(get_ema_partition, param_specs),
            scale=jax.tree_util.tree_map(get_scale_partition, param_specs),
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def adafactor_optimizer(
    learning_rate: schedule.Schedule,
    *,
    b1: Optional[schedule.Schedule],
    b2: schedule.Schedule,
    multiply_by_parameter_scale: bool,
    clipping_threshold: Optional[float],
    dtype_momentum: Any = jnp.float32,
    weight_decay: Optional[float] = None,
    weight_decay_scale_by_learning_rate_exponent: Optional[float] = None,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
    eps: float = 1e-30,
    factored: bool = True,
    apply_scale_by_trust_ratio: bool = False,
) -> PartitionedGradientTransformation:
    """Adafactor optimizer.

    References:
    https://arxiv.org/abs/1804.04235
    https://github.com/deepmind/optax/blob/c4a4790b85ad69cda00a425cc3dcf9c9f9465120/optax/_src/alias.py#L77

    WARNING: unlike adamw_optimizer, decay bias correction will *not* be applied if b1 or b2
    is set to a constant. However, users can enable bias correction by setting b1/b2 to
    config_for_function(schedule.decay_bias_correction).set(decay=<constant decay>).

    Args:
        learning_rate: (Schedule) The learning rate schedule.
        b1: (Schedule) first-moment exponential decay (beta1) schedule. If not None, enables
            momentum and uses extra memory.
        b2: (Schedule) second-moment exponential decay (beta2) schedule.
        multiply_by_parameter_scale: (bool): if True, then scale learning_rate by
            parameter norm. if False, provided learning_rate is absolute step size.
            Usually this should be set to False.
        clipping_threshold: (float>=1) optional value; if None, clipping disabled.
        dtype_momentum: (dtype) dtype of momentum buffers.
        weight_decay: (float) optional rate at which to decay weights.
        weight_decay_scale_by_learning_rate_exponent: (float) optional scale weight decay rate by
            (learning_rate ** exponent). Must not be None if weight_decay is not None.
            If set to 1, replicates the behavior of Adafactor weight decay in Lingvo and Praxis,
            https://github.com/google/praxis/blob/8fa3eb2e9ade0fd9a89a2ca56187882b12871605/praxis/optimizers.py#L1991-L1995.
            Set to 0 to disable scaling.
        weight_decay_per_param_scale: (optional) a Callable that returns a tree with same structure
            as the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
        eps: (float) regularization constant for root mean squared gradient.
        factored: (bool) whether to use factored second-moment estimates.
        apply_scale_by_trust_ratio: (bool) whether to use variable-wise adaptive moments (LAMB):
            https://arxiv.org/abs/1904.00962.

    Returns:
        A PartitionedGradientTransformation representing an Adafactor optimizer.

    Raises:
        ValueError: If weight_decay_scale_by_learning_rate_exponent is not specified with
            weight_decay.
    """
    tx = [scale_by_factored_rms(factored, decay_rate=b2, epsilon=eps)]
    # This basic rescaling is typically combined with one or more of the following
    # transformation (all can be disabled via adafactor's constructor args).
    if clipping_threshold is not None:
        tx.append(clip_by_block_rms(clipping_threshold))
    if learning_rate is not None:
        tx.append(scale_by_schedule(scale_from_learning_rate(learning_rate, flip_sign=False)))
    if multiply_by_parameter_scale:
        tx.append(scale_by_param_block_rms())
    if b1 is not None:
        tx.append(ema(b1, debias=False, accumulator_dtype=dtype_momentum))
    if weight_decay is not None:
        if weight_decay_scale_by_learning_rate_exponent is None:
            raise ValueError(
                "weight_decay_scale_by_learning_rate_exponent must be specified "
                "when weight_decay is not None"
            )
        tx.append(
            add_decayed_weights(
                weight_decay=weight_decay,
                learning_rate_exponent=weight_decay_scale_by_learning_rate_exponent,
                learning_rate=learning_rate,
                per_param_scale=weight_decay_per_param_scale,
            )
        )
    # TODO(zirui_wang): Enable params to exclude from layer-wise adaptation.
    if apply_scale_by_trust_ratio:
        tx.append(scale_by_trust_ratio(eps=eps))
    # In gradient "descent" we follow the negative gradient.
    tx.append(scale(-1))
    return chain(*tx)


def clip_by_global_norm(
    max_norm: Optional[float] = None, *, drop_norm: Optional[float] = None, eps: float = 1e-8
) -> PartitionedGradientTransformation:
    """Scales gradients s.t. global norm <= max_norm, and drop gradients that exceed drop_norm.

    If the gradient global norm >= drop_norm, we set global gradients on all parameters to zero.
    Note the zero gradients are still processed by the optimizer, so the updates may not be zero;
    And non-gradient state updates (e.g., batch norm stat updates) are still processed as usual.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = optax.global_norm(updates)
        context = current_context()
        if context is not None:
            context.add_summary("gradient_norm", g_norm)
        g_scale = 1.0
        if max_norm is not None:
            g_scale = jnp.minimum(1.0, max_norm / (g_norm + eps))
        if drop_norm is not None:
            # Drops this gradient step if g_norm >= drop_norm.
            g_scale *= (g_norm < drop_norm).astype(g_norm.dtype)

        if max_norm is not None or drop_norm is not None:
            if context is not None:
                context.add_summary("gradient_scale", g_scale)
            updates = jax.tree_util.tree_map(lambda t: t * g_scale, updates)
        return updates, state

    return PartitionedGradientTransformation(
        init=init_fn, update=update_fn, partition=lambda _: optax.EmptyState()
    )


def clip_by_block_rms(threshold: float) -> PartitionedGradientTransformation:
    """Clip updates to a max rms for the gradient of each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.
    A sub tree under a VDict will be vectorized and clipped separately so that we
    clip updates to different layers of a Repeat/Pipeline layer separately.

    Args:
        threshold: the maximum rms for the gradient of each param vector or matrix.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params

        def _clip_fn(u):
            clip_denom = jnp.maximum(1.0, jnp.sqrt(jnp.mean(u**2)) / threshold)
            return u / clip_denom

        # The only difference from the optax implementation:
        # vectorized_tree_map vs. jax.tree_util.tree_map.
        updates = vectorized_tree_map(_clip_fn, updates)
        return updates, state

    return PartitionedGradientTransformation(init_fn, update_fn, lambda _: optax.EmptyState())


def scale_by_param_block_rms(min_scale: float = 1e-3) -> PartitionedGradientTransformation:
    """Scale updates by rms of the gradient for each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.
    A sub tree under a VDict will be vectorized and scaled separately so that we
    scale updates to different layers of a Repeat/Pipeline layer separately.

    Args:
        min_scale: minimum scaling factor.

    Returns:
        A PartitionedGradientTransformation.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        updates = vectorized_tree_map(
            lambda u, p: u * optax.safe_root_mean_squares(p, min_scale), updates, params
        )
        return updates, state

    return with_partition_fn(
        optax.GradientTransformation(init_fn, update_fn), lambda _: optax.EmptyState()
    )


class ParamEmaState(NamedTuple):
    """Captures an exponential moving average for all params."""

    count: Union[Tensor, TensorSpec]  # Number of times the EMA has been updated.
    ema: NestedTree  # Exponential moving averages.


def param_ema(
    *,
    decay: Optional[schedule.Schedule] = None,
) -> PartitionedGradientTransformation:
    """Computes the EMA of model params.

    Also known as "polyak averaging".

    References:
        [Polyak et al, 1991](https://epubs.siam.org/doi/10.1137/0330046)
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Args:
        decay: The EMA decay rate. If None, EMA is disabled.
            To enable bias correction, wrap the decay with schedule.decay_bias_correction().

    Returns:
        A PartitionedGradientTransformation.
    """
    if decay is None:
        return _no_op()

    decay_fn = schedule.as_schedule_fn(decay)

    def init_fn(params):
        return ParamEmaState(
            count=jnp.zeros([], jnp.int32),
            ema=jax.tree_util.tree_map(lambda p: jnp.zeros_like(p.value), params),
        )

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("params are required for param_ema.")

        decay_t = decay_fn(state.count)

        # Transform updates and compute new per-tensor EMA.
        count_inc = optax.safe_int32_increment(state.count)
        new_ema = jax.tree_util.tree_map(
            lambda param, ema: (1 - decay_t) * param.value + decay_t * ema,
            params,
            state.ema,
        )
        return updates, ParamEmaState(count=count_inc, ema=new_ema)

    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        return ParamEmaState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            ema=copy_partition(param_specs),
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


class ScaleByLionState(NamedTuple):
    """State for the Lion algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
) -> PartitionedGradientTransformation:
    """Rescale updates according to the Lion algorithm.

    Args:
        b1: Rate for combining moment and the current grad.
        b2: Decay rate for the exponentially weighted average of grads.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype is inferred from `params` and `updates`.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        mu = jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=mu_dtype or t.dtype), params
        )  # moment
        return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.update_moment(updates, state.mu, b2, 1)
        if mu_dtype is not None:
            mu = jax.tree_map(lambda x: x.astype(mu_dtype), mu)
        count_inc = optax.safe_int32_increment(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, m: jnp.sign((1.0 - b1) * g + b1 * m), updates, state.mu
        )
        return updates, ScaleByLionState(count=count_inc, mu=mu)

    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        mu_specs = param_specs
        if mu_dtype is not None:
            mu_specs = jax.tree_util.tree_map(
                lambda param_spec: dataclasses.replace(param_spec, dtype=mu_dtype),
                mu_specs,
            )
        return ScaleByLionState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            mu=copy_partition(mu_specs),
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def lion_optimizer(
    learning_rate: schedule.Schedule,
    b1: float,
    b2: float,
    mu_dtype: Optional[jnp.dtype] = None,
    weight_decay: float = 0.0,
    multiply_by_parameter_scale: bool = False,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
) -> PartitionedGradientTransformation:
    """Lion optimizer with parameter scaling.

    https://arxiv.org/abs/2302.06675
    Adapted from https://github.com/google/automl/blob/master/lion/lion_optax.py

    Args:
        learning_rate: The learning rate schedule.
        b1: The exponential decay rate for the 1st moment estimates.
        b2: The exponential decay rate for the 2nd moment estimates.
        weight_decay: Optional rate at which to decay weights.
        weight_decay_per_param_scale: A Callable that returns a tree with same structure
            as the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
        mu_dtype: Optional `dtype` to be used for the first order accumulator;
            if `None` then the dtype is inferred from params and updates.
        multiply_by_parameter_scale: If `True`, then scale learning_rate by
            parameter RMS. if `False`, provided learning_rate is absolute step size.
            Usually this should be left as False.

    Returns:
        A PartitionedGradientTransformation representing an Lion optimizer with parameter scalin.
    """
    tx = [scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype)]
    if multiply_by_parameter_scale:
        tx.append(scale_by_param_block_rms())
    tx.extend(
        [
            add_decayed_weights(
                weight_decay=weight_decay,
                # Weight decay updates will already be scaled by the learning rate below.
                learning_rate_exponent=None,
                per_param_scale=weight_decay_per_param_scale,
            ),
            scale_by_schedule(scale_from_learning_rate(learning_rate)),
        ]
    )

    return chain(*tx)
