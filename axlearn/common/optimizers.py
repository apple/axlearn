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
"""Optimizers.

For new optimizers, using `UpdateTransformation` is preferred instead.

Despite this, there are no plans to stop supporting `PartitionedGradientTransformation`.
"""

import dataclasses
import re
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import optax
import typing_extensions
from absl import logging
from jax import numpy as jnp
from jax._src.sharding_impls import TransferToMemoryKind
from optax._src import numerics

from axlearn.common import flax_struct, schedule
from axlearn.common.base_layer import ParameterSpec, PartitionSpec
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
    MemoryKind,
    Nested,
    NestedTensor,
    NestedTree,
    Tensor,
    TensorSpec,
    expand_vdicts,
    flatten_items,
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


def named_chain(**kwargs):
    kwargs = {k: _to_partitioned_transformation(v) for k, v in kwargs.items()}

    def init_fn(params):
        return {k: v.init(params) for k, v in kwargs.items()}

    def update_fn(
        updates: NestedTensor, state: dict[str, Any], params: NestedOptParam
    ) -> tuple[NestedTensor, optax.EmptyState]:
        new_state = {}
        for k, v in kwargs.items():
            updates, new_state[k] = v.update(updates, state[k], params)
        return updates, new_state

    def partition_fn(param_spec):
        return {k: v.partition(param_spec) for k, v in kwargs.items()}

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def _no_op():
    def update_fn(
        updates: NestedTensor, state: optax.EmptyState, params: NestedOptParam
    ) -> tuple[NestedTensor, optax.EmptyState]:
        del params
        return updates, state

    return PartitionedGradientTransformation(
        init=lambda params: optax.EmptyState(),
        update=update_fn,
        partition=lambda param_specs: optax.EmptyState(),
    )


def opt_param_values(params: NestedOptParam) -> NestedTensor:
    return jax.tree.map(lambda opt_param: opt_param.value, params)


def with_partition_fn(
    base: optax.GradientTransformation, partition_fn: TransformPartitionSpecFn
) -> PartitionedGradientTransformation:
    def init_fn(params: NestedOptParam) -> NestedTensor:
        return base.init(opt_param_values(params))

    def update_fn(
        updates: optax.Updates, state: optax.OptState, params: NestedOptParam
    ) -> tuple[optax.Updates, optax.OptState]:
        return base.update(updates, state, opt_param_values(params))

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def copy_partition(
    specs: Nested[OptStateSpec],
    *,
    pattern: Union[None, str, re.Pattern] = None,
    memory_kind: Optional[MemoryKind] = None,
) -> Nested[OptStateSpec]:
    """Copies OptStateSpec and optionally assigns with a different memory kind.

    Args:
        specs: Nested[OptStateSpec] to copy from.
        pattern: Regex to match the full path of each spec. Matched specs will have their memory
            kind replaced with `memory_kind`.
        memory_kind: New memory kind. Default to None.

    Returns:
        A Nested[OptStateSpec] with possibly a different memory kind.
    """
    return jax.tree.map(
        lambda path, spec: OptStateSpec(
            dtype=spec.dtype,
            shape=spec.shape,
            mesh_axes=spec.mesh_axes,
            memory_kind=(
                memory_kind if pattern and re.fullmatch(pattern, path) else spec.memory_kind
            ),
        ),
        tree_paths(specs),
        specs,
    )


def trace_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    def partition_fn(param_specs: Nested[ParameterSpec]) -> Nested[OptStateSpec]:
        return optax.TraceState(trace=copy_partition(param_specs))

    return with_partition_fn(base, partition_fn)


def adam_partition(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    state: optax.ScaleByAdamState = base.init({})

    def partition_fn(
        param_specs: Nested[ParameterSpec],
    ) -> Nested[Union[OptStateSpec, optax.ScaleByAdamState]]:
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


def scale_by_schedule(
    step_size_fn: schedule.Schedule, *, name: Optional[str] = None
) -> PartitionedGradientTransformation:
    """Scales updates using a custom schedule for the step size.

    Unlike optax.scale_by_schedule, this implementation uses 1-based steps, i.e., the first
    step will be 1 to be consistent with the step count in trainer, summaries, and checkpoints.

    Args:
        step_size_fn: A function that takes the current step as input and returns a scale factor
            to multiply the updates by.
        name: Name for this transformation (used to group logged summaries).
            If None, will not group logged summaries under a name.

    Returns:
        A partitioned gradient transformation.
    """

    schedule_fn = schedule.as_schedule_fn(step_size_fn)
    summary_name_prefix = "" if name is None else f"{name}/"

    def init_fn(params):
        del params
        return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        count_inc = optax.safe_int32_increment(state.count)
        step_size = schedule_fn(count_inc)
        context = current_context()
        if context:
            context.add_summary(summary_name_prefix + "schedule_step", count_inc)
            context.add_summary(summary_name_prefix + "schedule_scale", step_size)
        updates = jax.tree.map(lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates)
        return updates, optax.ScaleByScheduleState(count=count_inc)

    return PartitionedGradientTransformation(
        init=init_fn,
        update=update_fn,
        partition=lambda _: optax.ScaleByScheduleState(
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
    *, scale_by_path: Sequence[tuple[str, float]], description: str, default_scale: float = 1.0
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

        return jax.tree.map(per_param_fn, tree_paths(params, separator="/"))

    return fn


def per_param_scale_by_rms(*, min_scale: float = 1e-4) -> Callable[[NestedOptParam], NestedTree]:
    """Computes per-parameter scales with its Root-Mean-Square (RMS).

    Args:
        min_scale: The minimum scale for each parameter.

    Returns:
        A function that computes per-parameter scales given a NestedOptParam tree. The returned
        scales will be a tree with the same structure, but with float numbers instead of OptParam
        as leaves.
    """

    def fn(params: NestedOptParam) -> Any:
        return jax.tree.map(lambda p: optax.safe_root_mean_squares(p.value, min_scale), params)

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
        # vectorized_tree_map vs. jax.tree.map.
        updates = vectorized_tree_map(_scale_update, updates, params)
        return updates, state

    return with_partition_fn(
        optax.GradientTransformation(init_fn, update_fn), lambda _: optax.ScaleByTrustRatioState()
    )


def _log_per_layer_stats(stats: NestedTensor, *, summary_suffix: str):
    """Expand the Nested Tensor `stats` and add summaries.

    Args:
        stats: A Nested Tensor, e.g., containing param norms or gradient statistics.
        summary_suffix: Adds summaries of name `{path}/{summary_suffix}`.
    """
    context = current_context()
    if context is not None:
        expanded_stats = expand_vdicts(stats)
        for path, value in flatten_items(expanded_stats):
            context.add_summary(f"{path}/{summary_suffix}", value)


def _compute_rms_norms(x: NestedTensor, *, summary_suffix: Optional[str] = None) -> NestedTensor:
    """Computes the RMS norm for each leaf tensor of `x` and optionally adds summaries.

    Summaries will be added if `summary_suffix` is not None *and* the current context is not None.

    Args:
        x: A Nested Tensor, e.g., representing params or gradients. May contain VDict, in which
            case each entry will be computed separately, therefore the norms of params of a
            repeated layer will be computed separately.
        summary_suffix: If not None, adds summaries of name `{path}/{summary_suffix}` of the norms.

    Returns:
        A NestedTensor with the same structure as `x` and each leaf node representing the norm
        of the tensor in `x`.
    """
    # Use vectorized_tree_map to compute separate norms for each layer in a Repeated.
    norms = vectorized_tree_map(lambda u: jnp.sqrt(jnp.mean(u**2)), x)
    if summary_suffix is not None:
        _log_per_layer_stats(norms, summary_suffix=summary_suffix)
    return norms


def _compute_covariance(
    x: NestedTensor,
    y: NestedTensor,
    *,
    summary_suffix: Optional[str] = None,
) -> NestedTensor:
    """Computes the covariance between leaf tensors in `x` and `y` and optionally adds summaries.

    Summaries will be added if `summary_suffix` is not None *and* the current context is not None.
    This function is used in adastar_optimizer() for adding (params, updates) correlation stats.

    Args:
        x: A Nested Tensor, e.g., representing params or gradients. May contain VDict, in which
            case each entry will be computed separately, therefore the norms of params of a
            repeated layer will be computed separately.
        y: A Nested Tensor similar to `x`.
        summary_suffix: If not None, adds summaries of name `{path}/{summary_suffix}` of the norms.

    Returns:
        A NestedTensor with the same structure as `x` and each leaf node representing the
        covariance between the leaf nodes in `x` and `y`.
    """
    # Use vectorized_tree_map to compute separate values for each layer in a Repeated.
    cov = vectorized_tree_map(lambda u, v: jnp.mean(u * v), x, y)
    if summary_suffix is not None:
        _log_per_layer_stats(cov, summary_suffix=summary_suffix)
    return cov


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
    ) -> tuple[NestedTensor, optax.EmptyState]:
        if params is None:
            raise ValueError(optax.NO_PARAMS_MSG)  # pylint: disable=no-member

        param_scales = maybe_instantiate(per_param_scale)(params)
        context = current_context()
        register_per_param_settings(
            param_scales,
            description=getattr(per_param_scale, "description", "update_scale"),
            path=context.path() if context else None,
        )

        updates = jax.tree.map(
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
        param_scales = jax.tree.map(lambda _: 1, params)
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

    scales = jax.tree.map(maybe_override_scale, tree_paths(params), params, param_scales)
    context = current_context()
    return register_per_param_settings(
        scales,
        description=getattr(per_param_scale, "description", "weight_decay_scale"),
        path=context.path() if context else None,
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

        if learning_rate_exponent is None:
            lr_scale = 1.0
            updated_state = state
        else:
            learning_rate_fn = schedule.as_schedule_fn(learning_rate)
            count_inc = optax.safe_int32_increment(state.count)
            lr = learning_rate_fn(count_inc)
            lr_scale = lr**learning_rate_exponent
            updated_state = AddDecayedWeightsState(count_inc)

        param_scales = _weight_decay_scales(params, per_param_scale=per_param_scale)
        f = lambda g, p, s: g + weight_decay * lr_scale * p.value * s
        updates = jax.tree.map(
            lambda x, y, z: None if x is None else f(x, y, z),
            updates,
            params,
            param_scales,
            is_leaf=lambda x: x is None,
        )
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
            updates = jax.tree.map(
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
    adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None,
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
        adam_update_transformation: A transformation applied directly on the adam updates
            (but before weight decay). If None, no transformation is applied.

    Returns:
        A PartitionedGradientTransformation representing an AdamW optimizer with parameter scaling.
    """
    tx = [adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype))]
    if adam_update_transformation is not None:
        tx.append(maybe_instantiate(adam_update_transformation))
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
    adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None,
) -> PartitionedGradientTransformation:
    """A "decoupled" version of the AdamW optimizer, with optional parameter scaling.

    Faithfully replicates Adam with "decoupled" weight decay from
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
        adam_update_transformation: A transformation applied directly on the adam updates
            (but before weight decay). If None, no transformation is applied.

    Returns:
        A PartitionedGradientTransformation representing a decoupled AdamW optimizer with
            parameter scaling.
    """
    tx = [adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype))]
    if adam_update_transformation is not None:
        tx.append(maybe_instantiate(adam_update_transformation))
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
            ema=jax.tree.map(lambda ema: ema.value, ema_tree),
            scale=jax.tree.map(lambda ema: ema.qstep_size, ema_tree),
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

        return _to_state(jnp.zeros([], jnp.int32), jax.tree.map(_init, params))

    @dataclasses.dataclass
    class _UpdateResult:
        # Captures an update and associated EMA value.
        update: Tensor  # The transformed update value.
        tensor_ema: _TensorEma  # The transformed tensor EMA.

    def update_fn(updates, state, params=None):
        del params
        count_inc = optax.safe_int32_increment(state.count)
        decay_t = decay_fn(count_inc)

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
        def _update(value: Tensor, ema: Tensor, qstep_size: Tensor) -> _UpdateResult:
            update = new_ema = (1 - decay_t) * value + decay_t * _to_float(ema, qstep_size)
            if debias:
                bias_correction = 1 - decay_t**count_inc
                update = new_ema / bias_correction.astype(new_ema.dtype)
            return _UpdateResult(update=update, tensor_ema=_to_tensor_ema(new_ema))

        # Transform updates and compute new per-tensor EMA.
        update_results = jax.tree.map(
            lambda update, ema, scale: _update(update, ema=ema, qstep_size=scale),
            updates,
            state.ema,
            state.scale,
        )

        # Unpack update, and pack state into EmaState.
        updates = jax.tree.map(lambda ur: ur.update, update_results)
        new_state = _to_state(
            count=count_inc,
            ema_tree=jax.tree.map(lambda ur: ur.tensor_ema, update_results),
        )
        return updates, new_state

    def partition_fn(param_specs: Nested[ParameterSpec]) -> Nested[Union[OptStateSpec, EmaState]]:
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
            ema=jax.tree.map(get_ema_partition, param_specs),
            scale=jax.tree.map(get_scale_partition, param_specs),
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
        tx.append(clip_by_block_rms(clipping_threshold, summary_suffix="norm"))
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
            updates = jax.tree.map(lambda t: t * g_scale, updates)
        return updates, state

    return PartitionedGradientTransformation(
        init=init_fn, update=update_fn, partition=lambda _: optax.EmptyState()
    )


class DropNormThresholdFn(typing_extensions.Protocol):
    """Protocol for drop norm threshold function."""

    def __call__(self, *, count: Tensor, mean: Tensor, stddev: Tensor) -> dict[str, Tensor]:
        """Returns the drop_norm thresholds given the gradient norm stats.

        Args:
            count: the number of previous updates to mean/stddev.
            mean: the running average of gradient norms.
            stddev: the running average of gradient norm variance.

        Returns:
            A dict where keys represent threshold names and values are scalar tensors representing
            threshold values. The names are used only for drop norm summaries. Gradients will be
            dropped if the norm exceeds any of the thresholds.
        """


def drop_norm_by_grad_norm_ema(multipliers: tuple = (20, 40, 100)) -> DropNormThresholdFn:
    """Return drop norm thresholds which are multiples of grad norm ema."""

    def fn(count: Tensor, mean: Tensor, stddev: Tensor) -> dict[str, Tensor]:
        del count
        del stddev
        thresholds = {}
        for v in multipliers:
            key = f"{v}x_mean"
            thresholds[key] = mean * v
        return thresholds

    return fn


def drop_norm_by_grad_norm_stddev(
    *,
    min_count: int = 500,
    multipliers: tuple = (20, 40, 100),
) -> DropNormThresholdFn:
    """Return drop norm thresholds based on grad norm stddev."""

    def fn(count: Tensor, mean: Tensor, stddev: Tensor) -> dict[str, Tensor]:
        # We do not drop norm until we have collected stats for at least `min_count` steps,
        # otherwise the threshold is `mean + stddev * k` for multiplier `k`.
        thresholds = {}
        for v in multipliers:
            key = f"{v}x_stddev"
            thresholds[key] = jnp.where(count < min_count, 1e10, mean + stddev * v)
        return thresholds

    return fn


class SkipClipState(NamedTuple):
    """State returned by functions in skip_and_clip_by_global_norm()."""

    count: Optional[Union[Tensor, TensorSpec]]
    nonvalid_count: Union[Tensor, TensorSpec]  # Number of non-valid steps.
    grad_norm_ema: Optional[Union[Tensor, TensorSpec]]  # The moving average of raw gradient norm.
    grad_norm_square_ema: Optional[
        Union[Tensor, TensorSpec]
    ]  # The moving average of grad norm variance.
    inner_state: Any  # State of the inner PartitionedGradientTransformation.
    drop_stats: Optional[
        dict[str, Union[Tensor, TensorSpec]]
    ]  # A dict to keep the counts when the grad norm exceeds thresholds.


def skip_and_clip_by_global_norm(
    inner: ConfigOr[PartitionedGradientTransformation],
    *,
    drop_norm: Optional[Union[float, ConfigOr[DropNormThresholdFn]]] = None,
    max_norm: Optional[float] = None,
    grad_norm_ema_decay: Optional[float] = None,
    eps: float = 1e-8,
) -> PartitionedGradientTransformation:
    """Skip updates when global norm >= drop_norm, otherwise clip the global norm.
    If we detect abnormal gradients that have global norm >= drop_norm, we skip the gradient updates
    and state updates. Otherwise we scale the gradients s.t. global norm <= max_norm, and apply the
    wrapped gradient transformation `inner`. Note the difference compared to clip_by_global_norm()
    is that this version skips all updates while clip_by_global_norm() still performs parameter
    updates and optimizer state updates.

    When drop_norm is a DropNormThresholdFn, the drop norm will be calculated based on the moving
    stats of recent gradient norms. This is useful since the gradient norms can initially be large
    but reduce to a small value during training.

    Example usage:
        ```
        config_for_function(skip_and_clip_by_global_norm).set(
            inner=config_for_function(optimizers.adamw_optimizer).set(
                learning_rate=learning_rate_schedule,
                b1=0.95,
                b2=0.995,
                eps=1e-8,
                weight_decay=0.05,
                weight_decay_per_param_scale=None,
                multiply_by_parameter_scale=False,
            ),
            drop_norm=100,
            max_norm=1,
        )
        ```

    Args:
        inner: the PartitionedGradientTransformation we wrapped over, e.g. adamw_optimizer().
        drop_norm: the threshold to detect abnormal gradients and skip gradient and state updates.
            When this is a DropNormThresholdFn, the actual drop norm will be calculated dynamically
            based on recent gradient stats.
        max_norm: the maximum global gradient norm. If this is set, larger gradients will be scaled
            and clipped.
        gradient_norm_ema_decay: the decay factor used to compute EMA of gradient norms. This must
            be set when `drop_norm` is a DropNormThresholdFn.
        eps: a small constant added to scaling factor, i.e. `1/(norm + eps)`.

    Returns:
        A new PartitionedGradientTransformation that applies skipping and clipping.
    """
    inner = maybe_instantiate(inner)
    use_adaptive_drop_norm = drop_norm is not None and not isinstance(drop_norm, (float, int))
    if use_adaptive_drop_norm:
        drop_norm = maybe_instantiate(drop_norm)
    if use_adaptive_drop_norm and grad_norm_ema_decay is None:
        raise ValueError("grad_norm_ema_decay must be set (e.g. 0.99).")

    def init_fn(params):
        if use_adaptive_drop_norm:
            one = jnp.ones([], jnp.float32)
            dict_thresholds = drop_norm(count=one, mean=one, stddev=one)
            drop_stats = {k: jnp.zeros([], jnp.int32) for k in dict_thresholds}
            return SkipClipState(
                count=jnp.zeros([], jnp.int32),
                nonvalid_count=jnp.zeros([], jnp.int32),
                # Set initial ema(s) to a positive value so we can avoid dropping norms for the
                # first step.
                grad_norm_ema=jnp.ones([], jnp.float32),
                grad_norm_square_ema=jnp.ones([], jnp.float32),
                inner_state=inner.init(params),
                drop_stats=drop_stats,
            )
        else:
            # Backward compatible when drop_norm is float or is not set.
            return SkipClipState(
                count=None,
                nonvalid_count=jnp.zeros([], jnp.int32),
                grad_norm_ema=None,
                grad_norm_square_ema=None,
                inner_state=inner.init(params),
                drop_stats=None,
            )

    def update_fn(updates, state, params=None):
        inner_state = state.inner_state
        grad_norm_ema = state.grad_norm_ema
        grad_norm_square_ema = state.grad_norm_square_ema
        drop_stats = state.drop_stats

        def _stddev(mean: Tensor, mean_square: Tensor):
            return (mean_square - mean**2) ** 0.5

        def _moment(
            val: Tensor,
            norm_ema: Tensor,
            norm_square_ema: Tensor,
            count: Tensor,
        ) -> tuple[Tensor, Tensor]:
            # bias correrction decay
            # Sec 7.1 https://arxiv.org/pdf/1804.04235.pdf
            decay = grad_norm_ema_decay
            decay *= (1 - decay**count) / (1 - decay ** (optax.safe_int32_increment(count)))
            new_norm_ema = decay * norm_ema + (1 - decay) * val
            new_square_ema = decay * norm_square_ema + (1 - decay) * (val**2)
            return new_norm_ema, new_square_ema

        def _is_valid_step(
            g_norm: Tensor,
            drop_norm: Union[float, DropNormThresholdFn],
            *,
            norm_ema: Optional[Tensor],
            norm_square_ema: Optional[Tensor],
            count: Optional[Tensor],
            drop_stats: Optional[dict[str, Tensor]],
        ) -> tuple[Tensor, Optional[dict[str, Tensor]]]:
            if isinstance(drop_norm, (float, int)):
                return g_norm < drop_norm, None
            else:
                stddev = _stddev(norm_ema, norm_square_ema)
                thresholds = drop_norm(count=count, mean=norm_ema, stddev=stddev)
                new_drop_stats = {}
                is_valid = None
                for key, val in thresholds.items():
                    less = g_norm < val
                    is_valid = less if is_valid is None else jnp.logical_and(is_valid, less)
                    new_drop_stats[key] = jnp.where(
                        less,
                        drop_stats[key],
                        optax.safe_int32_increment(drop_stats[key]),
                    )
                return is_valid, new_drop_stats

        # Check if every gradient is finite.
        flat_updates = jax.tree_util.tree_flatten(updates)[0]
        is_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates]))
        g_norm = optax.global_norm(updates)
        if drop_norm is not None:
            # Check if gradient norm is abnormal.
            is_valid_step, new_drop_stats = _is_valid_step(
                g_norm,
                drop_norm,
                norm_ema=grad_norm_ema,
                norm_square_ema=grad_norm_square_ema,
                # Note that `count` for `drop_norm` represents the number of updates to
                # mean/stddev, so we pass `state.count` rather than `count_inc`.
                count=state.count,
                drop_stats=drop_stats,
            )
            is_valid_step = jnp.logical_and(is_finite, is_valid_step)
        else:
            is_valid_step = is_finite
            new_drop_stats = None

        # Log useful statistics.
        nonvalid_count = jnp.where(
            is_valid_step,
            state.nonvalid_count,
            optax.safe_int32_increment(state.nonvalid_count),
        )
        if use_adaptive_drop_norm:
            count_inc = jnp.where(
                is_valid_step,
                optax.safe_int32_increment(state.count),
                state.count,
            )
            new_norm_ema, new_norm_square_ema = _moment(
                g_norm, grad_norm_ema, grad_norm_square_ema, state.count
            )
            new_norm_ema = jnp.where(is_valid_step, new_norm_ema, grad_norm_ema)
            new_norm_square_ema = jnp.where(
                is_valid_step, new_norm_square_ema, grad_norm_square_ema
            )
        else:
            count_inc = None
            new_norm_ema = None
            new_norm_square_ema = None
        context = current_context()
        if context is not None:
            context.add_summary("gradient_norm", g_norm)
            context.add_summary("nonvalid_count", nonvalid_count)
            if new_norm_ema is not None:
                context.add_summary("gradient_norm_ema", new_norm_ema)
            if new_norm_square_ema is not None:
                context.add_summary(
                    "gradient_norm_std_ema", _stddev(new_norm_ema, new_norm_square_ema)
                )
            if count_inc is not None:
                context.add_summary("count", count_inc)
            if new_drop_stats is not None:
                for key, val in new_drop_stats.items():
                    context.add_summary(f"count_exceeds_{key}", val)

        # Clip gradients s.t. grad norm <= max_norm.
        clipped_updates = updates
        if max_norm is not None:
            g_scale = jnp.minimum(1.0, max_norm / (g_norm + eps))
            clipped_updates = jax.tree.map(lambda t: t * g_scale, updates)
            if context is not None:
                context.add_summary("gradient_scale", g_scale)
        # Apply subsequent gradient transformation.
        new_updates, new_inner_state = inner.update(clipped_updates, inner_state, params)
        # Discard the updates and states in a nonvalid step.
        final_updates = jax.tree.map(
            lambda x, y: jnp.where(is_valid_step, x, jnp.zeros_like(y)),
            new_updates,
            updates,
        )
        final_inner_state = jax.tree.map(
            lambda x, y: jnp.where(is_valid_step, x, y),
            new_inner_state,
            inner_state,
        )

        return final_updates, SkipClipState(
            count=count_inc,
            nonvalid_count=nonvalid_count,
            grad_norm_ema=new_norm_ema,
            grad_norm_square_ema=new_norm_square_ema,
            inner_state=final_inner_state,
            drop_stats=new_drop_stats,
        )

    def partition_fn(
        param_specs: Nested[ParameterSpec],
    ) -> Nested[Union[OptStateSpec, SkipClipState]]:
        if use_adaptive_drop_norm:
            one = jnp.ones([], jnp.float32)
            dict_thresholds = drop_norm(count=one, mean=one, stddev=one)
            drop_stats = {
                k: OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec())
                for k in dict_thresholds
            }
            return SkipClipState(
                count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                nonvalid_count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                grad_norm_ema=OptStateSpec(dtype=jnp.float32, shape=[], mesh_axes=PartitionSpec()),
                grad_norm_square_ema=OptStateSpec(
                    dtype=jnp.float32, shape=[], mesh_axes=PartitionSpec()
                ),
                inner_state=inner.partition(param_specs),
                drop_stats=drop_stats,
            )
        else:
            return SkipClipState(
                count=None,
                nonvalid_count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                grad_norm_ema=None,
                grad_norm_square_ema=None,
                inner_state=inner.partition(param_specs),
                drop_stats=None,
            )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def clip_by_block_rms(
    threshold: Optional[float], *, summary_suffix: Optional[str] = None
) -> PartitionedGradientTransformation:
    """Clip updates to a max rms for the gradient of each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.
    A sub tree under a VDict will be vectorized and clipped separately so that we
    clip updates to different layers of a Repeat/Pipeline layer separately.

    Args:
        threshold: the maximum rms for the gradient of each param vector or matrix.
            If None, does not clip.
        summary_suffix: If not None, adds pre-clip update norms to summaries.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params

        if threshold is None and summary_suffix is None:
            # Do not compute norm.
            return updates, state

        def _clip_fn(u, norm):
            if threshold is None:
                clipped = u
            else:
                clipped = u / jnp.maximum(1.0, norm / threshold)
            return clipped

        norms = _compute_rms_norms(updates, summary_suffix=summary_suffix)
        # The only difference from the optax implementation:
        # vectorized_tree_map vs. jax.tree.map.
        updates = vectorized_tree_map(_clip_fn, updates, norms)
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

    Non floating point params will be assigned with current values, instead of being interpolated
    with EMA.

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
            ema=jax.tree.map(lambda p: jnp.zeros_like(p.value), params),
        )

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("params are required for param_ema.")

        count_inc = optax.safe_int32_increment(state.count)
        decay_t = decay_fn(count_inc)

        def _interpolate(param, ema_value):
            x = param.value
            if not jnp.issubdtype(x.dtype, jnp.floating):
                # For example, int32 for step counters.
                return x

            return (1 - decay_t) * x + decay_t * ema_value

        # Transform updates and compute new per-tensor EMA.
        new_ema = jax.tree.map(_interpolate, params, state.ema)
        return updates, ParamEmaState(count=count_inc, ema=new_ema)

    def partition_fn(
        param_specs: Nested[ParameterSpec],
    ) -> Nested[Union[OptStateSpec, ParamEmaState]]:
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
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, dtype=mu_dtype or t.dtype), params)  # moment
        return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.update_moment(updates, state.mu, b2, 1)
        if mu_dtype is not None:
            mu = jax.tree.map(lambda x: x.astype(mu_dtype), mu)
        count_inc = optax.safe_int32_increment(state.count)
        updates = jax.tree.map(lambda g, m: jnp.sign((1.0 - b1) * g + b1 * m), updates, state.mu)
        return updates, ScaleByLionState(count=count_inc, mu=mu)

    def partition_fn(
        param_specs: Nested[ParameterSpec],
    ) -> Nested[Union[OptStateSpec, ScaleByLionState]]:
        mu_specs = param_specs
        if mu_dtype is not None:
            mu_specs = jax.tree.map(
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


def adastar_optimizer(
    learning_rate: float,
    *,
    gradient_ema_decay: Optional[float],
    gradient_ema_debias: bool,
    gradient_square_ema_decay: float,
    gradient_square_ema_debias: bool,
    eps: float,
    eps_square: float,
    raw_update_clipping_threshold: Optional[float],
    update_ema_decay: Optional[float],
    update_ema_debias: bool,
    adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None,
    weight_decay: float = 0,
    weight_decay_per_param_scale: Optional[Callable[[NestedOptParam], Any]] = None,
    update_schedule: schedule.Schedule,
    verbosity: int = 0,
) -> PartitionedGradientTransformation:
    """An optimizer covering both {adamw_decoupled,adafactor}_optimizer (with factored=False).

    The generalized algorithm is:

        # Stage 1.
        smoothed_gradients = ema(gradients, gradient_ema_decay, gradient_ema_debias)
        smoothed_gradient_squares = ema(
            gradients ** 2 + eps_square, gradient_square_ema_decay, gradient_square_ema_debias)
        # Normalized gradients.
        raw_updates = smoothed_gradients / ((smoothed_gradient_squares) ** 0.5 + eps)
        clipped_updates = clip(raw_updates, raw_update_clipping_threshold)
        smoothed_updates = ema(clipped_updates, update_ema_decay, update_ema_debias)

        # Apply per-param transformation.
        transformed_updates = adam_update_transformation(smoothed_updates)

        # Stage 2.
        lr_scaled_updates = learning_rate * transformed_updates
        updates_with_wd = add_weight_decay(lr_scaled_updates)
        final_updates = - update_schedule * updates_with_wd

    Notable special cases of adastar:

        adamw_decoupled(b1, b2, eps) can represented by adastar(
            gradient_ema_decay=b1,
            gradient_ema_debias=True,
            gradient_square_ema_decay=b2,
            gradient_square_ema_debias=True,
            eps=eps,
            eps_square=0,
            update_ema_decay=None,  # disabled.
        )

        adafactor(b1, decay_bias_correction(b2), eps) can represented by adastar(
            gradient_ema_decay=None,  # disabled.
            gradient_square_ema_decay=b2,
            gradient_square_ema_debias=True,
            eps=0,
            eps_square=eps,  # adafactor eps is applied on the square.
            update_ema_decay=b1,
            update_ema_debias=False,
        )

    Usually only one of gradient_ema_* and update_ema_* is enabled, as each of them uses memory
    of the same size as the parameters.

    Args:
        learning_rate: the learning rate (will be scaled by the update_schedule).
        gradient_ema_decay: If not None, applies momentum on gradients to compute smoothed
            gradients.
        gradient_ema_debias: Whether to apply bias correction when computing smoothed gradients.
        gradient_square_ema_decay: The ema decay for the second order momentum of gradients.
        gradient_square_ema_debias: Whether to apply bias correction when computing
            gradient square ema.
        eps: (float) regularization constant added to the square root of smoothed_gradient_squares.
        eps_square: (float) regularization constant added to gradient_squares.
        raw_update_clipping_threshold: If not None, clips the norms of the raw updates
            to this value. `raw_update_norm` summaries will be logged either way.
        update_ema_decay: If not None, applies momentum on raw updates (normalized gradients) to
            compute smoothed updates.
        update_ema_debias: Whether to apply bias correction when computing smoothed updates.
        adam_update_transformation: An optional transformation applied on the smoothed updates
            (but before applying learning rate and weight decay).
            If None, no transformation is applied.
        weight_decay: (float) optional rate at which to decay weights. Note that weight_decay
            is decoupled from `learning_rate` but is subject to `update_schedule`. This is
            similar to adamw_adamw_decoupled_optimizer and different from adafactor_optimizer.
        weight_decay_per_param_scale: (optional) a Callable that returns a tree with same structure
            as the params PyTree, where each leaf is a float representing the per-param decay scale.
            The scale will be applied on top of the global decay rate:
            effective_decay_rate = global_decay_rate * per_param_scale.
            If None, all leaves will have a scale of 1.
        update_schedule: an update schedule, which is applied to scale both the learning rate
            and the weight decay.
        verbosity: The verbosity level of summaries. When verbosity > 0, adds update norms and
            param-update correlation stats to summaries.

    Returns:
        A PartitionedGradientTransformation representing an Adafactor optimizer.
    """

    class _AdastarPerParamState(flax_struct.PyTreeNode):
        gradient_ema: Optional[Tensor]
        gradient_square_ema: Tensor
        update_ema: Optional[Tensor]

    class _AdastarState(flax_struct.PyTreeNode):
        count: Tensor
        pps: Nested[_AdastarPerParamState]

    class _AdastarUpdateResult(flax_struct.PyTreeNode):
        """Opaque container that is not traversed by jax.tree.map."""

        updates: Tensor  # the update to apply to params.
        pps: _AdastarPerParamState

    update_schedule = schedule.as_schedule_fn(update_schedule)

    def init_fn(params: NestedOptParam):
        """Initializes the stage 1 state."""

        def _init(param: OptParam):
            v = param.value
            return _AdastarPerParamState(
                gradient_ema=None if gradient_ema_decay is None else jnp.zeros_like(v),
                gradient_square_ema=jnp.zeros_like(v),
                update_ema=None if update_ema_decay is None else jnp.zeros_like(v),
            )

        return _AdastarState(count=jnp.zeros([], jnp.int32), pps=jax.tree.map(_init, params))

    def update_fn(grads: NestedTensor, state: _AdastarState, params: NestedOptParam):
        """Applies (stage 1) gradient transformation to compute raw_updates."""
        count_inc = optax.safe_int32_increment(state.count)

        if params is None:
            raise ValueError("param is None")

        def _moment(
            x: Tensor, *, acc: Optional[Tensor], decay: Optional[float], debias: bool
        ) -> tuple[Tensor, Optional[Tensor]]:
            if decay is None:
                return x, None
            value = acc = decay * acc + (1 - decay) * x
            if debias:
                value = optax.bias_correction(acc, decay=decay, count=count_inc)
            return value, acc

        def _split_update_results(
            update_results: Nested[_AdastarUpdateResult],
        ) -> tuple[NestedTensor, Nested[_AdastarPerParamState]]:
            """Splits a tree of _AdastarUpdateResult to (updates, state)."""
            updates = jax.tree.map(
                lambda ur: ur.updates,
                update_results,
                is_leaf=lambda x: isinstance(x, _AdastarUpdateResult),
            )
            pps_tree = jax.tree.map(
                lambda ur: ur.pps,
                update_results,
                is_leaf=lambda x: isinstance(x, _AdastarUpdateResult),
            )
            return updates, pps_tree

        def _raw_updates(grad: Tensor, pps: _AdastarPerParamState) -> _AdastarUpdateResult:
            """Computes raw updates from gradients."""
            smoothed_gradient, gradient_ema = _moment(
                grad,
                acc=pps.gradient_ema,
                decay=gradient_ema_decay,
                debias=gradient_ema_debias,
            )
            smoothed_gradient_square, gradient_square_ema = _moment(
                grad**2 + eps_square,
                acc=pps.gradient_square_ema,
                decay=gradient_square_ema_decay,
                debias=gradient_square_ema_debias,
            )
            raw_updates = smoothed_gradient / ((smoothed_gradient_square) ** 0.5 + eps)
            if logging.vlog_is_on(3):
                jax.debug.print("adastar mu={mu} nu={nu}", mu=gradient_ema, nu=gradient_square_ema)
                jax.debug.print("adastar raw_updates={u}", u=raw_updates)
            new_pps = _AdastarPerParamState(
                gradient_ema=gradient_ema,
                gradient_square_ema=gradient_square_ema,
                update_ema=pps.update_ema,
            )
            return _AdastarUpdateResult(updates=raw_updates, pps=new_pps)

        def _smoothed_updates(
            raw_updates: Tensor, pps: _AdastarPerParamState
        ) -> _AdastarUpdateResult:
            """Computes smoothed updates from raw updates."""
            smoothed_updates, update_ema = _moment(
                raw_updates,
                acc=pps.update_ema,
                decay=update_ema_decay,
                debias=update_ema_debias,
            )
            new_pps = _AdastarPerParamState(
                gradient_ema=pps.gradient_ema,
                gradient_square_ema=pps.gradient_square_ema,
                update_ema=update_ema,
            )
            return _AdastarUpdateResult(updates=smoothed_updates, pps=new_pps)

        # First compute raw updates.
        raw_updates, pps_tree = _split_update_results(
            jax.tree.map(
                lambda g, s: None if g is None else _raw_updates(grad=g, pps=s),
                grads,
                state.pps,
                is_leaf=lambda x: x is None,
            )
        )
        # Clip raw updates if necessary.
        clip_fn = clip_by_block_rms(
            raw_update_clipping_threshold,
            summary_suffix="raw_update_norm" if verbosity > 0 else None,
        ).update
        raw_updates, _ = clip_fn(raw_updates, None, params)
        # Compute smoothed updates.
        smoothed_updates, pps_tree = _split_update_results(
            jax.tree.map(
                lambda g, s: _smoothed_updates(raw_updates=g, pps=s),
                raw_updates,
                pps_tree,
            )
        )
        # Computing extra stats increases step time. Only adds them to summaries in verbose mode.
        if verbosity > 0:
            # Add param and update stats to summaries.
            _compute_rms_norms(grads, summary_suffix="raw_grad_norm")
            param_values = jax.tree.map(lambda p: p.value, params)
            param_norm = _compute_rms_norms(param_values, summary_suffix="param_norm")
            # Note the covariance and correlation stats might be biased if params and updates do not
            # have zero mean.
            raw_update_norm = _compute_rms_norms(raw_updates)
            smoothed_update_norm = _compute_rms_norms(
                smoothed_updates,
                summary_suffix="smoothed_update_norm",
            )
            _log_per_layer_stats(
                vectorized_tree_map(
                    lambda cov, pn, un: cov / pn / un,
                    _compute_covariance(param_values, raw_updates),
                    param_norm,
                    raw_update_norm,
                ),
                summary_suffix="corr_param_raw_updates",
            )
            _log_per_layer_stats(
                vectorized_tree_map(
                    lambda cov, pn, un: cov / pn / un,
                    _compute_covariance(param_values, smoothed_updates),
                    param_norm,
                    smoothed_update_norm,
                ),
                summary_suffix="corr_param_smoothed_updates",
            )
        return smoothed_updates, _AdastarState(count=count_inc, pps=pps_tree)

    def partition_fn(param_specs):
        def _partition(param_spec: ParameterSpec):
            opt_state_spec = OptStateSpec(
                dtype=param_spec.dtype,
                shape=param_spec.shape,
                mesh_axes=param_spec.mesh_axes,
            )
            return _AdastarPerParamState(
                gradient_ema=None if gradient_ema_decay is None else opt_state_spec,
                gradient_square_ema=opt_state_spec,
                update_ema=None if update_ema_decay is None else opt_state_spec,
            )

        return _AdastarState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            pps=jax.tree.map(_partition, param_specs),
        )

    def update2_fn(updates, state: Tensor, params: NestedOptParam):
        step_inc = optax.safe_int32_increment(state)

        def _update2(u: Tensor, param: OptParam, weight_decay_scale: float = 1.0):
            lr_scaled_updates = learning_rate * u
            updates_with_wd = lr_scaled_updates + weight_decay * param.value * weight_decay_scale
            schedule_scale = update_schedule(step_inc)
            context = current_context()
            if context:
                context.add_summary("schedule_step", step_inc)
                context.add_summary("schedule_scale", schedule_scale)
                context.add_summary("learning_rate", learning_rate * schedule_scale)
                context.add_summary(
                    "weight_decay_rate", weight_decay * schedule_scale * weight_decay_scale
                )
            return -schedule_scale * updates_with_wd

        if weight_decay_per_param_scale is not None:
            weight_decay_scales = _weight_decay_scales(
                params, per_param_scale=weight_decay_per_param_scale
            )
            updates2 = jax.tree.map(
                lambda u, p, wds: None
                if u is None
                else _update2(u, param=p, weight_decay_scale=wds),
                updates,
                params,
                weight_decay_scales,
                is_leaf=lambda x: x is None,
            )
        else:
            updates2 = jax.tree.map(
                lambda u, p: None if u is None else _update2(u, param=p),
                updates,
                params,
                is_leaf=lambda x: x is None,
            )
        return updates2, step_inc

    # Stage 1.
    tx = {
        "compute_updates": PartitionedGradientTransformation(
            init=init_fn, update=update_fn, partition=partition_fn
        )
    }
    # Interlude.
    if adam_update_transformation is not None:
        tx["transform_updates"] = adam_update_transformation
    # Stage 2.
    tx["apply_lr_and_wd"] = PartitionedGradientTransformation(
        init=lambda _: jnp.zeros([], dtype=jnp.int32),
        update=update2_fn,
        partition=lambda _: OptStateSpec(shape=[], dtype=jnp.int32, mesh_axes=PartitionSpec()),
    )
    return named_chain(**tx)


def offload_optimizer(
    optimizer: ConfigOr[PartitionedGradientTransformation],
    *,
    pattern: Union[str, re.Pattern] = ".*",
    offload_src: MemoryKind = "device",
    offload_dst: MemoryKind = "pinned_host",
) -> PartitionedGradientTransformation:
    """Offload the state of the wrapped optimizer that matches `pattern` to `offload_dst`.

    Args:
        optimizer: The optimizer to offload.
        pattern: Regex pattern used to match the path of optimizer states. Fully matched states
            will be offloaded. Default to regex that matches all states.
        offload_src: Offload-from memory kind. Default to "device".
        offload_dst: Offload-to memory kind. Default to "pinned_host".

    Returns:
        A optimizer whose state is on `offload_dst` and does the same computation as `optimizer`.

    Raises:
        ValueError: when the `update` function of the returned optimizer is called outside of jit
            context.

    This function returns a new `PartitionedGradientTransformation` that
    1. Puts matched states of the wrapped optimizer on `offload_dst` through the partition function
       during state initialization in the trainer.
    2. Copies the matched states to `offload_src` before `optimizer.update` is called.
    3. Copies the matched updated states to `offload_dst` after `optimizer.update` is called.

    The regex pattern is matched against the full path of each optimizer state. An example full
    path is optimizer/1/0/mu/decoder/transformer/repeat/layer/feed_forward/linear1_0. If the
    pattern should not depend on model structure, you can use ".*/mu/.*" to offload all `mu`.

    The .update function of the returned `PartitionedGradientTransformation` must be called within
    a jit function.

    Example usage:
    ```python
    your_opt = adamw_optimizer(...)
    offloaded_opt = offload_optimizer(your_opt)
    ```

    When using `skip_and_clip_by_global_norm` with this offload optimizer, you must wrap the entire
    `skip_and_clip_by_global_norm` inside. Do not wrap the inner of `skip_and_clip_by_global_norm`
    or you will get errors. Correct example:
    ```
    offloaded_opt = offload_optimizer(skip_and_clip_by_global_norm(inner=adamw_optimizer(...)))
    ```
    The reason is that `skip_and_clip_by_global_norm` conditionally chooses the previous optimizer
    state and the updated new optimizer state using `jnp.where`, which doesn't support tensors on
    `pinned_host` memory space.
    """
    optimizer = maybe_instantiate(optimizer)
    if offload_src is None or offload_dst is None:
        raise ValueError(
            "offload_src and offload_dst cannot be None when using optimizer offloading."
        )

    logging.info("Optimizer offloading from %s to %s enabled.", offload_src, offload_dst)

    def init_fn(params: NestedOptParam):
        return optimizer.init(params)

    def _move_fn(state: optax.OptState, dst: MemoryKind) -> optax.OptState:
        # TransferToMemoryKind let us change the memory kind of tensors without specifying the full
        # sharding (i.e. jax.sharding.NamedSharding). Although there's no documentation about it,
        # it's specified in the API signature. Reference:
        # https://github.com/jax-ml/jax/blob/21f8885a9e104b8828c9a8b721eed0c68b622691/jax/_src/api.py#L2220
        # Note: device_put doesn't move everything at once. When we pass a pytree of arrays to
        # device_put, each array in the pytree is moved independent of one another. The exact order
        # is decided by the latency hiding scheduler. The scheduler will try to overlap the
        # transfers of each state with the state update on TPU whenever possible. There is some
        # memory spike due the the temporary state in HBM, but the spike is much less than the full
        # memory usage of all states. Moreover, when the optimizer is run, all activations are
        # released, so we have less memory pressure at that point in time.
        return jax.tree.map(
            lambda path, tensor: (
                jax.device_put(tensor, TransferToMemoryKind(dst))
                if re.fullmatch(pattern, path)
                else tensor
            ),
            tree_paths(state),
            state,
        )

    def update_fn(updates: optax.Updates, state: optax.OptState, params: NestedOptParam):
        state = _move_fn(state, offload_src)
        updates, state = optimizer.update(updates, state, params)
        state = _move_fn(state, offload_dst)
        return updates, state

    def partition_fn(param_spec: Nested[ParameterSpec]) -> Nested[OptStateSpec]:
        return copy_partition(
            optimizer.partition(param_spec), pattern=pattern, memory_kind=offload_dst
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)
