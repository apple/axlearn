# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# deepmind/optax:
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Factorized RMS.

Adapted from optax factorized.py.
"""

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import FactoredState

from axlearn.common.base_layer import FactorizationSpec, NestedParameterSpec, ParameterSpec
from axlearn.common.config import config_for_function
from axlearn.common.optimizer_base import OptStateSpec, PartitionedGradientTransformation
from axlearn.common.schedule import Schedule, adafactor_decay_rate, as_schedule_fn
from axlearn.common.utils import NestedPartitionSpec, PartitionSpec, Tensor


def _factored_dims(
    factored: bool,
    factorization_spec: Optional[FactorizationSpec],
) -> Optional[tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    This function returns a tuple with the two axes to reduce over or None.

    Args:
        factored: whether to use factored second-moment estimator for 2d vars.
        factorization_spec: the factorization spec.

    Returns:
        None or a tuple of ints representing (col_axis, row_axis).

    Raises:
        ValueError: If an invalid factorization spec is provided.
    """
    if not factored or factorization_spec is None:
        return None
    row_axes = [
        index for index, axis_name in enumerate(factorization_spec.axes) if axis_name == "row"
    ]
    col_axes = [
        index for index, axis_name in enumerate(factorization_spec.axes) if axis_name == "col"
    ]
    if not row_axes and not col_axes:
        return None
    if len(row_axes) != 1 or len(col_axes) != 1:
        raise ValueError(f"Invalid factorization_spec: {factorization_spec}")
    return col_axes[0], row_axes[0]


@dataclasses.dataclass
class _UpdateResult:
    """Opaque container that is not traversed by jax.tree.map."""

    update: Tensor  # the update to apply to params.
    v_row: Tensor  # used for factored params.
    v_col: Tensor  # used for factored params.
    v: Tensor  # used for params where factoring is skipped.


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: Schedule = config_for_function(adafactor_decay_rate),
    epsilon: float = 1e-30,
) -> PartitionedGradientTransformation:
    """Scaling by a factored estimate of the gradient rms (as in Adafactor).

    This is a so-called "1+epsilon" scaling algorithm, that is extremely memory
    efficient compared to RMSProp/Adam, and has had wide success when applied to
    large-scale training of attention-based models.

    References:
        [Shazeer et al, 2018](https://arxiv.org/abs/1804.04235)

    Args:
        factored: Indicator of whether to use factored second-moment estimates.
        decay_rate: The second-moment exponential decay schedule.
        epsilon: Regularization constant for squared gradient.

    Returns:
        The corresponding `PartitionedGradientTransformation`.
    """
    decay_rate = as_schedule_fn(decay_rate)

    def _to_state(count: Tensor, result_tree):
        """Maps from a tree of (factored) values to separate trees of values."""
        return FactoredState(
            count=count,
            v_row=jax.tree.map(lambda o: o.v_row, result_tree),
            v_col=jax.tree.map(lambda o: o.v_col, result_tree),
            v=jax.tree.map(lambda o: o.v, result_tree),
        )

    def init_fn(params):
        """Initialise the optimiser's state."""

        def _init(param):
            shape = param.shape
            factored_dims = _factored_dims(factored, param.factorization_spec)
            if factored_dims is not None:
                d1, d0 = factored_dims
                vr_shape = np.delete(shape, d0)
                vc_shape = np.delete(shape, d1)
                return _UpdateResult(
                    update=jnp.zeros((1,)),
                    v_row=jnp.zeros(vr_shape),
                    v_col=jnp.zeros(vc_shape),
                    v=jnp.zeros((1,)),
                )
            else:
                return _UpdateResult(
                    update=jnp.zeros((1,)),
                    v_row=jnp.zeros((1,)),
                    v_col=jnp.zeros((1,)),
                    v=jnp.zeros(param.shape),
                )

        return _to_state(jnp.zeros([], jnp.int32), jax.tree.map(_init, params))

    def update_fn(grads, state, params):
        """Apply gradient transformation."""
        if params is None:
            raise ValueError("param is None")

        def _update(grad, v_row, v_col, v, param, step):
            grad = grad.astype(jnp.float32)
            decay_rate_t = decay_rate(step)

            # Scaled by factorized second moment statistics.
            new_v_row = jnp.zeros((1,), dtype=jnp.float32)
            new_v_col = jnp.zeros((1,), dtype=jnp.float32)
            new_v = jnp.zeros((1,), dtype=jnp.float32)

            grad_sqr = jnp.square(grad) + epsilon
            factored_dims = _factored_dims(factored, param.factorization_spec)
            if factored_dims is not None:
                d1, d0 = factored_dims
                new_v_row = decay_rate_t * v_row + (1.0 - decay_rate_t) * jnp.mean(
                    grad_sqr, axis=d0
                )
                new_v_col = decay_rate_t * v_col + (1.0 - decay_rate_t) * jnp.mean(
                    grad_sqr, axis=d1
                )
                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
                row_factor = (new_v_row / row_col_mean) ** -0.5
                col_factor = (new_v_col) ** -0.5
                update = (
                    grad
                    * jnp.expand_dims(row_factor, axis=d0)
                    * jnp.expand_dims(col_factor, axis=d1)
                )
            else:
                new_v = decay_rate_t * v + (1.0 - decay_rate_t) * grad_sqr
                update = grad * (new_v) ** -0.5

            return _UpdateResult(update, new_v_row, new_v_col, new_v)

        # Transform grad and compute new per-parameter stats.
        output = jax.tree.map(
            lambda *args: _update(*args, state.count),
            grads,
            state.v_row,
            state.v_col,
            state.v,
            params,
        )

        # Unpack updates / stats and return.
        updates = jax.tree.map(lambda o: o.update, output)
        return updates, _to_state(optax.safe_int32_increment(state.count), output)

    @dataclasses.dataclass
    class VxSpec:
        v_row: Optional[OptStateSpec]
        v_col: Optional[OptStateSpec]
        v: Optional[OptStateSpec]

    def get_vx_spec(param_spec: ParameterSpec) -> VxSpec:
        p_shape = param_spec.shape
        p_partition = param_spec.mesh_axes
        factorization_spec = param_spec.factorization
        dummy_spec = OptStateSpec(
            dtype=jnp.float32,
            shape=(1,),
            mesh_axes=PartitionSpec(
                None,
            ),
        )
        if (
            not factored
            or factorization_spec is None
            or all(f is None for f in factorization_spec.axes)
        ):
            return VxSpec(
                v_row=dummy_spec,
                v_col=dummy_spec,
                v=OptStateSpec(dtype=param_spec.dtype, shape=p_shape, mesh_axes=p_partition),
            )
        factorization = factorization_spec.axes
        vr_spec = OptStateSpec(
            dtype=param_spec.dtype,
            shape=[dim for dim, f in zip(p_shape, factorization) if f != "row"],
            mesh_axes=PartitionSpec(*[p for p, f in zip(p_partition, factorization) if f != "row"]),
        )
        vc_spec = OptStateSpec(
            dtype=param_spec.dtype,
            shape=[dim for dim, f in zip(p_shape, factorization) if f != "col"],
            mesh_axes=PartitionSpec(*[p for p, f in zip(p_partition, factorization) if f != "col"]),
        )
        if (
            len(vr_spec.mesh_axes) != len(p_partition) - 1
            or len(vc_spec.mesh_axes) != len(p_partition) - 1
        ):
            raise ValueError(f"Unexpected factorization: {factorization} for {param_spec}")
        return VxSpec(v_row=vr_spec, v_col=vc_spec, v=dummy_spec)

    def partition_fn(param_specs: NestedParameterSpec) -> NestedPartitionSpec:
        vx_specs = jax.tree.map(get_vx_spec, param_specs)
        return optax.FactoredState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            v_row=jax.tree.map(lambda vx: vx.v_row, vx_specs),
            v_col=jax.tree.map(lambda vx: vx.v_col, vx_specs),
            v=jax.tree.map(lambda vx: vx.v, vx_specs),
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)
