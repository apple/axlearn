# Copyright © 2025 Apple Inc.
"""Ops for FP8 training. Doesn't support gradient accumulation yet."""

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax.linen import fp8_ops
from jax._src.typing import DTypeLike
from jax.custom_derivatives import custom_vjp

from axlearn.common.utils import Tensor


def _quantize(
    x: Tensor,
    scale: Tensor,
    amax_history: Optional[Tensor],
    *,
    dtype: DTypeLike,
    preferred_element_type: DTypeLike,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    dtype_max = fp8_ops.get_fp8_max(dtype, jnp.float32)

    # This branch handles unbalanced batch dimension, like (X, Y) @ (B, X, Y), e.g in
    # FusedQKVLinear. In general, it's not mathematically possible to compute the gradient when
    # each batch of lhs/rhs has different scaling factor and one of them is missing a batch dim.
    # In these cases, only use one scaling factor.
    is_non_balanced_batch = len(scale.shape) > 0
    if is_non_balanced_batch:
        full_scale = scale
        full_amax_history = amax_history
        if amax_history is not None:
            assert scale.shape[0] == amax_history.shape[0]
            amax_history = amax_history[0]
        scale = scale[0]

    if amax_history is None:
        amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
        new_history = None
    else:
        amax = jnp.max(amax_history, axis=0)
        new_history = fp8_ops.compute_amax_history(x, amax_history)
    new_scale = fp8_ops.compute_scale(amax, scale, dtype_max)
    q_x = fp8_ops.quantize(x, dtype, new_scale, preferred_element_type)

    if is_non_balanced_batch:
        new_scale = full_scale.at[0].set(new_scale)
        if new_history is not None:
            new_history = full_amax_history.at[0].set(new_history)

    return q_x, new_scale, new_history


def _dequantize(x: Tensor, scale: Tensor, *, dq_dtype: DTypeLike):
    if len(scale.shape) > 0:
        scale = scale[0]
    return x.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), x.shape)


def _q_dot_dq_impl(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
    is_training: bool,
) -> Union[Tensor, tuple[Tensor, tuple[Tensor, ...]]]:
    """See `q_dot_dq_in_batch`.

    Also returns the residuals for custom_vjp backward if `is_training` is True.
    """
    q_lhs, lhs_scale, lhs_amax_history = _quantize(
        lhs,
        lhs_scale,
        lhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )
    q_rhs, rhs_scale, rhs_amax_history = _quantize(
        rhs,
        rhs_scale,
        rhs_amax_history,
        dtype=jnp.float8_e4m3fn,
        preferred_element_type=preferred_element_type,
    )

    out = jax.lax.dot_general(
        q_lhs,
        q_rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=precision,
    )

    out = _dequantize(out, lhs_scale * rhs_scale, dq_dtype=preferred_element_type)
    if is_training:
        res = (
            lhs,
            rhs,
            q_lhs,
            q_rhs,
            lhs_scale,
            rhs_scale,
            out_grad_scale,
            lhs_amax_history,
            rhs_amax_history,
            out_grad_amax_history,
        )
        return out, res
    else:
        return out


# pylint: disable=unused-argument
@partial(custom_vjp, nondiff_argnums=(8, 9, 10))
def q_dot_q(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike = None,
) -> Tensor:
    """Computes lhs @ rhs in FP8 using either in-batch scaling or delayed scaling.

    If the amax histories are None, in-batch scaling is used. Otherwise, delayed scaling is used.

    lhs and rhs are divided by scales computed using the amax values before performing matmul in
    fp8 precision. The scales passed into this function are previous scales, used when the newly
    computed scales are zero or inf.

    Args:
        lhs: Left-hand side tensor of matmul.
        lhs: Right-hand side tensor of matmul.
        lhs_scale: The previous scale of lhs.
        rhs_scale: The previous scale of rhs.
        out_grad_scale: The previous scale of output gradient.
        lhs_amax_history: Amax history of lhs.
        rhs_amax_history: Amax history of rhs.
        out_grad_amax_history: Amax history of output gradient.
        dimension_numbers: See `lax.dot_general`.
        precision: Precision of the dot.
        preferred_element_type: See `lax.dot_general`.

    Returns:
        The result of lhs @ rhs after dequantization.
    """
    return _q_dot_dq_impl(**locals(), is_training=False)


def _q_dot_dq_fwd(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    lhs_amax_history: Optional[Tensor],
    rhs_amax_history: Optional[Tensor],
    out_grad_amax_history: Optional[Tensor],
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
):
    """See `q_dot_dq_in_batch`."""
    return _q_dot_dq_impl(
        **locals(),
        is_training=True,
    )


# pylint: enable=unused-argument


def _q_dot_dq_bwd(
    dimension_numbers: tuple,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DTypeLike,
    res: tuple[Tensor, ...],
    g: Tensor,
) -> tuple[Tensor, ...]:
    (
        lhs,
        rhs,
        q_lhs,
        q_rhs,
        new_lhs_scale,
        new_rhs_scale,
        out_grad_scale,
        lhs_amax_history,
        rhs_amax_history,
        out_grad_amax_history,
    ) = res

    q_g, new_out_grad_scale, out_grad_amax_history = _quantize(
        g,
        out_grad_scale,
        out_grad_amax_history,
        dtype=jnp.float8_e5m2,
        preferred_element_type=preferred_element_type,
    )

    grad_lhs = fp8_ops.dot_general_transpose_lhs(
        q_g,
        lhs,
        q_rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    grad_lhs = _dequantize(
        grad_lhs, new_rhs_scale * new_out_grad_scale, dq_dtype=preferred_element_type
    )

    grad_rhs = fp8_ops.dot_general_transpose_rhs(
        q_g,
        q_lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    grad_rhs = _dequantize(
        grad_rhs, new_lhs_scale * new_out_grad_scale, dq_dtype=preferred_element_type
    )

    return (
        grad_lhs,
        grad_rhs,
        new_lhs_scale,
        new_rhs_scale,
        new_out_grad_scale,
        lhs_amax_history,
        rhs_amax_history,
        out_grad_amax_history,
    )


q_dot_q.defvjp(_q_dot_dq_fwd, _q_dot_dq_bwd)
