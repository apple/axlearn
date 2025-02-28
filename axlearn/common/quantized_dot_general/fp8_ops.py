# Copyright © 2025 Apple Inc.
"""Ops for FP8 training."""

from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax.linen import fp8_ops

# pylint: disable-next=unused-import
from flax.linen.fp8_ops import q_dot_dq as q_dot_dq_delayed
from jax.custom_derivatives import custom_vjp

from axlearn.common.utils import Tensor


def _q_dot_dq_impl(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    dimension_numbers: tuple,
    preferred_element_type: Optional[Any],
    is_training: bool,
) -> Union[Tensor, tuple[Tensor, tuple[Tensor, ...]]]:
    """See `q_dot_dq_in_batch`.

    Also returns the residuals for custom_vjp backward if `is_training` is True.
    """
    dtype_max = fp8_ops.get_fp8_max(jnp.float8_e4m3fn, jnp.float32)
    amax_lhs = jnp.max(jnp.abs(lhs)).astype(lhs_scale.dtype)
    amax_rhs = jnp.max(jnp.abs(rhs)).astype(rhs_scale.dtype)
    new_lhs_scale = fp8_ops.compute_scale(amax_lhs, lhs_scale, dtype_max)
    new_rhs_scale = fp8_ops.compute_scale(amax_rhs, rhs_scale, dtype_max)

    q_lhs = fp8_ops.quantize(lhs, jnp.float8_e4m3fn, new_lhs_scale, preferred_element_type)
    q_rhs = fp8_ops.quantize(rhs, jnp.float8_e4m3fn, new_rhs_scale, preferred_element_type)

    out = jax.lax.dot_general(
        q_lhs,
        q_rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=jax.lax.Precision.HIGHEST,
    )

    out = fp8_ops.dequantize(out, preferred_element_type, new_lhs_scale * new_rhs_scale)
    if is_training:
        res = (
            lhs,
            rhs,
            q_lhs,
            q_rhs,
            new_lhs_scale,
            new_rhs_scale,
            out_grad_scale,
        )
        return out, res
    else:
        return out


# pylint: disable=unused-argument
@partial(custom_vjp, nondiff_argnums=(5, 6))
def q_dot_dq_in_batch(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    dimension_numbers: tuple,
    preferred_element_type: Optional[Any] = None,
) -> Tensor:
    """Computes lhs @ rhs using fp8 in-batch scaling.

    lhs and rhs are divided by scales computed using the amax values before performing matmul in
    fp8 precision. The scales passed into this function are previous scales, used when the newly
    computed scales are zero or inf.

    Args:
        lhs: Left-hand side tensor of matmul.
        lhs: Right-hand side tensor of matmul.
        lhs_scale: The previous scale of lhs.
        rhs_scale: The previous scale of rhs.
        out_grad_scale: The previous scale of output gradient.
        dimension_numbers: See `lax.dot_general`.
        preferred_element_type: See `lax.dot_general`.

    Returns:
        The result of lhs @ rhs after dequantization.
    """
    return _q_dot_dq_impl(
        **locals(),
        is_training=False,
    )


def _q_dot_dq_fwd(
    lhs: Tensor,
    rhs: Tensor,
    lhs_scale: Tensor,
    rhs_scale: Tensor,
    out_grad_scale: Tensor,
    dimension_numbers: tuple,
    preferred_element_type: Optional[Any],
):
    """See `q_dot_dq_in_batch`."""
    return _q_dot_dq_impl(
        **locals(),
        is_training=True,
    )


# pylint: enable=unused-argument


def _q_dot_dq_bwd(
    dimension_numbers: tuple,
    preferred_element_type: Optional[Any],
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
    ) = res

    dtype_max = fp8_ops.get_fp8_max(jnp.float8_e5m2, jnp.float32)
    amax_grad = jnp.max(jnp.abs(g)).astype(out_grad_scale.dtype)
    new_out_grad_scale = fp8_ops.compute_scale(amax_grad, out_grad_scale, dtype_max)
    q_g = fp8_ops.quantize(g, jnp.float8_e5m2, new_out_grad_scale, preferred_element_type)

    grad_lhs = fp8_ops.dot_general_transpose_lhs(
        q_g,
        lhs,
        q_rhs,
        dimension_numbers=dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=preferred_element_type,
    )
    grad_lhs = fp8_ops.dequantize(
        grad_lhs, preferred_element_type, new_rhs_scale * new_out_grad_scale
    )

    grad_rhs = fp8_ops.dot_general_transpose_rhs(
        q_g,
        q_lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=preferred_element_type,
    )
    grad_rhs = fp8_ops.dequantize(
        grad_rhs, preferred_element_type, new_lhs_scale * new_out_grad_scale
    )

    return (
        grad_lhs,
        grad_rhs,
        new_lhs_scale,
        new_rhs_scale,
        new_out_grad_scale,
    )


q_dot_dq_in_batch.defvjp(_q_dot_dq_fwd, _q_dot_dq_bwd)
