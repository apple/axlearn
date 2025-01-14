# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# HazyResearch/flash-attention:
# Copyright (c) 2022, Tri Dao, Dan Fu. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implements FlashAttention for GPU in JAX with logit bias support.

This implementation is ported from
https://github.com/jax-ml/jax/blob/ed4e9823b19591f8a4c98b1f895c284775d6e0c7/jax/experimental/pallas/ops/gpu/attention.py
and follows the original papers closely:
FlashAttention: https://arxiv.org/abs/2205.14135
FlashAttention2: https://arxiv.org/abs/2307.08691

Caveats of this implementation:
* Sequence length must be a multiple of block size (128).
* Only tested on A100/H100.

Compared to the implementation in the JAX repo, we made the following enhancements:
* Support kv_seq_len != q_seq_len.
* Support 2D/4D bias.
* Support dropout.
"""
import functools
from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.cudnn.fused_attention_stablehlo import MaskType, dot_product_attention
from jax.experimental import pallas as pl

from axlearn.common.attention import NEG_INF
from axlearn.common.layers import get_dropout_mask

Tensor = jax.Array


class NoPopDict(dict):
    """A dict that doesn't delete after pop.

    Used to workaround https://github.com/jax-ml/jax/issues/25714.
    """

    def pop(self, *args, **kwargs):
        return super().get(*args, **kwargs)


def _segment_mask(
    q_segment_ids: Tensor,
    kv_segment_ids: Tensor,
):
    """Build the segment mask for the given query and key bias ids.

    If mask[..., i, j] == True, query position i and key position j
    are in the same segment.
    """
    # [B, T, 1] or [T, 1]
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    # [B, 1, S] or [1, S]
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=-2)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


def _mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    s_ref,
    dropout_mask_ref,
    # Outputs.
    o_ref,
    # Residual outputs.
    *residual_refs,
    softmax_scale: float,
    causal: bool,
    dropout_rate: float,
    block_q: int,
    block_k: int,
):
    """Computes attention outputs for the given block.

    For details and references:
    https://github.com/jax-ml/jax-triton/blob/46991edf162d1d630f64524e7c999e041a7f5126/jax_triton/pallas/ops/attention.py
    https://arxiv.org/abs/2205.14135 Appendix B.3 Algorithm 2.

    See also `_mha_backward_kernel` for the backward pass.

    Note: the kernel name is used to do string matching for rematerialization in `remat.py`. Be
    careful when renaming this.

    Args:
        q_ref: Input query ref.
        k_ref: Input key ref.
        v_ref: Input value ref.
        b_ref: Input bias ref.
        s_ref: Input segment_ids ref.
        o_ref: Output ref.
        *residual_refs: Residual output refs, e.g. softmax statistics.
        **kwargs: See `flash_attention`.
    """
    kv_seq_len = k_ref.shape[0]
    block_d = q_ref.shape[-1]
    start_q = pl.program_id(0)
    precision = (
        lax.Precision.HIGHEST
        if jnp.float32 in (q_ref.dtype, k_ref.dtype, v_ref.dtype)
        else lax.Precision.DEFAULT
    )

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.full(block_q, NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = q_ref[...]
    q_segment_ids = None if s_ref is None else pl.load(s_ref, (curr_q_slice,))

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire kv_seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T, precision=precision)  # [block_q, block_k].
        if softmax_scale != 1.0:
            qk *= softmax_scale  # [block_q, block_k].
        if b_ref is not None:
            qk += pl.load(b_ref, (slice(None), curr_k_slice))
        qk = jnp.maximum(qk, NEG_INF)

        if causal or s_ref is not None:
            mask = None
            if s_ref is not None:
                kv_segment_ids = pl.load(s_ref, (curr_k_slice,))
                mask = _segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            # Apply mask to qk.
            qk = jnp.where(mask, qk, NEG_INF)

        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        if dropout_rate > 0:
            dropout_mask = pl.load(dropout_mask_ref, (slice(None), curr_k_slice))
            s_curr = jnp.where(dropout_mask, 0, s_curr / (1 - dropout_rate))
        o_curr = pl.dot(s_curr.astype(v.dtype), v, precision=precision)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    if causal:
        upper_bound = jnp.minimum(
            lax.div((start_q + 1) * block_q, block_k), pl.cdiv(kv_seq_len, block_k)
        )
    else:
        upper_bound = pl.cdiv(kv_seq_len, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    # We keep an unscaled version of o during the scan over kv_seq_len. Scaling it
    # by the last l_i gives us the correct final output. See section 3.1.1 in the
    # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log(l_i)
    # Write output to dram.
    o_ref[...] = o.astype(o_ref.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "causal",
        "block_q",
        "block_k",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "dropout_rate",
        "output_activations",
    ],
)
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor] = None,
    segment_ids: Optional[Tensor] = None,
    prng_key: Optional[Tensor] = None,
    softmax_scale: float = 1.0,
    causal: bool = False,
    dropout_rate: float = 0.0,
    block_q: int = 128,
    block_k: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    grid: Optional[Sequence[int]] = None,
    interpret: bool = False,
    debug: bool = False,
    # output_activations has to be the last arg for custom vjp to work.
    output_activations: bool = False,
):
    """Computes attention outputs following FlashAttention.

    If provided, bias, segment_ids, and any causal mask are applied on top of one another.

    Args:
        query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
        key: Key of shape [batch_size, source_length, num_heads, per_head_dim].
        value: Value of shape [batch_size, source_length, num_heads, per_head_dim].
        bias: Optional logit biases of shape [batch_size, num_heads, target_length, source_length].
        segment_ids: Optional segment ids of shape [batch_size, target_length].
        prng_key: PRNG key used for dropout. Must be specified when dropout_rate > 0.0.
        softmax_scale: Optional scale to apply to softmax. Defaults to 1.
        causal: Whether to apply causal mask.
        dropout_rate: Dropout rate. Default to 0.0 (no dropout).
        output_activations: Whether to output activations for backward. Default to False.
        **kwargs: Pallas/triton kwargs.

    Returns:
        The attention outputs of shape [batch_size, target_length, num_heads, per_head_dim].
    """
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    kv_seq_len = key.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    assert q_seq_len % block_q == 0
    assert kv_seq_len % block_k == 0
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    if num_stages is None:
        num_stages = (
            2 if bias is None and jnp.float32 not in (query.dtype, key.dtype, value.dtype) else 1
        )
    if num_warps is None:
        num_warps = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        _mha_forward_kernel,
        softmax_scale=softmax_scale,
        causal=causal,
        dropout_rate=dropout_rate,
        block_q=block_q,
        block_k=block_k,
    )
    out_shape = jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)  # out
    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    if bias is not None:
        assert bias.ndim == 4
        in_specs.append(
            pl.BlockSpec(
                index_map=lambda i, j, k: (
                    j if bias.shape[0] != 1 else 0,
                    k if bias.shape[1] != 1 else 0,
                    i,
                    0,
                ),
                block_shape=(None, None, block_q, kv_seq_len),
            )
        )
    else:
        in_specs.append(None)
    in_specs.append(
        None if segment_ids is None else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    if dropout_rate > 0:
        assert dropout_rate < 1
        assert prng_key is not None
        # TODO(hanzhi-zhou): Switch to in-kernel RNG when pallas supports it.
        dropout_mask = get_dropout_mask(
            (batch_size, num_heads, q_seq_len, kv_seq_len), prng_key=prng_key, rate=dropout_rate
        )
        in_specs.append(
            pl.BlockSpec((None, None, block_q, kv_seq_len), lambda i, j, k: (j, k, i, 0))
        )
    else:
        dropout_mask = None
        in_specs.append(None)
    out_specs = pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0))
    if output_activations:
        out_specs = [out_specs, pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i))]
        out_shape = [
            out_shape,
            jax.ShapeDtypeStruct(
                shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32
            ),  # lse
        ]
    pallas_out = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=NoPopDict(triton=NoPopDict(num_warps=num_warps, num_stages=num_stages)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(query, key, value, bias, segment_ids, dropout_mask)
    if output_activations:
        out, lse = pallas_out
        return out, (query, key, value, bias, segment_ids, prng_key, out, lse)
    return pallas_out


def _mha_forward(*args: Any):
    """Wraps flash_attention for custom vjp."""
    return flash_attention(*args[:-1], output_activations=True)


def _mha_backward_kernel_dkdv(
    # Inputs.
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    s_ref,
    dropout_mask_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs.
    dk_ref,
    dv_ref,
    *,
    softmax_scale: float,
    causal: bool,
    dropout_rate: float,
    block_q: int,
    block_k: int,
):
    """Computes dK and dV.
    1. Load a block of K and V of size (block_k, head_dim) in SMEM.
    2. Iterate through Q in chunks of (block_q, head_dim) to accumulate
       dK and dV.
    """
    q_seq_len = q_ref.shape[0]
    block_d = q_ref.shape[-1]
    precision = (
        lax.Precision.HIGHEST
        if jnp.float32 in (q_ref.dtype, k_ref.dtype, v_ref.dtype)
        else lax.Precision.DEFAULT
    )

    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)

    v = pl.load(v_ref, (curr_k_slice, slice(None)))
    k = pl.load(k_ref, (curr_k_slice, slice(None)))
    span_k = start_k * block_k + jnp.arange(block_k)
    kv_segment_ids = None if s_ref is None else pl.load(s_ref, (curr_k_slice,))

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q, block_q)

        q = pl.load(q_ref, (curr_q_slice, slice(None)))
        qk = pl.dot(q, k.T, precision=precision)  # type: ignore
        if softmax_scale != 1.0:
            qk *= softmax_scale
        if b_ref is not None:
            qk += pl.load(b_ref, (curr_q_slice, curr_k_slice))
        qk = jnp.maximum(qk, NEG_INF)

        if causal or s_ref is not None:
            mask = None
            if s_ref is not None:
                q_segment_ids = pl.load(s_ref, (curr_q_slice,))
                mask = _segment_mask(q_segment_ids, kv_segment_ids)

            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            qk = jnp.where(mask, qk, NEG_INF)

        lse = pl.load(lse_ref, (curr_q_slice,))
        di = pl.load(delta_ref, (curr_q_slice,))
        do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = p_dropped = jnp.exp(qk - lse[:, None])
        dp = dp_dropped = pl.dot(do, v.T, precision=precision)  # type: ignore
        if dropout_rate > 0:
            dropout_mask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            p_dropped = jnp.where(dropout_mask, 0, p / (1 - dropout_rate))
            dp = jnp.where(dropout_mask, 0, dp_dropped / (1 - dropout_rate))
        dv = dv + pl.dot(p_dropped.astype(do.dtype).T, do, precision=precision)
        dp = dp - di[:, None]
        ds = p * dp
        if softmax_scale != 1.0:
            ds = ds * softmax_scale
        dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q, precision=precision)

        return dv, dk

    lower_bound = lax.div(start_k * block_k, block_q) if causal else 0
    dv, dk = lax.fori_loop(lower_bound, pl.cdiv(q_seq_len, block_q), inner_loop_dkdv, (dv, dk))
    pl.store(dv_ref, (curr_k_slice, slice(None)), dv.astype(dv_ref.dtype))
    pl.store(dk_ref, (curr_k_slice, slice(None)), dk.astype(dk_ref.dtype))


def _mha_backward_kernel_dq(
    # Inputs.
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    s_ref,
    dropout_mask_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs.
    dq_ref,
    *,
    softmax_scale: float,
    causal: bool,
    dropout_rate: float,
    block_q: int,
    block_k: int,
):
    """Computes dQ.
    1. Load a block of Q of size (block_q, head_dim) in SMEM.
    2. Iterate through K and V in chunks of (block_k, head_dim) to
       accumulate dQ.
    """
    kv_seq_len = k_ref.shape[0]
    block_d = q_ref.shape[-1]
    precision = (
        lax.Precision.HIGHEST
        if jnp.float32 in (q_ref.dtype, k_ref.dtype, v_ref.dtype)
        else lax.Precision.DEFAULT
    )

    start_q = pl.program_id(2)
    curr_q_slice = pl.ds(start_q * block_q, block_q)
    span_q = start_q * block_q + jnp.arange(block_q)
    dq = jnp.zeros([block_q, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    q_segment_ids = None if s_ref is None else pl.load(s_ref, (curr_q_slice,))
    lse = pl.load(lse_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pl.load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        v = pl.load(v_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T, precision=precision)
        if softmax_scale != 1.0:
            qk *= softmax_scale
        if b_ref is not None:
            qk += pl.load(b_ref, (curr_q_slice, curr_k_slice))
        qk = jnp.maximum(qk, NEG_INF)

        if causal or s_ref is not None:
            mask = None
            if s_ref is not None:
                kv_segment_ids = pl.load(s_ref, (curr_k_slice,))
                mask = _segment_mask(q_segment_ids, kv_segment_ids)

            if causal:
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            qk = jnp.where(mask, qk, NEG_INF)

        p = jnp.exp(qk - lse[:, None])
        dp = dp_dropped = pl.dot(do, v.T, precision=precision)
        if dropout_rate > 0:
            dropout_mask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            dp = jnp.where(dropout_mask, 0, dp_dropped / (1 - dropout_rate))
        dp = dp - di[:, None]
        ds = p * dp
        if softmax_scale != 1.0:
            ds = ds * softmax_scale
        dq = dq + pl.dot(ds.astype(k.dtype), k, precision=precision)
        return dq

    if causal:
        upper_bound = jnp.minimum(
            pl.cdiv((start_q + 1) * block_q, block_k), pl.cdiv(kv_seq_len, block_k)
        )
    else:
        upper_bound = pl.cdiv(kv_seq_len, block_k)

    dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
    pl.store(dq_ref, (curr_q_slice, slice(None)), dq.astype(dq_ref.dtype))


def _mha_backward(
    softmax_scale: float,
    causal: bool,
    dropout_rate: float,
    block_q: int,
    block_k: int,
    num_warps: Optional[int],
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    output_activations: bool,
    res,
    do,
):
    """Calls Pallas kernels to compute dQ, dK and dV.

    Note: separating dKdV and dQ loops into two kernels in flash backward improved performance by
    10~15% when head_dim >= 128. Technically, fusing dKdVdQ into a single loop and use atomic add
    for dQ is the fastest solution, but pallas atomics are extremely slow according to empirical
    testing.
    """
    del num_warps, grid, output_activations
    q, k, v, bias, segment_ids, prng_key, out, lse = res
    # We must shrink the block size for float32 inputs to avoid OOM during bwd pass.
    if jnp.float32 in (q.dtype, k.dtype, v.dtype, jnp.bfloat16 if bias is None else bias.dtype):
        block_q = block_k = 32

    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    # Compute delta (D) as in Algorithm 2 Line 4 of FlashAttention2.
    delta = (
        (out.astype(jnp.float32) * do.astype(jnp.float32))
        .sum(axis=3)
        .transpose((0, 2, 1))
        .astype(lse.dtype)
    )

    if dropout_rate > 0:
        dropout_mask = get_dropout_mask(
            (batch_size, num_heads, q_seq_len, kv_seq_len), prng_key=prng_key, rate=dropout_rate
        )
    else:
        dropout_mask = None

    in_specs = [
        pl.BlockSpec((None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),  # q
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),  # k
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),  # v
        (
            None
            if bias is None
            else pl.BlockSpec(
                index_map=lambda i, j, _: (
                    i if bias.shape[0] != 1 else 0,
                    j if bias.shape[1] != 1 else 0,
                    0,
                    0,
                ),
                block_shape=(None, None, q_seq_len, kv_seq_len),
            )
        ),
        None if segment_ids is None else pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0)),
        (
            None
            if dropout_mask is None
            else pl.BlockSpec((None, None, q_seq_len, kv_seq_len), lambda i, j, _: (i, j, 0, 0))
        ),
        pl.BlockSpec((None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),  # do
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),  # lse
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),  # delta
    ]

    num_warps = 8
    if num_stages is None:
        num_stages = 2 if bias is None and jnp.float32 not in (q.dtype, k.dtype, v.dtype) else 1

    def call_kernel(*, kernel, grid, out_shape, out_specs):
        return pl.pallas_call(
            functools.partial(
                kernel,
                softmax_scale=softmax_scale,
                causal=causal,
                dropout_rate=dropout_rate,
                block_q=block_q,
                block_k=block_k,
            ),
            out_shape=out_shape,
            in_specs=in_specs,
            grid=grid,
            out_specs=out_specs,
            name=kernel.__name__,
            debug=debug,
            interpret=interpret,
            compiler_params=NoPopDict(triton=NoPopDict(num_warps=num_warps, num_stages=num_stages)),
        )(q, k, v, bias, segment_ids, dropout_mask, do, lse, delta)

    dk, dv = call_kernel(
        kernel=_mha_backward_kernel_dkdv,
        grid=(batch_size, num_heads, pl.cdiv(kv_seq_len, block_k)),
        out_shape=[
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        out_specs=[
            pl.BlockSpec(
                (None, kv_seq_len, None, head_dim),
                lambda i, j, _: (i, 0, j, 0),  # dk
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, head_dim),
                lambda i, j, _: (i, 0, j, 0),  # dv
            ),
        ],
    )

    dq = call_kernel(
        kernel=_mha_backward_kernel_dq,
        grid=(batch_size, num_heads, pl.cdiv(q_seq_len, block_q)),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        out_specs=pl.BlockSpec(
            (None, q_seq_len, None, head_dim),
            lambda i, j, _: (i, 0, j, 0),  # dq
        ),
    )
    return dq, dk, dv, None, None, None


flash_attention.defvjp(_mha_forward, _mha_backward)


# Interface to cuDNN's dot product attention.
# TODO(kelvin-zou): Add support for segment IDs.
def cudnn_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    causal: bool = False,
    *,
    softmax_scale: float = 1.0,
    seed: int = 42,
    dropout_rate: float = 0.0,
    qkv_layout: str = "BTNH",
):
    """Computes dot-product attention given query (Q), key (K), and value (V).

    If provided, bias, segment_ids, and any causal mask are applied on top of one another.

    Reference implementation:
    https://github.com/google/jax/blob/f4158ace933482844c145a6b919bf5dc86e084ba/jax/_src/cudnn/fused_attention_stablehlo.py#L927.
    https://github.com/openxla/xla/blob/536ba0b7d74f6637a7a772471a99ecf4f578aef2/xla/service/gpu/cublas_cudnn.cc#L77.

    Args:
        query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
        key: Key of shape [batch_size, source_length, num_heads, per_head_dim].
        value: Value of shape [batch_size, source_length, num_heads, per_head_dim].
        bias: Optional logit biases of shape [batch_size, num_heads, target_length, source_length].
        mask: Optional logit mask of shape [batch_size, num_heads, target_length, source_length].
        softmax_scale: Optional scale to apply to softmax. Defaults to 1.
        seed: Random seed for dropout.
        dropout_rate: Dropout rate.
        qkv_layout: Layout string, with supported formats being BTNH, BNTH, BSNH,
                    BNSH. Now it only supports BTNH.

    Returns:
        Output of the same shape as the query.

    Raises:
        NotImplementedError: If qkv_layout is not supported.
    """

    if qkv_layout != "BTNH":
        raise NotImplementedError(f"Unsupported qkv_layout: {qkv_layout}")

    mask_type = MaskType.NO_MASK if not causal else MaskType.CAUSAL

    output = dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        mask=mask,
        scale=softmax_scale,
        seed=seed,
        dropout_rate=dropout_rate,
        mask_type=mask_type,
        qkv_layout=qkv_layout,
    )
    return output
