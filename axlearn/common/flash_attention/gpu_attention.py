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

This implementation follows the original closely:
https://github.com/HazyResearch/flash-attention/blob/9818f85fee29ac6b60c9214bce841f8109a18b1b/flash_attn/flash_attn_triton.py
https://github.com/google/jax/blob/jaxlib-v0.4.25/jax/experimental/pallas/ops/attention.py

As well as the original paper: https://arxiv.org/abs/2205.14135

Due to the caveats mentioned in the above link, we make several simplifying assumptions:
* Sequence length is a multiple of block size (128).
* No dropout is applied.
* 4-d bias tensor is supported.
* Currently only tested on A100/H100.
"""
# pylint: disable=wrong-import-position,missing-param-doc,differing-param-doc
import functools
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp

# pytype: disable=import-error  # pylint: disable=import-error
from jax import lax
from jax.experimental import pallas as pl

from axlearn.common.attention import NEG_INF

# pytype: enable=import-error  # pylint: enable=import-error

Tensor = jax.Array


def _bias_mask(
    q_bias_ids: Tensor,
    kv_bias_ids: Tensor,
):
    """Build the segment mask for the given query and key bias ids."""
    # [B, T, 1] or [T, 1]
    q_bias_ids = jnp.expand_dims(q_bias_ids, axis=-1)
    # [B, 1, S] or [1, S]
    kv_bias_ids = jnp.expand_dims(kv_bias_ids, axis=-2)
    return jnp.equal(q_bias_ids, kv_bias_ids).astype(jnp.bool_)


def _mha_forward_kernel(
    # Inputs.
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    # Outputs.
    o_ref,
    # Residual outputs.
    *residual_refs,
    softmax_scale: float,
    bias_type: str,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
):
    """Computes attention outputs for the given block.

    For details and references:
    https://github.com/jax-ml/jax-triton/blob/46991edf162d1d630f64524e7c999e041a7f5126/jax_triton/pallas/ops/attention.py
    https://arxiv.org/abs/2205.14135 Appendix B.3 Algorithm 2.

    See also `_mha_backward_kernel` for the backward pass.

    Args:
        q_ref: Input query ref.
        k_ref: Input key ref.
        v_ref: Input value ref.
        b_ref: Input bias ref.
        o_ref: Output ref.
        *residual_refs: Residual output refs, e.g. softmax statistics.
        softmax_scale: Softmax scale.
        bias_type: Type of bias matrix.
        causal: Whether to apply causal mask.
        block_q: Block size for query seq dim.
        block_d: Block size for head dim.
        block_k: Block size for key seq dim.
    """
    seq_len = q_ref.shape[0]
    start_q = pl.program_id(0)

    # acc is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_q, dtype=jnp.float32) + NEG_INF
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    acc = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pl.load(q_ref, (curr_q_slice, pl.dslice(None)))

    # Effectively a segment id for padding mask.
    if bias_type == "vector":
        q_bias_ids = pl.load(b_ref, (curr_q_slice,))

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        acc, m_prev, l_prev = carry
        # This is slow loop over kv, essentially a scan through.
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = pl.load(k_ref, (curr_k_slice, pl.dslice(None)))
        qk = pl.dot(q, k.T)  # [block_q, block_k].
        if softmax_scale != 1.0:
            qk *= softmax_scale  # [block_q, block_k].

        if bias_type == "matrix":
            b = pl.load(
                b_ref,
                (curr_q_slice, curr_k_slice),
            )
            qk += b
        elif bias_type == "vector":
            kv_bias_ids = pl.load(b_ref, (curr_k_slice,))

        if causal or bias_type == "vector":
            mask = None
            if bias_type == "vector":
                mask = _bias_mask(q_bias_ids, kv_bias_ids)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            # Apply mask to qk.
            qk = jnp.where(mask, qk, NEG_INF)

        # Bring closer to XLA:GPU numerics.
        # These casts are needed to avoid precision issues.
        qk = qk.astype(jnp.float32)
        m_curr = qk.max(axis=-1)
        m_curr = jnp.maximum(m_curr, m_prev)
        l_prev *= jnp.exp(m_prev - m_curr)
        p = jnp.exp(qk - m_curr[:, None])
        l_curr = jnp.sum(p, axis=1) + l_prev
        l_rcp = 1.0 / l_curr
        p = p * l_rcp[:, None]
        acc_prev = (l_prev * l_rcp)[:, None] * acc

        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        acc_curr = pl.dot(p.astype(v.dtype), v)
        acc_next = acc_prev + acc_curr
        return acc_next, m_curr, l_curr

    if causal:
        upper_bound = lax.div(block_q * start_q, block_k) + 1
    else:
        upper_bound = pl.cdiv(seq_len, block_k)
    acc, m_i, l_i = lax.fori_loop(0, upper_bound, body, (acc, m_i, l_i))

    if residual_refs:
        l_ref, m_ref = residual_refs
        pl.store(l_ref, (curr_q_slice,), l_i)
        pl.store(m_ref, (curr_q_slice,), m_i)

    # Write output to dram.
    acc = acc.astype(o_ref.dtype)
    pl.store(o_ref, (curr_q_slice, pl.dslice(None)), acc)


@functools.partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor] = None,
    softmax_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "triton",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    grid: Optional[Sequence[int]] = None,
    interpret: bool = False,
    debug: bool = False,
):
    """Computes attention outputs following FlashAttention.

    Args:
        query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
        key: Key of shape [batch_size, source_length, num_heads, per_head_dim].
        value: Value of shape [batch_size, source_length, num_heads, per_head_dim].
        bias: Optional logit biases of shape [batch_size, num_heads, target_length, source_length].
        softmax_scale: Optional scale to apply to softmax. Defaults to 1/sqrt(per_head_dim).
        causal: Whether to apply causal mask.
        **kwargs: Pallas/triton kwargs.

    Returns:
        The attention outputs of shape [batch_size, target_length, num_heads, per_head_dim].
    """
    del backward_pass_impl
    # Configure the grid and triton kernel specs.
    batch_size, seq_len, num_heads, head_dim = query.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    # Bias.
    bias_type = "none"
    bias_block_spec = None
    if bias is not None:
        # If we have explicitly defined the bias type as a vector (segment ids).
        if bias.ndim == 2:
            bias_type = "vector"
            bias_block_spec = pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
        elif bias.ndim == 4:
            bias_type = "matrix"
            bias_block_spec = pl.BlockSpec(
                lambda _, j, k: (j, k, 0, 0), (None, None, seq_len, seq_len)
            )
        else:
            raise ValueError(f"Invalid bias shape: {bias.shape}")

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    num_stages_ = num_stages
    if num_stages_ is None:
        num_stages_ = 2 if head_dim <= 64 else 1
    kernel = functools.partial(
        _mha_forward_kernel,
        softmax_scale=softmax_scale,
        bias_type=bias_type,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
    )
    out_shape = jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)

    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # query
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # key
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # value
            bias_block_spec,  # bias
        ],
        out_specs=pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        compiler_params=dict(triton=dict(num_warps=num_warps_, num_stages=num_stages_)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(query, key, value, bias)


def _mha_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    softmax_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: Optional[int],
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    """Calls `_mha_forward_kernel`."""
    del backward_pass_impl
    # Configure the grid and triton kernel specs.
    batch_size, seq_len, num_heads, head_dim = query.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    # Bias.
    bias_type = "none"
    bias_block_spec = None
    if bias is not None:
        # If we have explicitly defined the bias type as a vector (segment ids).
        if bias.ndim == 2:
            bias_type = "vector"
            bias_block_spec = pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
        elif bias.ndim == 4:
            bias_type = "matrix"
            bias_block_spec = pl.BlockSpec(
                lambda _, j, k: (j, k, 0, 0), (None, None, seq_len, seq_len)
            )
        else:
            raise ValueError(f"Invalid bias shape: {bias.shape}")

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    num_stages_ = num_stages
    if num_stages_ is None:
        num_stages_ = 2 if head_dim <= 64 else 1
    kernel = functools.partial(
        _mha_forward_kernel,
        softmax_scale=softmax_scale,
        bias_type=bias_type,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype),  # out
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),  # l
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), dtype=jnp.float32),  # m
    ]

    out, l, m = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # query
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # key
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # value
            bias_block_spec,  # bias
        ],
        out_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        compiler_params=dict(triton=dict(num_warps=num_warps_, num_stages=num_stages_)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(query, key, value, bias)
    return out, (query, key, value, bias, out, l, m)


def _preprocess_backward_kernel(
    out_ref,
    dout_ref,
    l_ref,
    new_dout_ref,
    delta_ref,
    *,
    block_q: int,
):
    """Precomputes Di for the attention backwards pass.

    This optimization is described in https://arxiv.org/abs/2205.14135 Appendix B.4 observation 2.
    """
    pid_m = pl.program_id(0)

    off_m = pl.ds(pid_m * block_q, block_q)
    # Load.
    o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
    do = pl.load(dout_ref, (off_m, slice(None))).astype(jnp.float32)
    denom = pl.load(l_ref, (off_m,)).astype(jnp.float32)
    # Compute.
    do = do / denom[:, None]
    delta = jnp.sum(o * do, axis=1)
    # Write-back.
    pl.store(new_dout_ref, (off_m, slice(None)), do.astype(new_dout_ref.dtype))
    pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))


@jax.named_scope("preprocess_backward")
def _preprocess_backward(
    out,
    do,
    l,
    block_q: int,
    debug: bool,
    interpret: bool,
):
    """Calls `_preprocess_backward_kernel`."""
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = [
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct(l.shape, l.dtype),
    ]
    do_scaled, delta = pl.pallas_call(
        functools.partial(_preprocess_backward_kernel, block_q=block_q),
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        out_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        compiler_params=dict(triton=dict(num_warps=4, num_stages=3)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do, l)
    return do_scaled, delta


def _mha_backward_kernel(
    # Inputs.
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    out_ref,
    do_scaled_ref,
    l_ref,
    m_ref,
    delta_ref,
    _,
    # Outputs.
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    softmax_scale: float,
    bias_type: str,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
):
    """Computes the backward pass.

    This algorithm is described in https://arxiv.org/abs/2205.14135 Appendix B.4 Algorithm 4.

    See also `_mha_forward_kernel` for the forward pass.

    Args:
        q_ref: Input query ref.
        k_ref: Input key ref.
        v_ref: Input value ref.
        b_ref: Input bias ref.
        out_ref: Input forward output ref.
        do_scaled_ref: Preprocessed dOut ref. See `_preprocess_backward_kernel`.
        l_ref: Input l ref.
        m_ref: Input m ref.
        delta_ref: Input delta ref. See `_preprocess_backward_kernel`.
        dq_ref: Output dQuery ref.
        dk_ref: Output dKey ref.
        dv_ref: Output dValue ref.
        softmax_scale: Softmax scale.
        bias_type: Type of bias matrix.
        causal: Whether to apply causal mask.
        block_q: Block size for query seq dim.
        block_d: Block size for head dim.
        block_k: Block size for key seq dim.
    """
    del out_ref, l_ref  # Not needed
    seq_len = q_ref.shape[0]

    def outer_loop(start_k, _):
        dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        slice_k = pl.ds(start_k * block_k, block_k)
        k = pl.load(k_ref, (slice_k, slice(None)))
        v = pl.load(v_ref, (slice_k, slice(None)))
        span_k = start_k * block_k + jnp.arange(block_k)
        kv_bias_ids = None if bias_type != "vector" else pl.load(b_ref, (slice_k))

        def inner_loop(start_q, carry):
            dv, dk = carry
            slice_q = pl.ds(start_q * block_q, block_q)
            q = pl.load(q_ref, (slice_q, slice(None)))
            qk = pl.dot(q, k.T)

            # These casts are needed to avoid precision issues.
            qk = qk.astype(jnp.float32)

            if softmax_scale != 1.0:
                qk *= softmax_scale
            if bias_type == "matrix":
                # Load bias in transposed order, for hopefully better cache efficiency.
                b = pl.load(
                    b_ref,
                    (slice_k, slice_q),
                )
                b = b.astype(jnp.float32)
                qk += b.T  # Transpose back.
            elif bias_type == "vector":
                q_bias_ids = pl.load(b_ref, (slice_q))
            if causal or bias_type == "vector":
                mask = None
                if bias_type == "vector":
                    mask = _bias_mask(q_bias_ids, kv_bias_ids)
                if causal:
                    span_q = start_q * block_q + jnp.arange(block_q)
                    causal_mask = span_q[:, None] >= span_k[None, :]
                    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                qk = jnp.where(mask, qk, NEG_INF)

            m = pl.load(m_ref, (slice_q,))
            p = jnp.exp(qk - m[:, None])
            do = pl.load(do_scaled_ref, (slice_q, slice(None)))
            dv = dv + pl.dot(p.astype(do.dtype).T, do)
            di = pl.load(delta_ref, (slice_q,))
            dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
            dp = dp + pl.dot(do, v.T)
            ds = p * dp
            if softmax_scale != 1.0:
                ds = ds * softmax_scale
            dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)
            dq = pl.load(
                dq_ref,
                (slice_q, slice(None)),
                eviction_policy="evict_last",
            )
            dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
            pl.store(dq_ref, (slice_q, slice(None)), dq, eviction_policy="evict_last")
            return dv, dk

        if causal:
            lower_bound = lax.div(start_k * block_k, block_q)
        else:
            lower_bound = 0
        dv, dk = lax.fori_loop(lower_bound, pl.cdiv(seq_len, block_q), inner_loop, (dv, dk))
        pl.store(dv_ref, (slice_k, slice(None)), dv.astype(dv_ref.dtype))
        pl.store(dk_ref, (slice_k, slice(None)), dk.astype(dk_ref.dtype))

    lax.fori_loop(0, pl.cdiv(seq_len, block_k), outer_loop, None)


def _mha_backward(
    softmax_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: Optional[int],
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    res,
    do,
):
    """Calls `_mha_backward_kernel`."""
    del num_warps, num_stages, grid
    q, k, v, b, out, l, m = res

    # NOTE: temporarily removed the "xla" branch, which seems unused.
    if backward_pass_impl == "triton":
        batch_size, seq_len, num_heads, head_dim = q.shape
        # Backward heuristics, using the same block size for block q and block k.
        block_q = min(block_q, seq_len)
        block_k = min(block_k, seq_len)
        # Very tiny amount of time, not worth using pallas_call.
        do_scaled, delta = _preprocess_backward(out, do, l, block_q, debug, interpret)
        # We accumulate into dq so we need to initialize it to zeros.
        dq = jnp.zeros(q.shape, jnp.float32)
        out_shapes = [
            jax.ShapeDtypeStruct(dq.shape, dq.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]

        # Bias.
        bias_type = "none"
        bias_block_spec = None
        input_output_aliases = {8: 0}
        if b is not None:
            if b.ndim == 2:
                bias_type = "vector"
                bias_block_spec = pl.BlockSpec(lambda j, k: (j, 0), (None, seq_len))
            elif b.ndim == 4:
                bias_type = "matrix"
                # Transpose seq dims for cache efficiency.
                b = jnp.moveaxis(b, -1, -2)
                bias_block_spec = pl.BlockSpec(
                    lambda j, k: (j, k, 0, 0), (None, None, seq_len, seq_len)
                )
            else:
                raise ValueError(f"Invalid bias shape: {b.shape}")
            # We have one more non-None input (i.e., in_specs has another tree_leaf).
            input_output_aliases = {9: 0}

        grid = (batch_size, num_heads)
        # TODO(markblee): num_warps=8 seems to work from basic testing, confirm the below comment.
        # TODO(sharadmv): figure out why num_warps=8 doesn't work!
        num_warps = 8
        dq, dk, dv = pl.pallas_call(
            functools.partial(
                _mha_backward_kernel,
                softmax_scale=softmax_scale,
                bias_type=bias_type,
                causal=causal,
                block_q=block_q,
                block_d=head_dim,
                block_k=block_k,
            ),
            grid=grid,
            out_shape=out_shapes,
            in_specs=[
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # query
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # key
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),  # value
                bias_block_spec,  # bias
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
                pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
                pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
                pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            ],
            out_specs=[
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
                pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            ],
            name="mha_backward",
            debug=debug,
            interpret=interpret,
            compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=1)),
            input_output_aliases=input_output_aliases,
        )(q, k, v, b, out, do_scaled, l, m, delta, dq)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, None


flash_attention.defvjp(_mha_forward, _mha_backward)
