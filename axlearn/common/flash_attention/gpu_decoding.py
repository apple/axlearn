# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implements FlashDecoding.

Reference: https://pytorch.org/blog/flash-decoding.
TLDR: FlashDecoding addresses the issue of SM under-utilization during decoding when batch size is
small and kv sequence length is long by parallelizing over the kv sequence length dimension. Each
thread block handles a chunk of the kv sequence, writing softmax residuals to HBM. The outputs
from each block are then combined and rescaled using these residuals to get the final output.

This file is adapted from
https://github.com/jax-ml/jax/blob/861115ad4bf0f57e53f61d4d083cd2bda6877ab5/jax/experimental/pallas/ops/gpu/decode_attention.py,
but is heavily modified to improve performance and add support for bias and MaskFn:
1. Jax implementation uses double vmap to parallelize over batch and num_kv_heads. This requires
   a axis permute for k and v, resulting in a transpose kernel that doubles the execution time of
   decoding kernel. To remove this transpose, we only vmap the batch dimension and add an
   additional dimension to the Pallas BlockSpec to let Pallas handles the strided k and v load.
2. Added support for attention bias.
3. Added support for MaskFn. The MaskFn makes it possible to support sparse attentions such as
   sliding window attention or global-local attention without materializing the mask as attention
   bias. The kernel can take advantage of sparsity by skipping fully-masked blocks, significantly
   improving performance. Note that we do not materialize a compile time mask. Instead, we rely on
   runtime voting in thread blocks. This leads to simpler code and faster compilation at the cost
   of small runtime overhead (at microseconds level), which I find acceptable.

Performance note (see numbers in gpu_attention_benchmark.py for numerical values):
No sparsity:
1. FlashDecoding is faster than XLA across the board by some margin (5%~20%).
2. FlashDecoding is significantly faster than cudnn for long context when bs * num_kv_head is
   small.
3. FlashDecoding is slightly slower (~5%) than cudnn when bs * num_kv_head is large.
With sparsity such as with sliding window attention, FlashDecoding is few times faster. The
performance gain is proportional to total context length divided by window size.
"""
from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.cudnn.fused_attention_stablehlo import check_compute_capability
from jax.experimental import pallas as pl

from axlearn.common.attention_bias import NEG_INF, MaskFn
from axlearn.common.flash_attention.gpu_attention import NoPopDict
from axlearn.common.utils import Tensor


# Note: split_k_seq_len must be a multiple of block_k.
def _attn_forward_kernel(
    # Inputs:
    q_ref,  # [block_h, head_dim]
    k_ref,  # [split_k_seq_len, head_dim]
    v_ref,  # [split_k_seq_len, head_dim]
    bias_ref,  # [block_h, split_k_seq_len]
    kv_seq_len_ref,  # [] (i.e., scalar)
    # Outputs:
    o_ref,  # [block_h, head_dim]
    l_ref,  # [block_h,]
    m_ref,  # [block_h,]
    # Non-tensors:
    mask_fn: Optional[MaskFn],
    softmax_scale: float,
    block_k: int,
    block_h: int,
    qhead_per_kvhead: int,
):
    _, head_dim = q_ref.shape
    split_k_seq_len, _ = k_ref.shape
    precision = (
        lax.Precision.HIGHEST
        if jnp.float32 in (q_ref.dtype, k_ref.dtype, v_ref.dtype)
        else lax.Precision.DEFAULT
    )
    prog_i, prog_j = pl.program_id(1), pl.program_id(2)
    q_mask = (block_h * prog_i + jnp.arange(block_h) < qhead_per_kvhead)[:, None]

    def _compute(block_kv_start_idx, block_kv_seqlen, o, m_i, l_i):
        # Load q: it will stay in L1 throughout. Indices form a matrix because we
        # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
        # q tile has shape [block_h, head_dim].
        q = pl.load(q_ref, (slice(None), slice(None)), mask=q_mask) * softmax_scale

        mask_indices = jnp.arange(block_k)

        # Loop over blocks of kv to process entire kv seq_len.
        def body(start_k, carry):
            o_prev, m_prev, l_prev = carry

            indices = block_kv_start_idx + start_k * block_k + mask_indices
            # This mask guards against out-of-bound values only.
            mask = indices < block_kv_seqlen
            logits_mask = mask if mask_fn is None else mask_fn(kv_seq_len - 1, indices) & mask

            def compute():
                curr_k_slice = pl.ds(start_k * block_k, block_k)
                k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=mask[:, None], other=0.0)
                qk = pl.dot(q, k.T, precision=precision)  # [block_h, block_k]
                if bias_ref is not None:
                    qk += pl.load(
                        bias_ref, (slice(None), curr_k_slice), mask=mask[None, :], other=0.0
                    )

                qk = jnp.where(logits_mask[None, :], qk, NEG_INF)

                m_curr = qk.max(axis=-1)
                m_next = jnp.maximum(m_prev, m_curr)
                correction = jnp.exp(m_prev - m_next)
                l_prev_corr = correction * l_prev
                # Use m_next instead of m_curr to avoid a correction on l_curr.
                s_curr = jnp.exp(qk - m_next[:, None])
                l_curr = s_curr.sum(axis=-1)
                l_next = l_prev_corr + l_curr
                v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=mask[:, None], other=0.0)
                o_curr = pl.dot(s_curr.astype(v.dtype), v, precision=precision)

                # Flash2 unscaled_o.
                o_next = correction[:, None] * o_prev + o_curr
                return o_next, m_next, l_next

            def no_compute():
                return carry

            # Skip this block if this block is fully masked. Note: loading V is skipped. This
            # basically means that we assume qk is not fully masked across the entire kv_seq_len.
            # Note: cannot use jnp.all as reduce_and is not implemented in pallas/triton.
            return lax.cond(jnp.sum(logits_mask) > 0, compute, no_compute)

        max_it = pl.cdiv(block_kv_seqlen - block_kv_start_idx, block_k)
        (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))
        return o, m_i, l_i

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention2 paper) are updated during the k,v loop.
    m_i = jnp.full(block_h, NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_h, dtype=jnp.float32)
    o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

    block_kv_start_idx = prog_j * split_k_seq_len
    kv_seq_len = pl.load(kv_seq_len_ref, ())
    block_kv_seqlen = jnp.minimum((prog_j + 1) * split_k_seq_len, kv_seq_len)

    # Skip padding in seq dim.
    o, m_i, l_i = jax.lax.cond(
        block_kv_start_idx >= kv_seq_len,
        lambda: (o, m_i, l_i),
        lambda: _compute(block_kv_start_idx, block_kv_seqlen, o, m_i, l_i),
    )

    # Write output to HBM.
    vec_q_mask = q_mask.reshape(-1)
    pl.store(l_ref, slice(None), l_i, mask=vec_q_mask)
    pl.store(m_ref, slice(None), m_i, mask=vec_q_mask)
    pl.store(o_ref, (slice(None), slice(None)), o, mask=q_mask)


def _get_sm_count() -> int:
    """Returns number of SMs for the current GPU or 0 if unknown."""
    if check_compute_capability("9.0"):  # H100
        return 132
    if check_compute_capability("8.9"):  # L4, L40
        return 0
    if check_compute_capability("8.6"):  # A40, A10
        return 0
    # This assumes we're not using A30.
    if check_compute_capability("8.0"):  # A100, A30
        return 108
    return 0


def _decode_attn_unbatched(
    q,  # [kv_heads, qhead_per_kvhead, head_dim]
    k,  # [k_seq_len, kv_heads, head_dim]
    v,  # [k_seq_len, kv_heads, head_dim]
    bias,  # [kv_heads, qhead_per_kvhead, k_seq_len]
    kv_seq_len,  # []
    softmax_scale: float,
    mask_fn: Optional[MaskFn],
    block_h: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    debug: bool,
    batch_size: int,
):
    num_kvheads, qhead_per_kvhead, head_dim = q.shape
    padded_kv_seq_len = k.shape[0]
    head_splits = pl.cdiv(qhead_per_kvhead, block_h)
    # Calculate the intiial k_splits. Cap the k_splits at 16, but increase it if batch_size *
    # qhead_per_kvhead * 16 cannot fully utilize GPU and seqlen is long. Each block has 4 wraps.
    # Each SM can hold at least two of these blocks according to the smem usage.
    good_k_split_for_sm_util = _get_sm_count() // (batch_size * qhead_per_kvhead)
    k_splits = min(max(good_k_split_for_sm_util, 16), pl.cdiv(padded_kv_seq_len, block_k))
    split_k_seq_len = pl.cdiv(padded_kv_seq_len, k_splits)
    # Round up to a multiple of block_k.
    split_k_seq_len = pl.cdiv(split_k_seq_len, block_k) * block_k
    k_splits = pl.cdiv(padded_kv_seq_len, split_k_seq_len)

    grid = (num_kvheads, head_splits, k_splits)
    block_k = min(block_k, split_k_seq_len)
    kernel = functools.partial(
        _attn_forward_kernel,
        softmax_scale=softmax_scale,
        block_k=block_k,
        block_h=block_h,
        qhead_per_kvhead=qhead_per_kvhead,
        mask_fn=mask_fn,
    )

    o, l, m = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            # kv_h = axis along num kv heads.
            # q_h = axis along q heads per kv head.
            # k = axis along kv sequence length.
            pl.BlockSpec((None, block_h, head_dim), lambda kv_h, q_h, k: (kv_h, q_h, 0)),
            pl.BlockSpec((split_k_seq_len, None, head_dim), lambda kv_h, q_h, k: (k, kv_h, 0)),
            pl.BlockSpec((split_k_seq_len, None, head_dim), lambda kv_h, q_h, k: (k, kv_h, 0)),
        ]
        + [
            (
                None
                if bias is None
                else pl.BlockSpec(
                    (None, block_h, split_k_seq_len), lambda kv_h, q_h, k: (kv_h, q_h, k)
                )
            )
        ]
        + [pl.BlockSpec((), lambda kv_h, q_h, k: ())],
        out_specs=[
            pl.BlockSpec(
                (None, None, block_h, head_dim), lambda kv_h, q_h, k: (kv_h, k, q_h, 0)
            ),  # o
            pl.BlockSpec((None, None, block_h), lambda kv_h, q_h, k: (kv_h, k, q_h)),  # l
            pl.BlockSpec((None, None, block_h), lambda kv_h, q_h, k: (kv_h, k, q_h)),  # m
        ],
        compiler_params=NoPopDict(triton=NoPopDict(num_warps=num_warps, num_stages=num_stages)),
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=(num_kvheads, k_splits, *q.shape[1:]), dtype=jnp.float32
            ),  # o
            jax.ShapeDtypeStruct(
                shape=(num_kvheads, k_splits, qhead_per_kvhead), dtype=jnp.float32
            ),  # l
            jax.ShapeDtypeStruct(
                shape=(num_kvheads, k_splits, qhead_per_kvhead), dtype=jnp.float32
            ),  # m
        ],
        debug=debug,
        interpret=interpret,
        name="flash_decoding_forward",
    )(q, k, v, bias, kv_seq_len)

    # Combine the results from blocks into final output.
    m_next = m.max(axis=1, keepdims=True)  # [num_kv_heads, 1, qhead_per_kvhead]
    correction = jnp.exp(m - m_next)  # [num_kv_heads, k_splits, qhead_per_kvhead]
    o = o * correction[..., None]  # [num_kv_heads, k_splits, qhead_per_kvhead, head_dim]
    l_next = (l * correction).sum(axis=1)  # [num_kv_heads, qhead_per_kvhead]
    o = o.sum(axis=1) / (l_next[..., None] + jnp.finfo(l_next.dtype).eps)
    return o.astype(q.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "block_h",
        "block_k",
        "num_warps",
        "num_stages",
        "interpret",
        "debug",
        "mask_fn",
    ],
)
def flash_decoding(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_seq_len: Optional[Tensor],
    *,
    bias: Optional[Tensor] = None,
    softmax_scale: float = 1.0,
    mask_fn: Optional[MaskFn] = None,
    block_h: int = 16,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
    interpret: bool = False,
    debug: bool = False,
):
    """Runs flash decoding with GQA support.

    Args:
        q: Tensor of shape [batch_size, 1, num_q_heads, head_dim].
        k: Tensor of shape [batch_size, padded_kv_seq_len, num_kv_heads, head_dim].
        v: Tensor of shape [batch_size, padded_kv_seq_len, num_kv_heads, head_dim].
        kv_seq_len: Tensor that can broadcast to [batch_size], indicating the actual kv sequence
            length for each sequence in the batch. If None, assumes k and v are not padded in the
            sequence dimension.
        bias: Tensor that can broadcast to [batch_size, q_heads, 1, padded_kv_seq_len].
            Defaults to None.
        softmax_scale: Softmax scale.
        mask_fn: Mask function to use. Preferred over bias.
        block_h: Block dimension for num_q_heads // num_kv_heads. Defaults to 16, which is the
            minimal size required for pl.dot. Increase the block size for better performance if
            num_q_heads // num_kv_heads > 16.
        block_k: Block dimension along the sequence dim. Defaults to 128. Decrease if SMEM usage
            exceeds limit.
        num_warps: See the compiler options of `pl.pallas_call`. Default to 4 wraps which have the
            best performance in most settings.
        num_stages: See the compiler options of `pl.pallas_call`. Default to 2 which is faster than
            no software pipelining. Higher values may cause SMEM OOM.
        interpret: See `pl.pallas_call`.
        debug: See `pl.pallas_call`.

    Returns:
        A tensor with the same shape and dtype as q.
    """
    q = q.squeeze(1)
    batch_size, q_heads, head_dim = q.shape
    padded_kv_seq_len, kv_heads = k.shape[1], k.shape[2]
    if k.shape != v.shape:
        raise RuntimeError(f"Expect k and v to have the same shape. Got {k.shape=}, {v.shape=}!")
    if q_heads % kv_heads != 0:
        raise RuntimeError(
            f"Expect number of kv heads divides number of q heads. Got {kv_heads=}, {q_heads=}!"
        )
    if kv_seq_len is not None:
        kv_seq_len = jnp.broadcast_to(jnp.asarray(kv_seq_len), (batch_size,))
    else:
        kv_seq_len = jnp.full((batch_size,), padded_kv_seq_len, dtype=jnp.int32)
    q_heads_per_kv_head = q_heads // kv_heads
    q = q.reshape(batch_size, kv_heads, q_heads_per_kv_head, head_dim)

    if bias is not None:
        bias = jnp.broadcast_to(bias, (batch_size, q_heads, 1, padded_kv_seq_len))
        bias = bias.reshape(batch_size, kv_heads, q_heads_per_kv_head, padded_kv_seq_len)

    inner = functools.partial(
        _decode_attn_unbatched,
        softmax_scale=softmax_scale,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
        debug=debug,
        mask_fn=mask_fn,
        batch_size=batch_size,
    )
    o = jax.vmap(inner)(q, k, v, bias, kv_seq_len)
    return o.reshape(batch_size, 1, q_heads, head_dim)
