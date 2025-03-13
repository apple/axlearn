# Copyright © 2025 Apple Inc.
"""Implements TPU decoding.

Unlike GPU, TPU blocks are sequential (except when there're two cores). Therefore, unlike GPU
decoding, there's no need to parallelize over the KV sequence length. As the result, it works
very similar to full attention. The grid dimensions are
(batch_size, num_kv_heads, num_kv_blocks).

The main reason to use the kernel is that it can take advantage of the fact that most KV blocks
are padding in practical decoding scenarios. Also, it can take advantage of sparsity in
`mask_fn`.

Performance note:
1. When kv_seq_len == padded_kv_seq_len:
    This kernels performs similarly to non-fused (i.e. XLA) attention, or within 10% slower.
2. When kv_seq_len < padded_kv_seq_len or `mask_fn` has sparsity:
    This kernel provides speed up roughly equal to padded_kv_seq_len / kv_seq_len or number
    of masked kv blocks / total kv blocks.

The main reason why non-fused attention is faster when kv are not padded is that the non-fused
matmuls can flatten the non-head dimensions, thus having larger non-contracting dimensions.
This leads to have better utilization of the matrix and memory units.
"""
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.attention_bias import NEG_INF, MaskFn
from axlearn.common.flash_attention.common import build_mask, query_iterator_indices
from axlearn.common.utils import Tensor


def _tpu_decoding_kernel(
    # Scalars.
    kv_seq_len_ref,
    kv_block_offset,
    kv_block_offset_size,
    # Inputs.
    q_ref,
    k_ref,
    v_ref,
    b_ref,
    # Outputs.
    o_ref,
    # Scatch.
    m_i,
    l_i,
    o_scratch,
    # Compile time args.
    softmax_scale: float,
    mask_fn: Optional[MaskFn],
):
    batch_index = pl.program_id(0)
    non_empty_kv_block_index = pl.program_id(2)
    _, block_k = k_ref.shape
    precision = (
        lax.Precision.HIGHEST if jnp.float32 in (q_ref.dtype, k_ref.dtype, v_ref.dtype) else None
    )

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    @pl.when(non_empty_kv_block_index == 0)
    def init():
        m_i[...] = jnp.full_like(m_i, NEG_INF)
        l_i[...] = jnp.zeros_like(l_i)
        o_scratch[...] = jnp.zeros_like(o_scratch)

    # Note: on CPU interpret mode, pl.program_id() cannot appear in functions decorated by
    # pl.when.
    kv_offset = kv_block_offset[batch_index, non_empty_kv_block_index] * block_k
    kv_seq_len = kv_seq_len_ref[batch_index]
    num_non_empty_kv_blocks = kv_block_offset_size[batch_index]

    # Different batch may have different number of-non empty kv blocks.
    @pl.when(non_empty_kv_block_index < num_non_empty_kv_blocks)
    def compute():
        q = q_ref[...]
        k = k_ref[...].astype(q.dtype)
        qk = pl.dot(q, k, precision=precision)
        if softmax_scale != 1.0:
            qk *= softmax_scale
        if b_ref is not None:
            qk += b_ref[...]
            qk = jnp.maximum(qk, NEG_INF)
        # Note: Pallas TPU requires the use of lax.broadcasted_iota instead of jnp.arange as only
        # 2D range is supported.
        block_kv_indices = kv_offset + lax.broadcasted_iota(jnp.int32, qk.shape, 1)
        kv_mask = block_kv_indices < kv_seq_len
        if mask_fn is not None:
            kv_mask = kv_mask & mask_fn(kv_seq_len - 1, block_kv_indices)
        qk = jnp.where(kv_mask, qk, NEG_INF)

        m_prev = m_i[...]
        l_prev = l_i[...]
        o_prev = o_scratch[...]

        # We need to make sure each array has two dims, or we get TPU Mosaic lowering errors.
        m_curr = qk.max(axis=-1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp(m_prev - m_next)
        l_prev_corr = correction * l_prev
        # Use m_next instead of m_curr to avoid a correction on l_curr.
        s_curr = jnp.exp(qk - m_next)
        l_curr = s_curr.sum(axis=-1, keepdims=True)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction * o_prev
        v = v_ref[...].astype(q.dtype)
        o_curr = pl.dot(s_curr.astype(v.dtype), v.T, precision=precision)

        o_next = o_prev_corr + o_curr

        m_i[...] = m_next
        l_i[...] = l_next
        o_scratch[...] = o_next

    @pl.when(non_empty_kv_block_index == num_non_empty_kv_blocks - 1)
    def final():
        # We keep an unscaled version of o during the scan over kv_seq_len. Scaling it
        # by the last l_i gives us the correct final output. See section 3.1.1 in the
        # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
        o_ref[...] = (o_scratch[...] / l_i[...]).astype(o_ref.dtype)


@partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "mask_fn",
        "block_size",
        "interpret",
    ],
)
def tpu_decoding(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_seq_len: Optional[Tensor],
    bias: Optional[Tensor] = None,
    *,
    softmax_scale: float = 1.0,
    mask_fn: Optional[MaskFn] = None,
    block_size: int = 512,
    interpret: bool = False,
):
    """Implements TPU decoding with GQA support.

    The functionality of TPU decoding is similar to GPU FlashDecoding, except that
    padded_kv_seq_len must be divisible by block_size.

    Args:
        q: Tensor of shape [batch_size, 1, num_q_heads, head_dim].
        k: Tensor of shape [batch_size, padded_kv_seq_len, num_kv_heads, head_dim].
        v: Tensor of shape [batch_size, padded_kv_seq_len, num_kv_heads, head_dim].
        kv_seq_len: Tensor that can broadcast to [batch_size], indicating the actual kv sequence
            length for each sequence in the batch. If None, assumes k and v are not padded in the
            sequence dimension.
        bias: Tensor that can broadcast to [batch_size, num_q_heads, 1, padded_kv_seq_len].
            Defaults to None.
        softmax_scale: Softmax scale.
        mask_fn: Mask function to use. Preferred over bias.
        block_size: Block dimension along the sequence dim. Defaults to 512.

    Returns:
        A tensor with the same shape and dtype as q.

    Raises:
        ValueError if the shape of qkv doesn't satisfy assumptions.
    """
    if q.shape[1] != 1:
        raise ValueError("Multi-step decoding is not supported yet.")
    # Pallas TPU doesn't support pl.load(..., mask=xxx), so we kv len must divide block size.
    # However, we can reduce the block size to support the case where
    # padded_kv_seq_len < block_size.
    block_size = min(block_size, k.shape[1])
    if k.shape[1] % block_size != 0:
        raise ValueError(f"KV sequence length {k.shape[1]} must be divisible by {block_size=}.")
    orig_q_shape = q.shape
    q_seq_len = q.shape[1]
    block_kv = block_size
    q = q.squeeze(1)
    # Convert to bnhs which is the native shape of KV in the kv cache. These two transposes should
    # be elided by the compiler. See `BaseQKVLinear.init_states` from attention.py.
    k = jnp.einsum("bsnh->bnhs", k)
    v = jnp.einsum("bsnh->bnhs", v)
    bs, kv_heads, head_dim, padded_kv_seq_len = k.shape
    if kv_seq_len is not None:
        kv_seq_len = jnp.broadcast_to(jnp.asarray(kv_seq_len), (bs,))
    else:
        kv_seq_len = jnp.full((bs,), padded_kv_seq_len, dtype=jnp.int32)

    # Computes a full block map num_kv_blocks * num_kv_blocks.
    # Use a padding to ensure padding blocks aren't counted towards `kv_block_offset_size`.
    padding = -1
    with jax.ensure_compile_time_eval():
        if mask_fn is not None:
            bool_mask = build_mask(
                mask_fn,
                q_seq_len=padded_kv_seq_len,
                kv_seq_len=padded_kv_seq_len,
                block_q=block_size,
                block_k=block_size,
            )
            offset, _ = query_iterator_indices(bool_mask, padding=padding)
        else:
            padded_num_kv_blocks = pl.cdiv(padded_kv_seq_len, block_size)
            offset = lax.broadcasted_iota(
                jnp.int32, (padded_num_kv_blocks, padded_num_kv_blocks), 1
            )

    # Dynamically slice the rows according to the query position (which is kv_seq_len - 1).
    kv_block_offset = offset[(kv_seq_len - 1) // block_size]
    # Count the number of blocks with position < kv_seq_len.
    kv_block_offset_size = jnp.count_nonzero(
        (kv_block_offset != padding) & (kv_block_offset * block_size < kv_seq_len[:, None]), axis=1
    )
    # Replace padding with the last valid kv block's index. See
    # https://docs.jax.dev/en/latest/pallas/tpu/sparse.html#sparse-access-patterns-on-dense-data
    kv_block_offset = jnp.where(
        kv_block_offset == padding, kv_block_offset.max(axis=1, keepdims=True), kv_block_offset
    )

    q = q.reshape(bs, kv_heads, -1, head_dim)
    q_seq_head = q.shape[-2]  # = q_seq_len * num_q_heads_per_kv_head
    assert q_seq_head <= 512

    def kv_index_map(
        batch_idx, head_idx, kv_block_idx, kv_seq_len, kv_block_offset, kv_block_offset_size
    ):
        del kv_seq_len, kv_block_offset_size
        return (batch_idx, head_idx, 0, kv_block_offset[batch_idx, kv_block_idx])

    q_spec = pl.BlockSpec((None, None, q_seq_head, head_dim), lambda b, h, j, *args: (b, h, 0, 0))
    kv_spec = pl.BlockSpec((None, None, head_dim, block_kv), kv_index_map)
    bias_spec = None
    if bias is not None:
        if bias.shape[0] == 1 and bias.shape[1] == 1:

            def bias_index_map(
                batch_idx,
                head_idx,
                kv_block_idx,
                kv_seq_len,
                kv_block_offset,
                kv_block_offset_size,
            ):
                del head_idx, kv_seq_len, kv_block_offset_size
                return (0, 0, 0, kv_block_offset[batch_idx, kv_block_idx])

            bias_spec = pl.BlockSpec((None, None, q_seq_len, block_kv), bias_index_map)
        else:
            bias = bias.reshape(bs, kv_heads, q_seq_head, padded_kv_seq_len)
            bias_spec = pl.BlockSpec((None, None, q_seq_head, block_kv), kv_index_map)

    out: Tensor = pl.pallas_call(
        partial(_tpu_decoding_kernel, softmax_scale=softmax_scale, mask_fn=mask_fn),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_spec,
                kv_spec,
                kv_spec,
                bias_spec,
            ],
            out_specs=q_spec,
            scratch_shapes=[
                # VMEM requires 2D arrays.
                pltpu.VMEM((q_seq_head, 1), jnp.float32),
                pltpu.VMEM((q_seq_head, 1), jnp.float32),
                pltpu.VMEM((q_seq_head, head_dim), jnp.float32),
            ],
            grid=(bs, kv_heads, kv_block_offset_size.max()),
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(kv_seq_len, kv_block_offset, kv_block_offset_size, q, k, v, bias)
    return out.reshape(orig_q_shape)
