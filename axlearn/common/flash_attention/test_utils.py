# Copyright © 2023 Apple Inc.
"""Common test utils for FlashAttention tests."""

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention_bias import (
    CompositeAttentionBias,
    MaskFn,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    TensorAttentionBias,
)
from axlearn.common.utils import Tensor


def generate_attention_data(
    batch_size,
    query_len,
    kv_len,
    num_heads,
    per_head_dim,
    num_kv_heads: Optional[int] = None,
    mask_fn: Optional[MaskFn] = None,
    attention_bias_type: Literal[None, "2d", "4d"] = None,
    with_segment_ids: bool = False,
    dtype=jnp.bfloat16,
    query_offset: int = 0,
) -> tuple[Tensor, Tensor, Tensor, CompositeAttentionBias]:
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, query_len, num_heads, per_head_dim), dtype=dtype)
    num_kv_heads = num_kv_heads or num_heads
    k = jax.random.normal(k2, (batch_size, kv_len, num_kv_heads, per_head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch_size, kv_len, num_kv_heads, per_head_dim), dtype=dtype)
    attention_bias = None
    if attention_bias_type == "2d":
        attention_bias = jax.random.normal(k4, (1, 1, query_len, kv_len), dtype=dtype)
    elif attention_bias_type == "4d":
        attention_bias = jax.random.normal(
            k4, (batch_size, num_heads, query_len, kv_len), dtype=dtype
        )
    segment_ids = None
    if with_segment_ids:
        segment_ids = jax.random.bernoulli(k5, shape=(batch_size, kv_len)).astype(jnp.int32)
        segment_ids = jnp.cumsum(segment_ids, axis=1)

    bias_list = []
    if mask_fn is not None:
        bias_list.append(
            MaskFnAttentionBias(
                mask_fn,
                target_positions=jnp.arange(query_len)[None] + query_offset,
                source_positions=jnp.arange(kv_len)[None],
            )
        )
    if with_segment_ids:
        bias_list.append(SegmentIdAttentionBias(segment_ids))
    if attention_bias is not None:
        bias_list.append(TensorAttentionBias(attention_bias))
    bias = CompositeAttentionBias(bias_list)

    return q, k, v, bias
