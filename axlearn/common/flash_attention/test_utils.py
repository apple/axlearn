# Copyright © 2023 Apple Inc.
"""Common test utils for FlashAttention tests."""

from typing import Literal, Optional

import jax
import jax.numpy as jnp
import pytest

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    CompositeAttentionBias,
    MaskFn,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    SlidingWindowAttentionBias,
    TensorAttentionBias,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.utils import Tensor


def generate_attention_data(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_heads: int,
    per_head_dim: int,
    num_kv_heads: Optional[int] = None,
    mask_fn: Optional[MaskFn] = None,
    sliding_window_sz: Optional[int] = None,
    attention_bias_type: Literal[None, "2d", "4d"] = None,
    with_segment_ids: bool = False,
    dtype=jnp.bfloat16,
    query_offset: int = 0,
) -> tuple[Tensor, Tensor, Tensor, CompositeAttentionBias]:
    """Generates QKV and Bias for unit test purposes."""
    if sliding_window_sz is not None and sliding_window_sz != -1:
        # Sliding window size conflicts with the following mask fns.
        assert mask_fn is not causal_mask and mask_fn is not sliding_window_causal_mask
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, query_len, num_heads, per_head_dim), dtype=dtype)
    num_kv_heads = num_kv_heads or num_heads
    kv_len = kv_len or query_len
    if kv_len != query_len and with_segment_ids:
        pytest.skip(reason="segment ids require kv_seq_len == q_seq_len")
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
    pos = dict(
        target_positions=jnp.arange(query_len)[None] + query_offset,
        source_positions=jnp.arange(kv_len)[None],
    )
    if mask_fn is not None:
        if mask_fn is causal_mask:
            bias_list.append(CausalAttentionBias(**pos))
        else:
            bias_list.append(MaskFnAttentionBias(mask_fn, **pos))
    if sliding_window_sz is not None and sliding_window_sz != -1:
        bias_list.append(
            SlidingWindowAttentionBias(
                sliding_window_causal_mask(sliding_window_sz),
                **pos,
                sliding_window_size=sliding_window_sz,
            )
        )
    if with_segment_ids:
        bias_list.append(SegmentIdAttentionBias(segment_ids))
    if attention_bias is not None:
        bias_list.append(TensorAttentionBias(attention_bias))
    bias = CompositeAttentionBias(bias_list)

    return q, k, v, bias
