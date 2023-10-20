# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as tpu_flash_attention

from axlearn.common.utils import Tensor


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "softmax_scale",
        "block_sizes",
    ],
)
def flash_attention(
    query: Tensor,  # [batch_size, q_seq_len, num_heads, d_model]
    key: Tensor,  # [batch_size, kv_seq_len, num_heads, d_model]
    value: Tensor,  # [batch_size, kv_seq_len, num_heads, d_model]
    bias: Tensor = None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
    block_sizes: Optional[BlockSizes] = None,
):
    """Wraps JAX's TPU flash-attention, with reshapes and softmax-scaling outside kernel.

    N.B. we apply the softmax scale factor outside of the kernel because:
        1. within-kernel ordering of attention-bias addition and softmax scaling differ to axlearn,
        2. it's more efficient to scale outside the kernel vs. fix order of ops in kernel.

    Args:
        query: The query tensor, of shape [batch_size, target_seq_len, num_heads, head_dim].
        key: The key tensor, of shape [batch_size, source_seq_len, num_heads, head_dim].
        value: The value tensor, of shape [batch_size, source_seq_len, num_heads, head_dim].
        bias: The attention biases, of shape [batch_size, num_heads, q_seq_len, source_seq_len].
        causal: Whether the attention is causal (allows for additional optimizations).
        softmax_scale: A scaling factor applied to the query.

    Returns:
        The context tensor, of shape [batch_size, q_seq_len, num_heads, head_dim].

    """
    # Apply the softmax scale outside the kernel (see docstring for why).
    if softmax_scale != 1.0:
        query *= softmax_scale
    # Switch num_heads and seq_len axes.
    query = jnp.einsum("btnh->bnth", query)
    key = jnp.einsum("bsnh->bnsh", key)
    value = jnp.einsum("bsnh->bnsh", value)
    context = tpu_flash_attention(
        q=query,
        k=key,
        v=value,
        ab=bias,
        causal=causal,
        # If sm_scale==1.0, the kernel skips applying it.
        sm_scale=1.0,
        block_sizes=block_sizes,
        debug=False,
    )
    # Restore num_heads and seq_len axes.
    return jnp.einsum("bnth->btnh", context)
