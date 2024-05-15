# Copyright © 2023 Apple Inc.

"""FlashAttention utilities shared amongst CPU/GPU/TPU backends."""
import functools
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from absl import logging
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes

from axlearn.common.attention import NEG_INF
from axlearn.common.flash_attention.tpu_attention import flash_attention as tpu_flash_attention
from axlearn.common.utils import Tensor


@functools.partial(jax.jit, static_argnames=["causal", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Optional[Tensor] = None,
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
) -> Tensor:
    """Reference multi-headed attention implementation.

    Args:
        q: query tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        k: key tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        v: value tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        bias: bias tensor with shape [batch_size, num_heads, seq_len, seq_len] for matrix bias,
                and [batch_size, seq_len] for vector bias.
        causal: whether the attention is causal.
        softmax_scale: a scalar value applied to the logits before softmax.
        bias_type: the type of bias to apply. "matrix" for matrix bias, "vector" for additive bias.

    Returns:
        A tensor with shape [batch_size, seq_len, num_heads, per_head_dim].
    """
    # We apply the scale factor before the attention biases.
    q *= softmax_scale
    logits = jnp.einsum("btnh,bsnh->bnts", q, k)

    # Check if we need to build a segment id mask.
    if bias is not None:
        # matrix bias, shape [batch_size, ..., seq_len, seq_len]
        if bias.ndim >= 3:
            logits += bias.astype(logits.dtype)
        else:  # vector bias, shape [batch_size, seq_len]
            target_segment_ids = jnp.expand_dims(bias, -1)
            source_segment_ids = jnp.expand_dims(bias, -2)
            # Target [b..., t] + Source [b..., s] -> [b..., t, s]
            # [b, 1, ..., t, s] where the value at [..., i, j] = false if
            # target_segments[..., i] == source_segments[..., j], or true otherwise.
            mask = jax.lax.ne(source_segment_ids, target_segment_ids)[:, None, ...]
            logits = jnp.where(mask, NEG_INF, logits)

    if causal:
        mask_shape = (q.shape[1], k.shape[1])
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (row_ids < col_ids)[None, None, :, :]
        logits = jnp.where(causal_mask, NEG_INF, logits)

    logits_dtype = logits.dtype
    logits = logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1).astype(logits_dtype)
    return jnp.einsum("bnts,bsnh->btnh", probs, v)


# Accepts [query, key, value, attention_bias] tensors and returns the context Tensor.
MultiHeadAttentionImpl = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def flash_attention_implementation(
    backend: Literal["cpu", "tpu", "gpu"],
    *,
    causal: bool,
    softmax_scale: float,
    block_size: int = 128,
) -> MultiHeadAttentionImpl:
    """Returns a jitted "flash" multihead-attention implementation for the given backend.

    Args:
        backend: A valid XLA backend name. 'cpu' intended for testing only.
        causal: Whether the attention is causal (allows for additional efficiency optimizations).
        softmax_scale: A scalar value applied to the logits before softmax.
        block_size: The size of the computation-block unit, only applies to the 'tpu' backend.
            A multiple of 128, and should be less than the target sequence length.
            Smaller values are more memory efficient but less compute efficient.

    Returns:
        A jitted function implementing multi-head attention for the given backend.

    Raises:
        NotImplementedError: If implementation for the backend is not available.
    """
    if backend == "gpu":
        try:
            # pylint: disable-next=import-outside-toplevel
            import transformer_engine.jax as te

            logging.info("Using TE flash-attention implementation.")

            # TODO(kelvin-zou): Update to raw te library over flax library.
            # TODO(kelvin-zou): Support bias in the future.
            def jit_attn(query, key, value, _):
                _, _, q_heads, per_head_dim = query.shape
                _, _, kv_heads, _ = key.shape
                te_flash_attention = te.flax.DotProductAttention(
                    head_dim=per_head_dim,
                    num_attention_heads=q_heads,
                    num_gqa_groups=kv_heads,
                    attn_mask_type="causal" if causal else "no_mask",
                    attn_bias_type="no_bias",
                    attention_dropout=0.0,
                    dropout_rng_name="dropout",
                    dtype=query.dtype,
                    float32_logits=True,
                    qkv_layout="BSHD_BSHD_BSHD",
                    scale_factor=softmax_scale,
                    transpose_batch_sequence=False,
                )
                context = te_flash_attention.apply({}, query, key, value)
                return context

        except ModuleNotFoundError:
            logging.warning("TE flash-attention implementation not found.")
            # Lazy import GPU flash-attention to avoid file-level dependency on jax-triton.
            # Fall back to triton based flash attention.

            # pylint: disable-next=import-outside-toplevel
            from axlearn.common.flash_attention.gpu_attention import (
                flash_attention as gpu_flash_attention,
            )

            logging.info("Using Triton flash-attention implementation.")

            # shard_map-decorated function needs to be jitted.
            @jax.jit
            def jit_attn(query, key, value, bias):
                return gpu_flash_attention(
                    query, key, value, bias=bias, causal=causal, softmax_scale=softmax_scale
                )

        return jit_attn

    elif backend == "tpu":
        # TODO(tom_gunter): See if we can do better block-size tuning.
        block_sizes = BlockSizes(
            block_q=block_size,
            block_k_major=block_size,
            block_k=block_size,
            block_b=1,
            block_q_major_dkv=block_size,
            block_k_major_dkv=block_size,
            block_k_dkv=block_size,
            block_q_dkv=block_size,
            block_k_major_dq=block_size,
            block_k_dq=block_size,
            block_q_dq=block_size,
        )

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias):
            context = tpu_flash_attention(
                query,
                key,
                value,
                bias=bias,
                causal=causal,
                softmax_scale=softmax_scale,
                block_sizes=block_sizes,
            )
            return context

        return jit_attn

    elif backend == "cpu":
        logging.warning("Flash attention CPU backend is for testing only.")

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias):
            return mha_reference(
                query, key, value, bias=bias, causal=causal, softmax_scale=softmax_scale
            )

        return jit_attn

    else:
        raise NotImplementedError(f"Backend ({backend}) does not have an implementation.")
