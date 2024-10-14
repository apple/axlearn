# Copyright © 2023 Apple Inc.

"""FlashAttention utilities shared amongst CPU/GPU/TPU backends."""
import functools
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from absl import logging

from axlearn.common.attention import NEG_INF, MaskFn, causal_mask, softmax_with_biases
from axlearn.common.flash_attention.gpu_attention import cudnn_dot_product_attention
from axlearn.common.flash_attention.gpu_attention import flash_attention as gpu_flash_attention
from axlearn.common.flash_attention.tpu_attention import tpu_flash_attention
from axlearn.common.utils import Tensor


@functools.partial(jax.jit, static_argnames=["causal", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Optional[Tensor] = None,
    segment_ids: Optional[Tensor] = None,
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
) -> Tensor:
    """Reference multi-headed attention implementation.

    Args:
        q: query tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        k: key tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        v: value tensor with shape [batch_size, seq_len, num_heads, per_head_dim]
        bias: bias tensor with a shape that can broadcast to
            [batch_size, num_heads, seq_len, seq_len], e.g. [1, 1, seq_len, seq_len].
        segment_ids: segment ids tensor with shape [batch_size, seq_len].
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
    if segment_ids is not None:
        assert segment_ids.ndim == 2  # shape [batch_size, seq_len]
        target_segment_ids = jnp.expand_dims(segment_ids, -1)
        source_segment_ids = jnp.expand_dims(segment_ids, -2)
        # Target [b..., t] + Source [b..., s] -> [b..., t, s]
        # [b, 1, ..., t, s] where the value at [..., i, j] = false if
        # target_segments[..., i] == source_segments[..., j], or true otherwise.
        mask = jax.lax.ne(source_segment_ids, target_segment_ids)[:, None, ...]
        logits = jnp.where(mask, NEG_INF, logits)

    if causal:
        mask_shape = (q.shape[1], k.shape[1])
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        mask = (row_ids < col_ids)[None, None, :, :]  # Causal mask.
        logits = jnp.where(mask, NEG_INF, logits)

    probs = softmax_with_biases(logits, bias)
    context = jnp.einsum("bnts,bsnh->btnh", probs, v).astype(v.dtype)
    return context


# Accepts [query, key, value, attention_bias, segment_ids] tensors and returns the context Tensor.
MultiHeadAttentionImpl = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


def flash_attention_implementation(
    backend: Literal["cpu", "tpu", "gpu", "xla"],
    *,
    mask: Optional[MaskFn] = None,
    softmax_scale: float,
    block_size: int = 128,
) -> MultiHeadAttentionImpl:
    """Returns a jitted "flash" multihead-attention implementation for the given backend.

    Args:
        backend: A valid XLA backend name. 'cpu' intended for testing only.
        mask: A mask to use when computing the attention. This allows for more efficient
            computation than setting bias = -inf on certain backends.
        softmax_scale: A scalar value applied to the logits before softmax.
        block_size: The size of the computation-block unit, only applies to the 'tpu' backend.
            A multiple of 128, and should be less than the target sequence length.
            Smaller values are more memory efficient but less compute efficient.

    Returns:
        A jitted function implementing multi-head attention for the given backend.

    Raises:
        NotImplementedError: If implementation for the backend is not available.
    """
    causal = mask is causal_mask
    if mask is not None and not causal and backend != "tpu":
        raise NotImplementedError(
            "Custom (non-causal, non-full) mask only supported on TPU.\n"
            "You can use NEG_INF biases instead, but it won't "
            "have the sparsity optimizations."
        )
    if backend == "gpu":
        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias, segment_ids):
            # Fall back to triton gpu kernel if:
            # - segment_ids is not None,
            # - bias is not None,
            # - query/key/value are in float32.
            if (
                segment_ids is not None
                or bias is not None
                or jnp.float32 in (query.dtype, key.dtype, value.dtype)
            ):
                logging.warning("Flash attention falling back to Triton GPU kernel.")
                return gpu_flash_attention(
                    query,
                    key,
                    value,
                    bias=bias,
                    segment_ids=segment_ids,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                return cudnn_dot_product_attention(
                    query,
                    key,
                    value,
                    bias=bias,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    dropout_rate=0.0,
                )

        return jit_attn

    elif backend == "tpu":
        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias, segment_ids):
            context = tpu_flash_attention(
                query,
                key,
                value,
                bias=bias,
                segment_ids=segment_ids,
                mask=mask,
                softmax_scale=softmax_scale,
                block_size=block_size,
            )
            return context

        return jit_attn

    elif backend in ("cpu", "xla"):
        if backend == "cpu":
            logging.warning("Flash attention CPU backend is for testing only.")
        logging.warning("Flash attention falling back using plain MHA implementation")

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_attn(query, key, value, bias, segment_ids):
            return mha_reference(
                query,
                key,
                value,
                bias=bias,
                segment_ids=segment_ids,
                causal=causal,
                softmax_scale=softmax_scale,
            )

        return jit_attn

    else:
        raise NotImplementedError(f"Backend ({backend}) does not have an implementation.")
