# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
from typing import Any, Callable, Hashable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes as LegacyBlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as pallas_tpu_flash_attention,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)

from axlearn.common.attention import MaskFn, causal_mask
from axlearn.common.utils import Tensor


def tpu_flash_attention(
    query: Tensor,  # [batch_size, source_len, num_heads, head_dim]
    key: Tensor,  # [batch_size, target_len, num_heads, head_dim]
    value: Tensor,  # [batch_size, target_len, num_heads, head_dim]
    bias: Tensor = None,  # [batch_size, num_heads, source_len, target_len]
    segment_ids: Tensor = None,  # [batch_size, source_len]
    *,
    mask: Optional[MaskFn] = None,
    softmax_scale: float = 1.0,
    block_size: int = 128,
):
    """Wraps JAX's TPU flash-attention, with reshapes and softmax-scaling outside kernel.

    An implementation is automatically selected based on the arguments.

    N.B. we apply the softmax scale factor outside of the kernel because:
        1. within-kernel ordering of attention-bias addition and softmax scaling differ to axlearn,
        2. it's more efficient to scale outside the kernel vs. fix order of ops in kernel.

    Args:
        query: The query tensor, of shape [batch_size, source_len, num_heads, head_dim].
        key: The key tensor, of shape [batch_size, target_len, num_heads, head_dim].
        value: The value tensor, of shape [batch_size, target_len, num_heads, head_dim].
        bias: The attention biases, of shape [batch_size, num_heads, source_len, target_len].
        segment_ids: The id of which segment each token belongs to. Attention is not computed
             between tokens in different segments.
             Shape:  [batch_size, source_len].
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        softmax_scale: A scaling factor applied to the query.
        block_size: The block size to use for chunking data in the kernel.

    Returns:
        The context tensor, of shape [batch_size, source_len, num_heads, head_dim].

    Raises:
        NotImplementedError: If no implementation with support for the arguments is found.
        ValueError: If the head_dim of the query, key, and value are not all equal."""
    if segment_ids is not None:
        assert query.shape[1] == key.shape[1] and query.shape[1] == value.shape[1]
    # Apply the softmax scale outside the kernel (see docstring for why).
    if softmax_scale != 1.0:
        query *= softmax_scale

    head_dim = query.shape[3]

    if head_dim != key.shape[3]:
        raise ValueError(
            f"Per-head dimension of query doesn't equal that of key: "
            f"{head_dim} != {key.shape[3]}"
        )
    if head_dim != value.shape[3]:
        raise ValueError(
            f"Per-head dimension of query doesn't equal that of value: "
            f"{head_dim} != {value.shape[3]}"
        )

    # Switch num_heads and seq_len axes.
    query = jnp.einsum("btnh->bnth", query)
    key = jnp.einsum("bsnh->bnsh", key)
    value = jnp.einsum("bsnh->bnsh", value)
    try:
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=block_size,
            block_kv=block_size,
            block_kv_compute=block_size,
            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_kv_dkv_compute=block_size,
            # The fused kernel is neutral in small models and a ~5%-15% improvement in larger ones.
            # E.g., 1.03x speedup in a 12.6b simulated model, 1.06x speedup in 29.6b ,
            # and 1.14x in 539.5b.
            use_fused_bwd_kernel=True,
        )
        context = _tpu_splash_attention(
            query, key, value, bias, segment_ids=segment_ids, mask=mask, block_sizes=block_sizes
        )
        logging.info("Using SplashAttention.")
    except SplashAttentionUnsupportedError as e:
        # TODO(tom_gunter): See if we can do better block-size tuning.
        block_sizes = LegacyBlockSizes(
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
        context = _legacy_tpu_flash_attention(
            query,
            key,
            value,
            bias,
            segment_ids=segment_ids,
            mask=mask,
            block_sizes=block_sizes,
        )
        logging.warning(
            "Falling back to legacy flash attention because SplashAttention is not supported.\n"
            "Reason: %s",
            e,
        )

    # Restore num_heads and seq_len axes.
    return jnp.einsum("bnth->btnh", context)


@functools.partial(
    jax.jit,
    static_argnames=[
        "mask",  # Mask objects don't actually contain jax arrays, so they are static.
        "block_sizes",
    ],
)
def _legacy_tpu_flash_attention(
    query: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    key: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    value: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    bias: Tensor = None,  # [batch_size, num_heads, source_len, target_len]
    segment_ids: Tensor = None,  # [batch_size, source_len]
    *,
    mask: Optional[MaskFn] = None,
    block_sizes: Optional[LegacyBlockSizes] = None,
) -> Tensor:  # [batch_size, num_heads, source_len, head_dim].
    """Wraps JAX's legacy TPU flash-attention.

    Args:
        query: The query tensor, of shape [batch_size, num_heads, source_len, head_dim].
        key: The key tensor, of shape [batch_size, num_heads, target_len, head_dim].
        value: The value tensor, of shape [batch_size, num_heads, source_len, head_dim].
        bias: The attention biases, of shape [batch_size, num_heads, source_len, target_len].
        segment_ids: The id of which segment each token belongs to. Attention is not computed
             between tokens in different segments.
             Shape:  [batch_size, source_len].
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        block_sizes: An object containing values that can be used to tune the performance
            such as the block size to chunk things into.

    Returns:
        The context tensor, of shape [batch_size, num_heads, source_len, head_dim].

    Raises:
        NotImplementedError: If a custom (non-causal, non-full) mask is specified.
    """
    if mask is not None and mask is not causal_mask:
        raise NotImplementedError("Custom masks are not supported by legacy attention.")
    causal = mask is causal_mask
    context = pallas_tpu_flash_attention(
        q=query,
        k=key,
        v=value,
        ab=bias,
        segment_ids=SegmentIds(q=segment_ids, kv=segment_ids) if segment_ids is not None else None,
        causal=causal,
        # If sm_scale==1.0, the kernel skips applying it.
        sm_scale=1.0,
        block_sizes=block_sizes,
        debug=False,
    )

    return context


class SplashAttentionUnsupportedError(NotImplementedError):
    """An error indicating splash attention is not supported."""


@functools.partial(
    jax.jit,
    static_argnames=[
        "mask",  # Mask objects don't actually contain jax arrays, so they are static.
        "block_sizes",
    ],
)
def _tpu_splash_attention(
    query: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    key: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    value: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    bias: Tensor = None,  # [batch_size, num_heads, source_len, target_len]
    segment_ids: Tensor = None,  # [batch_size, source_len]
    *,
    mask: Optional[MaskFn] = None,
    block_sizes: Optional[splash_attention_kernel.BlockSizes] = None,
) -> Tensor:  # [batch_size, num_heads, source_len, head_dim].
    """Wraps JAX's sparse TPU flash-attention.

    Args:
        query: The query tensor, of shape [batch_size, num_heads, source_len, head_dim].
        key: The key tensor, of shape [batch_size, num_heads, target_len, head_dim].
        value: The value tensor, of shape [batch_size, num_heads, source_len, head_dim].
        bias: The attention biases, of shape [batch_size, num_heads, source_len, target_len].
        segment_ids: The id of which segment each token belongs to. Attention is not computed
            between tokens in different segments.
             Shape:  [batch_size, source_len].
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        block_sizes: An object containing values that can be used to tune the performance
            such as the block size to chunk things into.

    Returns:
        The context tensor, of shape [batch_size, num_heads, source_len, head_dim].

    Raises:
        NotImplementedError: If a bias is also specified or the head_dim is not divisible by
            128.
    """

    source_len = query.shape[2]
    target_len = key.shape[2]
    num_heads = query.shape[1]
    head_dim = query.shape[3]

    if bias is not None:
        raise SplashAttentionUnsupportedError("SplashAttention does not support specifying a bias.")
    if head_dim % splash_attention_kernel.NUM_LANES != 0:
        raise SplashAttentionUnsupportedError(
            "SplashAttention requires "
            f"head_dim=={splash_attention_kernel.NUM_LANES}, "
            f"got {head_dim}."
        )
    if segment_ids is not None:
        raise SplashAttentionUnsupportedError(
            "The public API for SplashAttention that we "
            "currently use does not support segment ids."
        )

    mask_shape = (source_len, target_len)
    if mask is None:
        mask = splash_attention_mask.FullMask(mask_shape)
    else:

        def wrap_mask(mask: MaskFn) -> MaskFn:
            """Wrap `mask` so that the return type is a numpy array
            if the original input was, even if we are inside of jit.
            """

            def wrapped_mask(*args, **kwargs) -> Union[np.ndarray, Tensor]:
                if all(
                    isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves([args, kwargs])
                ):
                    with jax.ensure_compile_time_eval():
                        result = mask(*args, **kwargs)
                        return jax.tree_util.tree_map(np.asarray, result)
                return mask(*args, **kwargs)

            return wrapped_mask

        mask = ComputableMask(mask_shape, mask_function=wrap_mask(mask))

    kernel = splash_attention_kernel.make_splash_mha(
        mask=splash_attention_mask.MultiHeadMask(masks=[mask] * num_heads),
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
    )
    kernel = jax.vmap(kernel)
    context = kernel(q=query, k=key, v=value)
    return context


class ComputableMask(splash_attention_mask.Mask):
    """A Mask that can be lazily computed using a callable.

    This implementation is mostly copied from

    `jax.experimental.pallas.ops.flash_attention.splash_attention_mask._ComputableMask`

    in order to avoid relying on that private API.
    """

    # The shape of the mask.
    _shape: tuple[int, int]
    # The sequence of query indices that the mask covers.
    q_sequence: np.ndarray
    # A function compute the mask value given indices.
    mask_function: MaskFn

    def __init__(
        self,
        shape: tuple[int, int],
        mask_function: Callable[..., Any],
        shard_count: int = 1,
    ):
        self._shape = shape
        self.mask_function = mask_function
        source_len = self.shape[0]

        if source_len % (shard_count * shard_count) != 0:
            raise ValueError(
                f"Shard count squared ({shard_count * shard_count}) must"
                f" divide Q seq_len ({self.shape[0]}) evenly."
            )

        self.q_sequence = np.arange(source_len, dtype=np.int32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        """Return the entries of the mask specified by the row and column slice in idx."""
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_slice, kv_slice = idx
        if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")

        def _fill_slice(inp_slice: slice, size: int) -> slice:
            assert inp_slice.step is None or inp_slice.step == 1
            start = 0 if inp_slice.start is None else inp_slice.start
            stop = size if inp_slice.stop is None else inp_slice.stop
            assert start >= 0
            assert stop <= size
            return slice(start, stop, None)

        q_slice = _fill_slice(q_slice, self.shape[0])
        kv_slice = _fill_slice(kv_slice, self.shape[1])

        rows = self.q_sequence[q_slice]
        cols = np.arange(kv_slice.start, kv_slice.stop)

        return self.mask_function(rows[:, None], cols[None, :])

    def _to_hashable(self) -> Hashable:
        """Returns a hashable representation of this object that can be used for equality
        comparisons.
        """
        return (
            type(self),
            self.shape,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
            self.mask_function,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComputableMask):
            raise NotImplementedError
        return self._to_hashable() == other._to_hashable()

    def __hash__(self) -> int:
        return hash(self._to_hashable())
