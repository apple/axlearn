# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
from typing import Optional

import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.flash_attention import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    NUM_LANES,
    NUM_SUBLANES,
)
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes as LegacyBlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import (
    SegmentIds,
    _flash_attention_dkv_kernel,
    _flash_attention_dq_kernel,
    _flash_attention_kernel,
    _verify_block,
    below_or_on_diag,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)

from axlearn.common.attention import apply_attention_logit_biases
from axlearn.common.attention_bias import (
    CausalAttentionBias,
    MaskFnAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    as_attention_bias,
)
from axlearn.common.flash_attention.remat import FLASH_ATTN_RESIDUAL_NAME
from axlearn.common.utils import Tensor

MaskFnOrZero = MaskFnAttentionBias | ZeroAttentionBias


def tpu_flash_attention(
    query: Tensor,  # [batch_size, target_len, num_heads, head_dim]
    key: Tensor,  # [batch_size, source_len, num_heads, head_dim]
    value: Tensor,  # [batch_size, source_len, num_heads, head_dim]
    bias: Tensor = None,  # [batch_size, num_heads, target_len, source_len]
    segment_ids: Tensor = None,  # [batch_size, target_len]
    *,
    mask: MaskFnOrZero,
    softmax_scale: float = 1.0,
    is_decoding: bool = False,
    block_size: int = 128,
    interpret: bool = False,
):
    """Wraps JAX's TPU flash-attention, with reshapes and softmax-scaling outside kernel.

    An implementation is automatically selected based on the arguments.

    N.B. we apply the softmax scale factor outside of the kernel because:
        1. within-kernel ordering of attention-bias addition and softmax scaling differ to axlearn,
        2. it's more efficient to scale outside the kernel vs. fix order of ops in kernel.

    If provided, bias, segment_ids, and mask are applied on top of one another.

    Args:
        query: The query tensor, of shape [batch_size, target_len, num_heads, head_dim].
        key: The key tensor, of shape [batch_size, source_len, num_heads, head_dim].
        value: The value tensor, of shape [batch_size, source_len, num_heads, head_dim].
        bias: The attention biases, can broadcast to shape
            [batch_size, num_heads, target_len, source_len].
        segment_ids: The id of which segment each token belongs to. Attention is not computed
             between tokens in different segments.
             Shape:  [batch_size, target_len].
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        softmax_scale: A scaling factor applied to the query.
        is_decoding: Whether it is in decoding.
        block_size: The block size to use for chunking data in the kernel.
        interpret: If True, interpret the kernel using the pallas interpreter. CPU needs it.

    Returns:
        The context tensor, of shape [batch_size, target_len, num_heads, head_dim].

    Raises:
        NotImplementedError: If no implementation with support for the arguments is found.
        ValueError: If the head_dim of the query, key, and value are not all equal.
        ValueError: if the target or source sequence length is not divisible by block_size.`
    """
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

    if query.shape[1] % block_size != 0:
        raise ValueError(
            f"Target seq len {query.shape[1]} must be divisible by block size {block_size}."
        )
    if key.shape[1] % block_size != 0:
        raise ValueError(
            f"Source seq len {key.shape[1]} must be divisible by block size {block_size}."
        )

    mask: MaskFnOrZero = as_attention_bias(mask)

    # Switch num_heads and seq_len axes.
    query = jnp.einsum("btnh->bnth", query)
    key = jnp.einsum("bsnh->bnsh", key)
    value = jnp.einsum("bsnh->bnsh", value)
    try:
        check_tpu_splash_attention(
            target_len=query.shape[2],
            source_len=key.shape[2],
            head_dim=query.shape[3],
            mask=mask,
            is_decoding=is_decoding,
            has_segment_ids=(segment_ids is not None),
            has_bias=(bias is not None),
        )
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
        splash_mask = _to_splash_mask(mask, mask_shape=(query.shape[2], key.shape[2]))
        context = _tpu_splash_attention(
            query,
            key,
            value,
            splash_mask=splash_mask,
            segment_ids=segment_ids,
            block_sizes=block_sizes,
            interpret=interpret,
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
        causal = isinstance(mask, CausalAttentionBias)
        if not causal and mask.has_value():
            bias = apply_attention_logit_biases(mask.value(), bias)
        context = _legacy_tpu_flash_attention(
            query,
            key,
            value,
            bias,
            segment_ids=segment_ids,
            causal=causal,
            block_sizes=block_sizes,
            interpret=interpret,
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
        "causal",
        "block_sizes",
        "interpret",
    ],
)
def _legacy_tpu_flash_attention(
    query: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    key: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    value: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    bias: Tensor = None,  # [batch_size, num_heads, target_len, source_len]
    segment_ids: Tensor = None,  # [batch_size, target_len]
    *,
    causal: bool = False,
    block_sizes: Optional[LegacyBlockSizes] = None,
    interpret: bool = False,
) -> Tensor:  # [batch_size, num_heads, target_len, head_dim].
    """Wraps JAX's legacy TPU flash-attention.

    If provided, bias, segment_ids, and mask are applied on top of one another.

    Args:
        query: The query tensor, of shape [batch_size, num_heads, target_len, head_dim].
        key: The key tensor, of shape [batch_size, num_heads, source_len, head_dim].
        value: The value tensor, of shape [batch_size, num_heads, source_len, head_dim].
        bias: The attention biases, of shape [batch_size, num_heads, target_len, source_len].
        segment_ids: The id of which segment each token belongs to. Attention is not computed
             between tokens in different segments.
             Shape:  [batch_size, target_len].
        causal: Whether it's causal attention.
        block_sizes: An object containing values that can be used to tune the performance
            such as the block size to chunk things into.
        interpret: If True, interpret the kernel using the pallas interpreter. CPU needs it.

    Returns:
        The context tensor, of shape [batch_size, num_heads, target_len, head_dim].

    Raises:
        NotImplementedError: If a custom (non-causal, non-full) mask is specified.
    """
    context = pallas_tpu_flash_attention(
        q=query,
        k=key,
        v=value,
        ab=bias,
        segment_ids=SegmentIds(q=segment_ids, kv=segment_ids) if segment_ids is not None else None,
        causal=causal,
        # If softmax_scale==1.0, the kernel skips applying it.
        softmax_scale=1.0,
        block_sizes=block_sizes,
        debug=False,
        interpret=interpret,
    )
    return context


class SplashAttentionUnsupportedError(NotImplementedError):
    """An error indicating splash attention is not supported."""


def check_tpu_splash_attention(
    *,
    target_len: int,
    source_len: int,
    head_dim: int,
    mask: MaskFnOrZero,
    is_decoding: bool = False,
    has_segment_ids: bool = False,
    has_bias: bool = False,
):
    """Checks if splash attention is supported on TPU for the given arguments.

    Args:
        target_len: The length of the target sequence.
        source_len: The length of the source sequence.
        head_dim: The dimension of each head.
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        is_decoding: Whether it is in decoding.
        has_segment_ids: Whether segment_ids is None or not.
        has_bias: Whether attention involves a bias.

    Raises:
        SplashAttentionUnsupportedError: If splash attention is not supported for the given
            arguments.
    """
    if has_bias:
        raise SplashAttentionUnsupportedError("SplashAttention does not support specifying a bias.")
    with jax.ensure_compile_time_eval():
        if jnp.any(
            jnp.asarray([target_len, source_len, head_dim]) % splash_attention_kernel.NUM_LANES != 0
        ):
            raise SplashAttentionUnsupportedError(
                "SplashAttention requires target_len, source_len, head_dim are divisible by"
                f" {splash_attention_kernel.NUM_LANES}, got {target_len, source_len, head_dim}."
            )
    if has_segment_ids:
        raise SplashAttentionUnsupportedError(
            "The public API for SplashAttention that we "
            "currently use does not support segment ids."
        )
    if mask.has_value():
        assert isinstance(mask, MaskFnAttentionBias)
        if not isinstance(mask, (CausalAttentionBias, SlidingWindowAttentionBias)):
            raise SplashAttentionUnsupportedError(f"{mask=} is not supported.")
        if is_decoding:
            # TODO(dhwang2): support splash decoding via splash NumPy mask.
            raise SplashAttentionUnsupportedError(
                "Query and key/value must have same length when mask is used."
            )
        else:
            if target_len != source_len:
                raise ValueError(f"{target_len=} and {source_len=} must be same in forward().")


def _to_splash_mask(
    mask: MaskFnOrZero,
    *,
    mask_shape: tuple[int, int],
    q_seq_shards: int = 1,
) -> splash_attention_mask.Mask:
    """Converts a mask to a splash mask."""
    if not mask.has_value():
        return splash_attention_mask.FullMask(mask_shape)
    assert isinstance(mask, MaskFnAttentionBias)
    if isinstance(mask, CausalAttentionBias):
        return splash_attention_mask.CausalMask(shape=mask_shape, shard_count=q_seq_shards)
    elif isinstance(mask, SlidingWindowAttentionBias):
        left_size = mask.sliding_window_size
        return splash_attention_mask.LocalMask(
            shape=mask_shape, window_size=(left_size, 0), offset=0, shard_count=q_seq_shards
        )

    with jax.ensure_compile_time_eval():
        # MaskFn always supports compile time eval.
        mask_array = np.asarray(mask.bool_value())
        # Squeeze first two leading dimensions.
        mask_array = mask_array.reshape(mask_array.shape[-2:])

    # NumpyMask is backed by a dense [target_len, source_len] numpy array.
    # May consume a large amount of host memory for long sequences at compile time.
    return splash_attention_mask.NumpyMask(array=mask_array)


@functools.partial(
    jax.jit,
    static_argnames=["splash_mask", "block_sizes", "interpret"],
)
def _tpu_splash_attention(
    query: Tensor,  # [batch_size, num_heads, target_len, head_dim]
    key: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    value: Tensor,  # [batch_size, num_heads, source_len, head_dim]
    *,
    splash_mask: splash_attention_mask.Mask,
    segment_ids: Optional[Tensor] = None,  # [batch_size, target_len]
    block_sizes: Optional[splash_attention_kernel.BlockSizes] = None,
    interpret: bool = False,
) -> Tensor:  # [batch_size, num_heads, target_len, head_dim].
    """Wraps JAX's sparse TPU flash-attention.

    Args:
        query: The query tensor, of shape [batch_size, num_heads, target_len, head_dim].
        key: The key tensor, of shape [batch_size, num_heads, source_len, head_dim].
        value: The value tensor, of shape [batch_size, num_heads, source_len, head_dim].
        mask: The mask to apply. This is more compute efficient compared to setting bias = -inf.
        segment_ids: The id of which segment each token belongs to. Attention is not computed
            between tokens in different segments, [batch_size, target_len].
        block_sizes: An object containing values that can be used to tune the performance
            such as the block size to chunk things into.
        interpret: If True, interpret the kernel using the pallas interpreter. CPU needs it.

    Returns:
        The context tensor, of shape [batch_size, num_heads, target_len, head_dim].

    Raises:
        SplashAttentionUnsupportedError: If splash attention does not support the given arguments.
            This happens if any of the following is true:
            - bias is not None.
            - The per_head_dim is not divisible by 128.
            - segment_ids is not None.
            - The source and target lengths are different and a nonzero mask is used.
        TypeError: If mask is not an instance of `MaskFnAttentionBias.
    """

    # TODO(dhwang2): splash attention can support segment_ids. Support it when needed.
    del segment_ids
    num_heads = query.shape[1]
    kernel = splash_attention_kernel.make_splash_mha(
        mask=splash_attention_mask.MultiHeadMask(masks=[splash_mask] * num_heads),
        block_sizes=block_sizes,
        # TODO(dhwang2): support "seq" and "model" shard.
        head_shards=1,
        q_seq_shards=1,
        interpret=interpret,
        residual_checkpoint_name=f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}",
    )
    kernel = jax.vmap(kernel)
    context = kernel(q=query, k=key, v=value)
    return context


# The following code is adapted from jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "softmax_scale",
        "block_sizes",
        "debug",
        "interpret",
    ],
)
def pallas_tpu_flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len] or [1, 1, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
    block_sizes: Optional[LegacyBlockSizes] = None,
    debug: bool = False,
    interpret: bool = False,
):
    batch_size, num_heads, q_seq_len, d_model = q.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
            f" {batch_size_v} (for q, k, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k},"
            f" {num_heads_v} (for q, k, v respectively)"
        )
    if d_model != d_model_k:
        raise ValueError(
            f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k" " respectively)"
        )
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if ab is not None:
        if ab.shape not in [
            (batch_size, num_heads, q_seq_len, kv_seq_len),
            (batch_size, 1, q_seq_len, kv_seq_len),
            (1, num_heads, q_seq_len, kv_seq_len),
            (1, 1, q_seq_len, kv_seq_len),
        ]:
            raise ValueError(
                f"Attention bias shape mismatch: expected to broadcast ({batch_size=},"
                f" {num_heads=}, {q_seq_len=}, {kv_seq_len=})"
            )
    if segment_ids is not None:
        if segment_ids.q.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=},"
                f" {q_seq_len=},), got {segment_ids.q.shape}"
            )
        if segment_ids.kv.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=},"
                f" {kv_seq_len=},), got {segment_ids.kv.shape}"
            )
    if block_sizes is None:
        block_sizes = LegacyBlockSizes.get_default(
            batch_size, num_heads, q_seq_len, kv_seq_len, d_model
        )
    return _flash_attention(
        q, k, v, ab, segment_ids, causal, softmax_scale, block_sizes, debug, interpret
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    causal,
    softmax_scale,
    block_sizes,
    debug,
    interpret,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        False,
        causal,
        softmax_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
        interpret,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    causal,
    softmax_scale,
    block_sizes,
    debug,
    interpret,
):
    o, l, m = _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        True,
        causal,
        softmax_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
        interpret,
    )
    return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_bwd(
    causal: bool,
    softmax_scale: float,
    block_sizes: LegacyBlockSizes,
    debug: bool,
    interpret: bool,
    residuals,
    do,
):
    """VJP rule for FlashAttention."""
    (q, k, v, ab, segment_ids, o, l, m) = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError(
            "Program is being differentiated, but not all backward blocks are" " specified"
        )

    di = jnp.sum(
        o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1
    )  # [batch_size, num_heads, q_seq_len]

    dk, dv = _flash_attention_bwd_dkv(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,
        block_k_major=block_sizes.block_k_major_dkv,
        block_k=block_sizes.block_k_dkv,
        block_q=block_sizes.block_q_dkv,
        softmax_scale=softmax_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
        interpret=interpret,
    )

    dq, ds = _flash_attention_bwd_dq(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,
        block_k_major=block_sizes.block_k_major_dq,
        block_k=block_sizes.block_k_dq,
        softmax_scale=softmax_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
        interpret=interpret,
    )
    return dq, dk, dv, ds, None


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
    interpret,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    # TODO(apaszke): Tile over heads as well.
    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major)
            # If the ab block is skipped, prefetch the next valid ab block, i.e. the
            # 0th kv to be used for the next block_q rows.
            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index
        return (
            batch_index if ab.shape[0] != 1 else 0,
            head_index if ab.shape[1] != 1 else 0,
            next_q_index,
            next_kv_index,
        )

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=softmax_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        m = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = (
        pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map) if ab is not None else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec(
            (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        compiler_params=dict(
            mosaic=dict(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        o = jax.ad_checkpoint.checkpoint_name(o, f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}")
        l = jax.ad_checkpoint.checkpoint_name(l, f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}")
        m = jax.ad_checkpoint.checkpoint_name(m, f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}")
        return (o, l, m)
    else:
        return o


def _flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: Optional[int],
    block_q: Optional[int],
    block_k_major: Optional[int],
    block_k: Optional[int],
    softmax_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
    interpret: bool = False,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    # kv index needs to be before q index since q index is the contractng
    # dimension.
    grid = (
        batch_size,
        num_heads,
        kv_seq_len // block_k_major,
        q_seq_len // block_q_major,
    )

    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        if causal:
            # If the q block is skipped, stay at the 0th q block.
            next_q_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                q_seq_index,
                0,
            )
        else:
            next_q_index = q_seq_index

        return (batch_index, head_index, next_q_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, _, q_seq_index):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        return (
            batch_index if ab.shape[0] != 1 else 0,
            head_index if ab.shape[1] != 1 else 0,
            q_seq_index,
            kv_seq_index,
        )

    dab_spec = (
        pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
            del head_index
            if causal:
                next_q_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    q_seq_index,
                    0,
                )
            else:
                next_q_index = q_seq_index
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec(
            (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), k.dtype),
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), v.dtype),
    ]

    def dkv_index_map(batch_index, head_index, kv_seq_index, _):
        return (batch_index, head_index, kv_seq_index, 0)

    dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
    out_specs = [dkv_spec, dkv_spec]
    scratch_shapes = [
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        block_q=block_q,
        block_k=block_k,
        sm_scale=softmax_scale,
        causal=causal,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
    )
    name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dk, dv = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            interpret=interpret,
            compiler_params=dict(
                mosaic=dict(
                    dimension_semantics=(
                        "parallel",
                        "parallel",
                        "parallel",
                        "arbitrary",
                    )
                )
            ),
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: Optional[int],
    block_k_major: Optional[int],
    block_k: Optional[int],
    softmax_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
    interpret: bool,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_seq_len // block_k_major,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (
            batch_index if ab.shape[0] != 1 else 0,
            head_index if ab.shape[1] != 1 else 0,
            q_seq_index,
            kv_seq_index,
        )

    dab_spec = (
        pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                # If the kv block is skipped, prefetch the next valid kv block, i.e. the
                # 0th one to be used for the next block_q rows.
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec(
            (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    out_specs = [
        dq_spec,
        dab_spec,
    ]
    scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        sm_scale=softmax_scale,
        causal=causal,
        mask_value=mask_value,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, ds = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            interpret=interpret,
            compiler_params=dict(
                mosaic=dict(
                    dimension_semantics=(
                        "parallel",
                        "parallel",
                        "parallel",
                        "arbitrary",
                    )
                )
            ),
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

    # dab is just ds
    return dq, ds
