# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
import logging
from typing import Optional

import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
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
from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds as LegacySegmentIds
from jax.experimental.pallas.ops.tpu.flash_attention import (
    _flash_attention_dkv_kernel,
    _flash_attention_dq_kernel,
    _flash_attention_kernel,
    _verify_block,
    below_or_on_diag,
)
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

import axlearn.common.flash_attention.tpu_splash_attention as splash_attention_kernel
from axlearn.common.attention_bias import (
    BaseAttentionBias,
    CausalAttentionBias,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    split,
)
from axlearn.common.flash_attention.common import (
    BaseFlashAttention,
    get_segment_ids,
    repeat_kv_heads,
)
from axlearn.common.flash_attention.remat import FLASH_ATTN_RESIDUAL_NAME
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.utils import Nested, Tensor

MaskFnOrZero = MaskFnAttentionBias | ZeroAttentionBias


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

    # Because mask.mask() may use jnp ops. e.g. jnp.logical_and.
    with jax.ensure_compile_time_eval():
        # This code is reached only when `kv_cache_type=None` (i.e., forward and prefill) and
        # `target_len == source_len` (i.e., self-attention) (see `check_tpu_splash_attention`).
        # `target_positions` and `source_positions` are always in the range [0, seq_len].
        target_positions = np.arange(mask_shape[0])[None, :, None]
        source_positions = np.arange(mask_shape[1])[None, None, :]
        # `mask.mask` expects rank 3 tensors.
        mask_array = np.asarray(mask.mask(target_positions, source_positions))
        mask_array = np.squeeze(mask_array, axis=0)

    # NumpyMask is backed by a dense [target_len, source_len] numpy array.
    # May consume a large amount of host memory for long sequences at compile time.
    return splash_attention_mask.NumpyMask(array=mask_array)


# The following code is adapted from jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
def _pallas_tpu_flash_attention(
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
    _, _, kv_seq_len, _ = k.shape
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


class TPUFlashAttention(BaseFlashAttention):
    """Wraps the common checks for TPU attention implementations."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch=input_batch, kv_cache_type=kv_cache_type):
            return False
        block_size = self.cfg.tpu_block_size
        if not self._check_block_size(input_batch=input_batch, block_size=block_size):
            return False
        return True


class TPUSplashAttention(TPUFlashAttention):
    """Wraps SplashAttention.

    This kernel should be used for majority of the cases, except when
    1. explicit bias is used.
    2. head_dim is not a multiple of 128.

    In these two cases, we fallback to the legacy implementation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._use_fused = True

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch, kv_cache_type=kv_cache_type):
            return False
        bias: BaseAttentionBias = input_batch["bias"]
        _, _, explicit_bias = split(bias, MaskFnAttentionBias, SegmentIdAttentionBias)
        query: Tensor = input_batch["query"]
        head_dim = query.shape[-1]

        if explicit_bias.has_value():
            return self._log_unsupported("explicit bias is not supported.")

        if head_dim % splash_attention_kernel.NUM_LANES != 0:
            return self._log_unsupported(
                f"{head_dim=} is not divisible by {splash_attention_kernel.NUM_LANES=}"
            )

        if (
            not self.get_backend_overrides("splash_use_fused_bwd_kernel", True)
            and self.cfg.dropout_rate > 0.0
        ):
            # TODO (bailin): Support dropout with non-fused bwd kernel.
            return self._log_unsupported("dropout with non-fused bwd kernel is not supported.")

        # If user doesn't specify splash_use_fused_bwd_kernel, we have some defaults
        # or heuristics to decide whether to use fused bwd kernel.
        if (
            not self.cfg.backend_overrides
            or "splash_use_fused_bwd_kernel" not in self.cfg.backend_overrides
        ):
            # When dropout is enabled, we always use the fused bwd kernel.
            if self.cfg.dropout_rate > 0.0:
                self._use_fused = True
            else:
                # Heuristic for sliding window attention.
                sliding, _ = split(bias, SlidingWindowAttentionBias)
                key: Tensor = input_batch["key"]
                kv_seq_len = key.shape[1]
                # TODO(c_lan): Support logit_sinks for non-fused bwd kernel.
                if sliding.has_value() and "logit_sinks" not in input_batch:
                    if kv_seq_len >= 16 * 1024 and kv_seq_len / sliding.sliding_window_size >= 4.0:
                        logging.info(
                            "Not using fused kernel for splash attention backward pass for better "
                            "performance, because sliding_window_size=%d << kv_seq_len=%d.",
                            sliding.sliding_window_size,
                            kv_seq_len,
                        )
                        self._use_fused = False
        else:
            self._use_fused = self.get_backend_overrides("splash_use_fused_bwd_kernel", True)

        return True

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BaseFlashAttention.__call__`."""
        cfg = self.config
        bias: BaseAttentionBias = input_batch["bias"]
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        logit_sink: Optional[Tensor] = input_batch.get("logit_sink", None)
        prng_key = input_batch.get("prng_key", None)

        if cfg.dropout_rate > 0.0 and prng_key is None:
            raise ValueError(
                "TPU SplashAttention requires a prng_key to be provided when dropout is enabled."
            )

        mask, segment_ids, _ = split(bias, MaskFnAttentionBias, SegmentIdAttentionBias)
        segment_id_tensor = get_segment_ids(query=query, key=key, segment_ids=segment_ids)
        seg_ids = None
        if segment_id_tensor is not None:
            seg_ids = SplashSegmentIds(q=segment_id_tensor, kv=segment_id_tensor)

        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query) * self.cfg.softmax_scale
        key = jnp.einsum("bsnh->bnsh", key)
        value = jnp.einsum("bsnh->bnsh", value)

        block_size = self.cfg.tpu_block_size
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=self.get_backend_overrides("splash_block_q", block_size),
            block_kv=self.get_backend_overrides("splash_block_kv", block_size),
            block_kv_compute=self.get_backend_overrides("splash_block_kv_compute", block_size),
            # When fused bwd kernel is used, dq and dk/dv are computed in the same kernel. Only
            # *dkv* block sizes are used. When fused bwd kernel is not used, dk and dv are computed
            # in one kernel using *dkv* block sizes, and dq is computed in another kernel using *dq
            # block sizes.
            block_q_dkv=self.get_backend_overrides("splash_block_q_dkv", block_size),
            block_kv_dkv=self.get_backend_overrides("splash_block_kv_dkv", block_size),
            block_kv_dkv_compute=self.get_backend_overrides(
                "splash_block_kv_dkv_compute", block_size
            ),
            block_q_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_q_dq", block_size),
            block_kv_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_kv_dq", block_size),
            # The fused kernel is neutral in small models and a ~5%-15% improvement in larger ones.
            # E.g., 1.03x speedup in a 12.6b simulated model, 1.06x speedup in 29.6b ,
            # and 1.14x in 539.5b.
            # NOTE(hanzhi-zhou): Fused bwd kernel may require more memory usage because it needs to
            # keep a temporary unreduced dq tensor of shape (kv_seq_len // block_kv_dkv, *q.shape)
            # in HBM. If memory usage is a problem, consider increasing block_kv_dkv or disabling
            # fused kernel.
            use_fused_bwd_kernel=self._use_fused,
        )
        splash_mask = _to_splash_mask(
            mask, mask_shape=(query.shape[2], key.shape[2]), q_seq_shards=1
        )

        num_heads = query.shape[1]
        mha_mask = splash_attention_mask.MultiHeadMask(masks=[splash_mask] * num_heads)

        num_heads = query.shape[1]
        kernel = splash_attention_kernel.make_splash_mha(
            mask=mha_mask,
            block_sizes=block_sizes,
            # TODO(dhwang2): support "seq" and "model" shard.
            head_shards=1,
            q_seq_shards=1,
            dropout_rate=cfg.dropout_rate,
            interpret=self.cfg.interpret,
            residual_checkpoint_name=f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}",
        )
        p_kernel = functools.partial(kernel, prng_key=prng_key, logit_sink=logit_sink)
        vp_kernel = jax.vmap(p_kernel, axis_name="batch")
        context = vp_kernel(q=query, k=key, v=value, segment_ids=seg_ids)
        return jnp.einsum("bnth->btnh", context)

    def get_dropout_mask(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """Auxiliary function to get the dropout mask for debugging purposes.
        It will return a boolean dropout mask of shape [batch, num_heads, seq_len, seq_len].
        """
        cfg = self.config
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        prng_key = input_batch.get("prng_key", None)

        if cfg.dropout_rate > 0.0 and prng_key is None:
            raise ValueError(
                "TPU SplashAttention requires a prng_key to be provided when dropout is enabled."
            )

        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query) * self.cfg.softmax_scale
        key = jnp.einsum("bsnh->bnsh", key)

        block_size = self.cfg.tpu_block_size
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=self.get_backend_overrides("splash_block_q", block_size),
            block_kv=self.get_backend_overrides("splash_block_kv", block_size),
            block_kv_compute=self.get_backend_overrides("splash_block_kv_compute", block_size),
            block_q_dkv=self.get_backend_overrides("splash_block_q_dkv", block_size),
            block_kv_dkv=self.get_backend_overrides("splash_block_kv_dkv", block_size),
            block_kv_dkv_compute=self.get_backend_overrides(
                "splash_block_kv_dkv_compute", block_size
            ),
            block_q_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_q_dq", block_size),
            block_kv_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_kv_dq", block_size),
            use_fused_bwd_kernel=self._use_fused,
        )

        kernel = functools.partial(
            splash_attention_kernel.get_dropout_mask,
            prng_key=prng_key,
            block_sizes=block_sizes,
            dropout_rate=cfg.dropout_rate,
        )
        v_kernel = jax.vmap(kernel, axis_name="batch")
        dropout_mask = v_kernel(query, key)
        return dropout_mask


class LegacyTPUFlashAttention(TPUFlashAttention):
    """Wraps the legacy (deprecated) implementation of TPU attention."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch, kv_cache_type=kv_cache_type):
            return False
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        if query.dtype != key.dtype:
            return self._log_unsupported(f"{query.dtype=} != {key.dtype=}")
        if self.cfg.dropout_rate != 0.0:
            return self._log_unsupported("dropout is not supported.")
        logit_sink = input_batch.get("logit_sink", None)
        if logit_sink is not None:
            return self._log_unsupported("LegacyTPUFlashAttention doesn't support logit sink.")
        return True

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BaseFlashAttention.__call__`."""
        bias: BaseAttentionBias = input_batch["bias"]
        causal_mask, segment_ids, explicit_bias = split(
            bias, CausalAttentionBias, SegmentIdAttentionBias
        )
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        segment_id_tensor = get_segment_ids(query=query, key=key, segment_ids=segment_ids)
        seg_ids = None
        if segment_id_tensor is not None:
            seg_ids = LegacySegmentIds(q=segment_id_tensor, kv=segment_id_tensor)
        key = repeat_kv_heads(query.shape[2], key)
        value = repeat_kv_heads(query.shape[2], value)
        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query) * self.cfg.softmax_scale
        key = jnp.einsum("bsnh->bnsh", key)
        value = jnp.einsum("bsnh->bnsh", value)

        block_size = self.cfg.tpu_block_size
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
        context = _pallas_tpu_flash_attention(
            query,
            key,
            value,
            ab=explicit_bias.value(),
            segment_ids=seg_ids,
            causal=causal_mask.has_value(),
            block_sizes=block_sizes,
            interpret=self.cfg.interpret,
        )
        return jnp.einsum("bnth->btnh", context)
