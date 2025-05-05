# Copyright © 2025 Apple Inc.
#
# Some of the code in this file is adapted from:
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implements PagedAttention for GPU in JAX with logit bias support.

This implementation is ported from
https://github.com/jax-ml/jax/blob/jax-v0.6.0/jax/experimental/pallas/ops/gpu/paged_attention.py

"""
import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from axlearn.common.attention_bias import (
    NEG_INF,
    BaseAttentionBias,
    MaskFn,
    MaskFnAttentionBias,
    split,
)
from axlearn.common.flash_attention.common import BasePagedAttention, get_gpu_dot_precision
from axlearn.common.flash_attention.gpu_attention import NoPopDict
from axlearn.common.flash_attention.gpu_decoding import _get_sm_count as get_sm_count
from axlearn.common.utils import Nested, Tensor


def _paged_attention_kernel(
    # inputs
    q_ref,  # [block_h, head_dim]
    key_ref,  # [total_num_pages, page_size, head_dim]
    value_ref,  # [total_num_pages, page_size, head_dim]
    page_tables_ref,  # [pages_per_partition]
    bias_ref,  # [block_h, pages_per_partition * page_size]
    lengths_ref,  # []
    # outputs
    o_ref,  # [block_h, head_dim]
    *residual_refs,  # Residual outputs: [block_h,], [block_h,]
    pages_per_compute_block: int,
    mask_value: float,
    softmax_scale: float,
    mask_fn: Optional[MaskFn],
) -> None:
    """Computes attention outputs for the given block.

    Compared to jax-ml implementation, we add supports for bias and mask.
    """

    partition_idx = pl.program_id(2)
    block_h, head_dim = q_ref.shape
    precision = get_gpu_dot_precision(q_ref.dtype)
    page_size = key_ref.shape[-2]
    pages_per_partition = page_tables_ref.shape[0]
    block_k = pages_per_compute_block * page_size

    def _compute(
        start_page_idx,
        end_page_idx,
        o,
        m_i,
        l_i,
    ):
        """Computes attention for a range of pages.

        Args:
            start_page_idx: Start page index.
            end_page_idx: End page index.
            o: Output buffer of shape [block_h, head_dim].
            m_i: Maximum logits buffer of shape [block_h].
            l_i: Sum of exponentials buffer of shape [block_h].

        Returns:
            Tuple of (output, max_logits, sum_exponentials).
        """
        q_slice = pl.ds(0, block_h)
        q = pl.load(q_ref, (q_slice, slice(None))) * softmax_scale

        def body(start_k, carry):
            o_prev, m_prev, l_prev = carry

            page_tables_slice = pl.ds(start_k * pages_per_compute_block, pages_per_compute_block)
            page_tables = pl.load(page_tables_ref, page_tables_slice)
            k = key_ref[page_tables].reshape(block_k, head_dim)
            v = value_ref[page_tables].reshape(block_k, head_dim)
            logits = pl.dot(q, k.T, precision=precision)  # [block_h, block_k]
            curr_start_page_idx = (
                partition_idx * pages_per_partition + start_k * pages_per_compute_block
            )
            curr_start_token_idx = curr_start_page_idx * page_size
            mask = jnp.arange(block_k) + curr_start_token_idx < length

            if bias_ref is not None:
                bias_slice = pl.ds(start_k * block_k, block_k)
                bias = pl.load(bias_ref, (slice(None), bias_slice))
                logits += bias

            if mask_fn is not None:
                mask &= mask_fn(length - 1, jnp.arange(block_k) + curr_start_token_idx)
            mask = lax.broadcast_in_dim(mask, (block_h, block_k), (1,))
            logits = jnp.where(mask, logits, mask_value)

            log2e = math.log2(math.e)
            m_curr = logits.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp2((m_prev - m_next) * log2e)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp2((logits - m_next[:, None]) * log2e)
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            o_prev_corr = correction[:, None] * o_prev
            o_curr = pl.dot(s_curr.astype(v.dtype), v)

            o_next = o_prev_corr + o_curr
            return o_next, m_next, l_next

        max_it = pl.cdiv(end_page_idx - start_page_idx, pages_per_compute_block)
        (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))

        return o, m_i, l_i

    m_i = jnp.zeros(block_h, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
    l_i = jnp.zeros(block_h, dtype=jnp.float32)
    o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)
    length = pl.load(lengths_ref, ())

    start_page_idx = partition_idx * pages_per_partition
    end_page_idx = start_page_idx + pages_per_partition

    end_page_idx = jnp.minimum(pl.cdiv(length, page_size), end_page_idx)

    o, m_i, l_i = jax.lax.cond(
        start_page_idx >= end_page_idx,
        lambda: (o, m_i, l_i),
        lambda: _compute(start_page_idx, end_page_idx, o, m_i, l_i),
    )

    o_ref[...] = o.astype(o_ref.dtype)

    if residual_refs is not None:
        l_ref, m_ref = residual_refs
        l_ref[...] = l_i
        m_ref[...] = m_i


def _largest_divisor_leq(x: int, y: int):
    if x < y:
        x, y = y, x
    root = int(math.isqrt(x))
    best = None

    for d in range(1, root + 1):
        if x % d:
            continue
        big = x // d
        if big < y:
            return big
        if d < y:
            best = d
    return best


def _paged_attention_unbatched(
    q,  # [num_q_heads, head_dim]
    key,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    value,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    page_tables,  # [pages_per_sequence]
    bias,  # [num_q_heads, pages_per_sequence * page_size]
    lengths,  # [1]
    *,
    block_h: int,
    pages_per_compute_block: int,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    debug: bool,
    mask_value: float,
    softmax_scale: float,
    batch_size: int,
    mask_fn: Optional[MaskFn],
):
    """Partition unbatched input and feed into the compute kernel.

    Compared to jax-ml implementation, we support bias and mask with proper block partition.
    """

    num_q_heads, head_dim = q.shape
    num_kv_heads, total_num_pages, page_size, _ = key.shape
    pages_per_sequence = page_tables.shape[0]

    q_heads_per_kv_head = num_q_heads // num_kv_heads
    q_reshaped = q.reshape(num_kv_heads, q_heads_per_kv_head, head_dim)

    # k_split for SM utilization
    good_k_split_for_sm_util = max(16, get_sm_count() // (q_heads_per_kv_head * batch_size))
    max_k_split = pl.cdiv(total_num_pages, pages_per_compute_block)
    k_splits = min(max_k_split, good_k_split_for_sm_util)
    k_splits = _largest_divisor_leq(pages_per_sequence, k_splits)
    assert k_splits is not None
    assert (
        pages_per_sequence % k_splits == 0
    ), f"{pages_per_sequence=} must be divisible by {k_splits=}."

    pages_per_partition = pages_per_sequence // k_splits
    pages_per_compute_block = min(pages_per_partition, pages_per_compute_block)

    assert (
        pages_per_partition % pages_per_compute_block == 0
    ), f"{pages_per_partition=} must be divisible by {pages_per_compute_block=}."

    page_tables = page_tables.reshape(k_splits, pages_per_partition)

    bias_reshaped = None
    if bias is not None:
        assert (
            bias.shape[0] == q.shape[0]
        ), f"Bias's first dim should be num_q_heads {q.shape[0]}, got {bias.shape[0]}"
        assert bias.shape[1] == pages_per_sequence * page_size, (
            f"Bias's second dim should be kv_len, "
            f"{pages_per_sequence} * {page_size}, got {bias.shape[1]}"
        )
        bias_reshaped = bias.reshape(
            num_kv_heads, q_heads_per_kv_head, k_splits, pages_per_partition * page_size
        )

    if q_heads_per_kv_head % block_h:
        n_elements_to_pad = -q_heads_per_kv_head % block_h
        q_reshaped = jnp.pad(q_reshaped, ((0, 0), (0, n_elements_to_pad), (0, 0)))
        if bias_reshaped is not None:
            bias_reshaped = jnp.pad(bias_reshaped, ((0, 0), (0, n_elements_to_pad), (0, 0), (0, 0)))

    head_splits = pl.cdiv(q_heads_per_kv_head, block_h)
    grid = (num_kv_heads, head_splits, k_splits)
    kernel = functools.partial(
        _paged_attention_kernel,
        pages_per_compute_block=pages_per_compute_block,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        mask_fn=mask_fn,
    )

    o, l, m = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((None, block_h, head_dim), lambda h, i, k: (h, i, 0)),  # q
            pl.BlockSpec(
                (None, total_num_pages, page_size, head_dim),
                lambda h, i, k: (h, 0, 0, 0),
            ),  # key
            pl.BlockSpec(
                (None, total_num_pages, page_size, head_dim),
                lambda h, i, k: (h, 0, 0, 0),
            ),  # value
            pl.BlockSpec((None, pages_per_partition), lambda h, i, k: (k, 0)),  # page_tables
            (
                None
                if bias is None
                else pl.BlockSpec(
                    (None, block_h, None, pages_per_partition * page_size),
                    lambda h, i, k: (h, i, k, 0),
                )
            ),  # bias
            (None if lengths is None else pl.BlockSpec((), lambda h, i, k: ())),  # lengths
        ],
        out_specs=[
            pl.BlockSpec((None, None, block_h, head_dim), lambda h, i, k: (k, h, i, 0)),  # q
            pl.BlockSpec((None, None, block_h), lambda h, i, k: (k, h, i)),  # l
            pl.BlockSpec((None, None, block_h), lambda h, i, k: (k, h, i)),  # m
        ],
        out_shape=[
            jax.ShapeDtypeStruct((k_splits, *q_reshaped.shape), dtype=q.dtype),  # o
            jax.ShapeDtypeStruct((k_splits, *q_reshaped.shape[:-1]), dtype=jnp.float32),  # l
            jax.ShapeDtypeStruct((k_splits, *q_reshaped.shape[:-1]), dtype=jnp.float32),  # m
        ],
        debug=debug,
        interpret=interpret,
        compiler_params=NoPopDict(triton=NoPopDict(num_warps=num_warps, num_stages=num_stages)),
        name=f"paged_attention_{block_h=}_{pages_per_compute_block=}",
    )(q_reshaped, key, value, page_tables, bias_reshaped, lengths)

    if q_heads_per_kv_head % block_h:
        o = o[..., :q_heads_per_kv_head, :]
        l = l[..., :q_heads_per_kv_head]
        m = m[..., :q_heads_per_kv_head]

    # final round of flash
    m_next = m.max(axis=0)
    correction = jnp.exp(m - m_next[None])
    o = o * correction[..., None].astype(o.dtype)
    l_next = (l * correction).sum(axis=0)
    eps = jnp.finfo(l_next.dtype).eps
    o = o.sum(axis=0) / ((l_next[..., None] + eps).astype(o.dtype))

    o = o.reshape(q.shape).astype(q.dtype)
    return o


class GPUPagedAttention(BasePagedAttention):
    """Implements GPU PagedAttention ."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """See `BasePagedAttention.is_supported`."""
        if not super().is_supported(input_batch):
            return False
        key: Tensor = input_batch["key"]
        if not self._check_block_size(input_batch, block_size=self.cfg.gpu_block_size):
            return False
        if self.cfg.gpu_block_size / key.shape[2] >= 16:
            self._log_unsupported(
                f"We want to keep pages per compute block less than 16 for shared memory "
                f"size limit, got {self.cfg.gpu_block_size} / {key.shape[2]}"
            )

            return False
        return True

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BasePagedAttention.__call__`."""
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        page_tables: Tensor = input_batch["page_tables"]
        bias: BaseAttentionBias = input_batch["bias"]
        query = query.squeeze(1)
        batch_size, q_heads, head_dim = query.shape
        q_heads_per_kv_head = q_heads // key.shape[0]
        mask, explicit_bias = split(bias, MaskFnAttentionBias)
        mask_fn, lengths = None, None
        if mask is None or not hasattr(mask, "target_positions") or mask.target_positions is None:
            raise ValueError("Cannot retrieve MaskFnAttentionBias or target_positions.")
        if hasattr(mask, "mask"):
            mask_fn = mask.mask
        lengths = mask.target_positions[:, -1] + 1
        lengths = jnp.broadcast_to(jnp.asarray(lengths), (batch_size,))

        bias = explicit_bias.value()
        if bias is not None:
            bias = jnp.broadcast_to(bias, (batch_size, q_heads, 1, bias.shape[-1])).squeeze(2)

        # We use pages_per_compute_block to get block_k in the kernel
        # which is computed as pages_per_compute_block * page_size
        pages_per_compute_block = self.cfg.gpu_block_size // key.shape[2]

        impl = functools.partial(
            _paged_attention_unbatched,
            pages_per_compute_block=pages_per_compute_block,
            # Minimum block size is 16 to allow pl.dot to lower successfully.
            block_h=max(16, pl.next_power_of_2(q_heads_per_kv_head)),
            num_warps=4,
            num_stages=2,
            interpret=self.cfg.interpret,
            debug=False,
            mask_value=NEG_INF,
            mask_fn=mask_fn,
            batch_size=batch_size,
            softmax_scale=self.cfg.softmax_scale,
        )

        o = jax.vmap(impl, (0, None, None, 0, 0, 0), 0)(
            query,
            key,
            value,
            page_tables,
            bias,
            lengths,
        ).reshape(batch_size, 1, q_heads, head_dim)

        return o
