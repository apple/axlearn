# Copyright © 2025 Apple Inc.
#
# Some of the code in this file is adapted from:
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Implements PagedAttention for TPU Pallas kernel with logit bias and mask_fn support.

Base implementation is ported from
https://github.com/jax-ml/jax/blob/jax-v0.6.0/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py

Added Block Sparse kernel to avoid unnecessary loads,
particularly for sliding window with long context, where we
1. Pre-compute a block-sparse mask with offset of shape (n_kv_blocks, n_kv_blocks)
2. We only load from the offset pointed to unmasked blocks in the kernel.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.attention_bias import (
    NEG_INF,
    BaseAttentionBias,
    MaskFn,
    SlidingWindowAttentionBias,
)
from axlearn.common.flash_attention.common import (
    build_mask,
    build_sliding_window_mask,
    query_iterator_indices,
)
from axlearn.common.utils import Tensor


class MultiPageAsyncCopyDescriptor:
    """Descriptor for async copy of multiple K/V pages from HBM.

    Ported from
    https://github.com/jax-ml/jax/blob/127aa7621868cb77e552b5d1f90e4a42b09c13fa/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L33
    """

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buffer,
        sem,
        page_indices,
        page_indices_start_offset: int,
        num_pages_to_load: int,
        head_index: int,
    ):
        self._vmem_buffer = vmem_buffer
        self._num_pages_to_load = num_pages_to_load
        if head_index is not None:
            self._pages_hbm_ref = pages_hbm_ref.at[head_index]
        else:
            self._pages_hbm_ref = pages_hbm_ref
        self._sem = sem
        self._page_indices = page_indices
        self._page_indices_start_offset = page_indices_start_offset
        self._async_copies = [self._make_async_copy(i) for i in range(self._num_pages_to_load)]

    def _make_async_copy(self, i):
        page_index = self._page_indices[self._page_indices_start_offset + i]
        return pltpu.make_async_copy(
            self._pages_hbm_ref.at[page_index],
            self._vmem_buffer.at[i],
            self._sem,
        )

    def start(self):
        """Starts the async copies."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait_and_get_loaded(self) -> Tensor:
        """Wait async copies and gets the loaded buffer as a Tensor."""
        for async_copy in self._async_copies:
            async_copy.wait()
        head_dim = self._vmem_buffer.shape[-1]
        tensor = self._vmem_buffer[...].astype(jnp.float32)
        return tensor.reshape(-1, head_dim)


def prepare_block_sparse_map(
    mask: BaseAttentionBias,
    lengths: Tensor,
    block_size: int,
    seq_len: int,
) -> Tuple[Any, Any]:
    """
    Computes a full block map num_kv_blocks * num_kv_blocks.
    """
    # Use a padding to ensure padding blocks aren't counted towards `kv_block_offset_size`.
    padding = -1
    mask_fn = mask.mask

    with jax.ensure_compile_time_eval():
        if mask_fn is not None:
            mask_args = dict(
                q_seq_len=seq_len,
                kv_seq_len=seq_len,
                block_q=block_size,
                block_k=block_size,
            )
            if isinstance(mask, SlidingWindowAttentionBias):
                bool_mask = build_sliding_window_mask(
                    **mask_args, sliding_window_size=mask.sliding_window_size
                )
            else:
                bool_mask = build_mask(mask_fn, **mask_args)
            offset, _ = query_iterator_indices(bool_mask, padding=padding)
        else:
            padded_num_kv_blocks = pl.cdiv(seq_len, block_size)
            offset = lax.broadcasted_iota(
                jnp.int32, (padded_num_kv_blocks, padded_num_kv_blocks), 1
            )

    kv_block_offset = offset[(lengths - 1) // block_size]
    # Count the number of blocks with position < kv_seq_len.
    kv_block_offset_size = jnp.count_nonzero(
        (kv_block_offset != padding) & (kv_block_offset * block_size < lengths[:, None]),
        axis=1,
    )
    # Replace padding with the last valid kv block's index. See
    # https://docs.jax.dev/en/latest/pallas/tpu/sparse.html#sparse-access-patterns-on-dense-data
    kv_block_offset = jnp.where(
        kv_block_offset == padding, kv_block_offset.max(axis=1, keepdims=True), kv_block_offset
    )
    return kv_block_offset, kv_block_offset_size


def _make_index_map(
    megacore_mode: Optional[str] = None,
    num_cores: int = 2,
    is_rearranged: bool = False,
    is_query: bool = False,
    is_sparse: bool = False,
):
    """Creates an index map function for query/bias tensor."""

    def dense_index_map(core_index, b, h, i, *_):
        if megacore_mode is None:
            batch_idx = b
            head_idx = h
        elif megacore_mode == "batch":
            batch_idx = b * num_cores + core_index
            head_idx = h
        elif megacore_mode == "kv_head":
            batch_idx = b
            head_idx = h * num_cores + core_index
        else:
            raise ValueError("Unsupported megacore mode")

        if is_query:
            i = 0
        if is_rearranged:
            return (batch_idx, head_idx, 0, i)
        return (batch_idx, head_idx, i)

    def sparse_index_map(core_index, b, h, i, kv_block_offset, *_):
        if megacore_mode is None:
            batch_idx = b
            head_idx = h
        elif megacore_mode == "batch":
            batch_idx = b * num_cores + core_index
            head_idx = h
        elif megacore_mode == "kv_head":
            batch_idx = b
            head_idx = h * num_cores + core_index
        else:
            raise ValueError("Unsupported megacore mode")

        i = kv_block_offset[batch_idx, i]
        if is_rearranged:
            return (batch_idx, head_idx, 0, i)
        return (batch_idx, head_idx, i)

    if is_sparse and not is_query:
        return sparse_index_map
    return dense_index_map


def _paged_flash_attention_sparse_kernel(
    # Scalars
    kv_block_offset,  # (batch_size, num_kv_blocks)
    kv_block_offset_size,  # (batch_size,)
    lengths_ref,  # (batch_size,)
    page_indices_ref,  # (batch_size * pages_per_sequence,)
    buffer_index_ref,  # (1,)
    init_flag_ref,  # (1,)
    # Inputs
    q_ref,  # (n_groups, head_dim)
    k_pages_hbm_ref,  # (n_kv_heads, batch_size * pages_per_sequence, page_size, head_dim)
    v_pages_hbm_ref,  # (n_kv_heads, batch_size * pages_per_sequence, page_size, head_dim)
    bias_ref,  # (n_groups, pages_per_compute_block * page_size)
    # Outputs
    o_ref,  # (n_groups, head_dim)
    # scratches
    m_i,  # (n_groups, 1)
    l_i,  # (n_groups, 1)
    k_vmem_buffer,  # (2, pages_per_compute_block, page_size, head_dim)
    v_vmem_buffer,  # (2, pages_per_compute_block, page_size, head_dim)
    sem,  # (1, )
    # Compile time args
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    softmax_scale: float,
    mask_fn: Optional[MaskFn] = None,
    megacore_mode: Optional[str] = None,
):
    core_index, batch_index, head_index, valid_block_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
    )
    num_cores = pl.num_programs(0)
    num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
    block_k = pages_per_compute_block * page_size

    b_step, b_start = 1, 0
    if megacore_mode == "batch":
        b_step, b_start = num_cores, core_index
    h_step, h_start = 1, 0
    if megacore_mode == "kv_head":
        h_step, h_start = num_cores, core_index
    h = head_index * h_step + h_start
    b = batch_index * b_step + b_start

    length = lengths_ref[b]
    num_valid_kv_blocks = kv_block_offset_size[b]

    def compute_block_indices(b, h, i):
        """Given current block indices prefetch next block.

        Args:
            b: batch index.
            h: head index.
            i: valid kv block index from block-sparse kv mask.

        Returns:
            Next valid kernel block indices.
        """

        def advance_b():
            next_b = b + b_step

            def advance_to_next_non_zero_length():
                next_next_b = next_b + b_step
                cond = (lengths_ref[b] == 0) | (kv_block_offset_size[b] == 0)
                return lax.fori_loop(
                    lax.div(next_next_b, b_step),
                    lax.div(batch_size, b_step),
                    lambda _, b: jnp.where(cond, b + b_step, b),
                    next_next_b,
                )

            return (
                lax.cond(
                    jnp.logical_and(
                        next_b < batch_size,
                        jnp.logical_or(
                            lengths_ref[next_b] == 0,
                            kv_block_offset_size[next_b] == 0,
                        ),
                    ),
                    advance_to_next_non_zero_length,
                    lambda: next_b,
                ),
                h_start,
                0,
            )

        def advance_h():
            next_h = h + h_step
            return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

        return lax.cond(i < num_valid_kv_blocks, lambda: (b, h, i), advance_h)

    def create_kv_async_copy_descriptors(b, h, i, buffer_index):
        block_offset = kv_block_offset[b, i]
        page_offset = b * pages_per_sequence + block_offset * pages_per_compute_block
        pages_to_load = pages_per_compute_block
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        return async_copy_k, async_copy_v

    # Traverse each batch's valid KV blocks
    @pl.when(valid_block_index < num_valid_kv_blocks)
    def flash_attention():
        init_flag = init_flag_ref[0]
        init_flag_ref[0] = 0
        buffer_index = buffer_index_ref[0]
        block_offset = kv_block_offset[b, valid_block_index]
        next_b, next_h, next_valid_block_index = compute_block_indices(b, h, valid_block_index + 1)

        @pl.when(valid_block_index == 0)
        def init():
            m_i[...] = jnp.full_like(m_i, NEG_INF)
            l_i[...] = jnp.zeros_like(l_i)
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(init_flag)
        def prefetch_first_block():
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
                b,
                h,
                valid_block_index,
                buffer_index,
            )
            async_copy_k.start()
            async_copy_v.start()

        @pl.when(next_b < batch_size)
        def prefetch_next_block():
            next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
            async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
                next_b,
                next_h,
                next_valid_block_index,
                next_buffer_index,
            )
            async_copy_next_k.start()
            async_copy_next_v.start()
            buffer_index_ref[0] = next_buffer_index

        async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
            b,
            h,
            valid_block_index,
            buffer_index,
        )
        q = q_ref[...].astype(jnp.float32)
        k = async_copy_k.wait_and_get_loaded()
        precision = jax.lax.Precision.DEFAULT
        qk = pl.dot(q, k.T, precision=precision)
        if softmax_scale != 0:
            qk *= softmax_scale
        if bias_ref is not None:
            qk += bias_ref[...]
            qk = jnp.maximum(qk, NEG_INF)

        block_kv_indices = block_offset * block_k + lax.broadcasted_iota(jnp.int32, qk.shape, 1)
        mask = block_kv_indices < length
        if mask_fn is not None:
            mask = mask & mask_fn(length - 1, block_kv_indices)

        qk = jnp.where(mask, qk, NEG_INF)
        m_prev, l_prev = m_i[...], l_i[...]
        m_curr = qk.max(axis=-1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = jnp.exp(qk - m_next)
        l_curr = s_curr.sum(axis=-1, keepdims=True)

        alpha = jnp.exp(m_prev - m_next)
        l_prev_corr = alpha * l_prev
        beta = jnp.exp(m_curr - m_next)
        l_curr_corr = beta * l_curr
        l_next = l_prev_corr + l_curr_corr

        m_i[...], l_i[...] = m_next, l_next
        v = async_copy_v.wait_and_get_loaded()
        o_curr = pl.dot(s_curr, v, precision=precision)

        o_ref[...] = ((l_prev_corr * o_ref[...] + beta * o_curr) / l_next).astype(o_ref.dtype)


def _paged_flash_attention_kernel(
    # inputs
    lengths_ref,  # (batch_size,)
    page_indices_ref,  # (batch_size * pages_per_sequence,)
    buffer_index_ref,  # (1,)
    init_flag_ref,  # (1,)
    q_ref,  # (n_groups, head_dim)
    k_pages_hbm_ref,  # (num_kv_heads, batch_size * pages_per_sequence, page_size, head_dim)
    v_pages_hbm_ref,  # (num_kv_heads, batch_size * pages_per_sequence, page_size, head_dim)
    bias_ref,  # (n_groups, pages_per_compute_block * page_size)
    # outputs
    o_ref,  # (n_groups, head_dim)
    # scratchs
    m_i,  # (n_groups, 1)
    l_i,  # (n_groups, 1)
    k_vmem_buffer,  # (2, pages_per_compute_block, page_size, head_dim)
    v_vmem_buffer,  # (2, pages_per_compute_block, page_size, head_dim)
    sem,  # (1, )
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    softmax_scale: float,
    mask_fn: Optional[MaskFn] = None,
    megacore_mode: Optional[str] = None,
    program_id: Optional[Tuple[int, int, int, int]] = None,
):
    """Compute paged flash attention.

    Ported from
    https://github.com/jax-ml/jax/blob/127aa7621868cb77e552b5d1f90e4a42b09c13fa/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L113
    Compared to original implementation, we
    1. accept customized bias and mask,
    2. move m_i, l_i from outputs to scratchs memory,
    3. rearrange some of the operations for better performance.
    """

    if program_id:
        core_index, b, h, i = program_id
    else:
        core_index, b, h, i = (
            pl.program_id(0),
            pl.program_id(1),
            pl.program_id(2),
            pl.program_id(3),
        )
    num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
    bk = page_size * pages_per_compute_block
    num_cores = pl.num_programs(0)

    b_step, b_start = 1, 0
    if megacore_mode == "batch":
        b_step, b_start = num_cores, core_index
    h_step, h_start = 1, 0
    if megacore_mode == "kv_head":
        h_step, h_start = num_cores, core_index

    h = h * h_step + h_start
    b = b * b_step + b_start
    length = lengths_ref[b]

    def compute_block_indices(b, h, i):
        """Given current block indices, get (next_b, next_h, next_i) for pre-fetching.

        Order of the increment is inner-to-outer, i.e. (k_split, head_split, batch_split).
        """

        def advance_b():
            next_b = b + b_step

            def advance_to_next_non_zero_length():
                next_next_b = next_b + b_step
                return lax.fori_loop(
                    lax.div(next_next_b, b_step),
                    lax.div(batch_size, b_step),
                    lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
                    next_next_b,
                )

            return (
                lax.cond(
                    jnp.logical_and(next_b < batch_size, lengths_ref[next_b] == 0),
                    advance_to_next_non_zero_length,
                    lambda: next_b,
                ),
                h_start,
                0,
            )

        def advance_h():
            next_h = h + h_step
            return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

        return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

    def create_kv_async_copy_descriptors(b, h, i, buffer_index):
        page_offset = b * pages_per_sequence + i * pages_per_compute_block
        pages_to_load = pages_per_compute_block
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        return async_copy_k, async_copy_v

    @pl.when(i * bk < length)
    def flash_attention():
        init_flag = init_flag_ref[0]
        init_flag_ref[0] = 0
        buffer_index = buffer_index_ref[0]
        next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

        @pl.when(init_flag)
        def prefetch_first_block():
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
                b,
                h,
                i,
                buffer_index,
            )
            async_copy_k.start()
            async_copy_v.start()

        @pl.when(i == 0)
        def init():
            m_i[...] = jnp.full_like(m_i, NEG_INF)
            l_i[...] = jnp.zeros_like(l_i)
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(next_b < batch_size)
        def prefetch_next_block():
            next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
            async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
                next_b,
                next_h,
                next_i,
                next_buffer_index,
            )
            async_copy_next_k.start()
            async_copy_next_v.start()
            buffer_index_ref[0] = next_buffer_index

        async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
            b,
            h,
            i,
            buffer_index,
        )
        q = q_ref[...].astype(jnp.float32)
        k = async_copy_k.wait_and_get_loaded()
        # Note: Using HIGHEST here would cause numerical
        # instability for query_step > 1
        precision = jax.lax.Precision.DEFAULT
        qk = pl.dot(q, k.T, precision=precision)
        if softmax_scale != 0:
            qk *= softmax_scale
        if bias_ref is not None:
            qk += bias_ref[...]
            qk = jnp.maximum(qk, NEG_INF)

        block_kv_indices = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
        mask = block_kv_indices < length
        if mask_fn is not None:
            mask = mask & mask_fn(length - 1, block_kv_indices)
        # (n_groups, block_k)
        qk = jnp.where(mask, qk, NEG_INF)
        m_prev, l_prev = m_i[...], l_i[...]

        m_curr = qk.max(axis=-1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_curr)

        s_curr = jnp.exp(qk - m_next)
        l_curr = s_curr.sum(axis=-1, keepdims=True)

        alpha = jnp.exp(m_prev - m_next)
        l_prev_corr = alpha * l_prev
        beta = jnp.exp(m_curr - m_next)
        l_curr_corr = beta * l_curr
        l_next = l_prev_corr + l_curr_corr

        m_i[...], l_i[...] = m_next, l_next

        v = async_copy_v.wait_and_get_loaded()
        o_curr = pl.dot(s_curr, v, precision=precision)

        o_ref[...] = ((l_prev_corr * o_ref[...] + beta * o_curr) / l_next).astype(o_ref.dtype)
