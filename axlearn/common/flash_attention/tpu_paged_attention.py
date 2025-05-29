# Copyright © 2025 Apple Inc.
#
# Some of the code in this file is adapted from:
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implements PagedAttention for TPU in JAX with logit bias and mask_fn support.

This implementation is ported from
https://github.com/jax-ml/jax/blob/jax-v0.6.0/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py
"""

import functools
from typing import Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from absl import logging
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.attention_bias import (
    NEG_INF,
    BaseAttentionBias,
    MaskFn,
    MaskFnAttentionBias,
    split,
)
from axlearn.common.flash_attention.common import BasePagedAttention
from axlearn.common.utils import Nested, Tensor


def _get_tpu_cores_per_chip(interpret: bool = False) -> int:
    """Return number of physical cores per TPU if available.

    Fall back to 1 when `interpret` is True.

    Raises
        RuntimeError If TPU detection fails and `interpret` is False.
    """
    try:
        local_devices = jax.local_devices()
        if not local_devices:
            raise RuntimeError("No local JAX devices found on this host.")
        for dev in local_devices:
            if dev.platform == "tpu":
                return dev.num_cores
        raise RuntimeError(
            f"Found {len(local_devices)} JAX device(s) but none are TPU: "
            f"{[d.platform for d in local_devices]}"
        )
    except RuntimeError as exc:
        if interpret:
            logging.warning("TPU detection failed: %s — falling back to CPU.", exc)
            return 1
        raise


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


def _paged_flash_attention_kernel_inline_seq_dim(
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
):
    core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    o_ref[...] = jnp.zeros_like(o_ref)

    def body(i, _):
        _paged_flash_attention_kernel(
            lengths_ref,
            page_indices_ref,
            buffer_index_ref,
            init_flag_ref,
            q_ref,
            k_pages_hbm_ref,
            v_pages_hbm_ref,
            bias_ref,
            o_ref,
            m_i,
            l_i,
            k_vmem_buffer,
            v_vmem_buffer,
            sem,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            pages_per_sequence=pages_per_sequence,
            softmax_scale=softmax_scale,
            mask_fn=mask_fn,
            megacore_mode=megacore_mode,
            program_id=(core_index, b, h, i),
        )
        return ()

    bk = pages_per_compute_block * k_pages_hbm_ref.shape[-2]
    if megacore_mode == "batch":
        num_cores = pl.num_programs(0)
        length = lengths_ref[b * num_cores + core_index]
    else:
        length = lengths_ref[b]

    lax.fori_loop(0, lax.div(length + bk - 1, bk), body, ())


class TPUPagedAttention(BasePagedAttention):
    """Wraps TPU paged flash attention kernel."""

    def megacore_mode_heuristic(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Literal["kv_head", "batch", None]:
        """
        Simple heuristic to enable megacore parallelism on TPUs with 2 cores

        It prioritizes parallelizing the 'batch' dimension if the batch size
        is divisible by 2. If not, it attempts to parallelize the 'kv_head'
        dimension if the number of KV heads is divisible by 2.
        """
        megacore_mode = None
        cores_per_chip = _get_tpu_cores_per_chip(self.cfg.interpret)
        if cores_per_chip == 2:
            query: Tensor = input_batch["query"]
            if query.shape[0] % 2 == 0:
                megacore_mode = "batch"
            else:
                key: Tensor = input_batch["key"]
                if key.shape[0] % 2 == 0:
                    megacore_mode = "kv_head"
        return megacore_mode

    def inline_seq_dim_heuristic(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """Always use inline sequence dim kernel except when we set bias.

        Inline Seq Dim mode fuses the sequence-block axis into an internal loop
        otherwise, we keep it as an explicit grid dimension.
        By default, we trade a bit of parallelism for lighter
        compilation and launch overhead.
        """
        bias: BaseAttentionBias = input_batch["bias"]
        _, explicit_bias = split(bias, MaskFnAttentionBias)
        bias_val = explicit_bias.value()
        # TODO(senyut): To enable inline kernel with bias, we need to specifically
        #               update index map and bias block spec.
        if bias_val is not None:
            return False

        return True

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """See `BasePagedAttention.is_supported`."""
        if not super().is_supported(input_batch=input_batch):
            return False

        key: Tensor = input_batch["key"]
        if not self._check_block_size(input_batch, block_size=self.cfg.tpu_block_size):
            return False

        if key.shape[-1] % 128 != 0:
            return self._log_unsupported(
                f"Head dimension has to be a multiple of 128 for double-buffering DMA, "
                f"got {key.shape[-1]}"
            )
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
        batch_size, num_q_heads, head_dim = query.shape
        num_kv_heads, _, page_size, _ = key.shape
        _, pages_per_sequence = page_tables.shape

        num_groups = num_q_heads // num_kv_heads
        megacore_mode = self.megacore_mode_heuristic(input_batch)
        inline_seq_dim = self.inline_seq_dim_heuristic(input_batch)

        num_cores = _get_tpu_cores_per_chip(self.cfg.interpret)
        assert num_cores in (1, 2), f"Got unexpected number of TPU cores {num_cores}"

        per_core_batch = batch_size // num_cores if megacore_mode == "batch" else batch_size
        per_core_kv_heads = (
            num_kv_heads // num_cores if megacore_mode == "kv_head" else num_kv_heads
        )
        pages_per_compute_block = self.cfg.tpu_block_size // page_size

        mask, explicit_bias = split(bias, MaskFnAttentionBias)
        mask_fn, lengths = None, None
        if mask is None or not hasattr(mask, "target_positions") or mask.target_positions is None:
            raise ValueError("Cannot retrieve MaskFnAttentionBias or target_positions.")
        if hasattr(mask, "mask"):
            mask_fn = mask.mask
        lengths = mask.target_positions[:, -1] + 1
        lengths = jnp.broadcast_to(jnp.asarray(lengths), (batch_size,))

        bias = explicit_bias.value()
        bias_spec = None
        if bias is not None:
            block_k = pages_per_compute_block * page_size
            # TODO(senyut): we don't necessarily need to broadcast 2D bias
            #               (target_len, kv_len) to 4D for.
            bias = jnp.broadcast_to(bias, (batch_size, num_q_heads, 1, bias.shape[-1])).squeeze(2)
            # TODO(senyut): handle bias index map for inline kernel and remove this assertion
            assert not inline_seq_dim
            if megacore_mode == "batch":
                bias_spec = pl.BlockSpec(
                    (None, num_groups, block_k),
                    lambda core_index, b, h, i, *_: (b * num_cores + core_index, h, i),
                )
            elif megacore_mode == "kv_head":
                bias_spec = pl.BlockSpec(
                    (None, num_groups, block_k),
                    lambda core_index, b, h, i, *_: (b, h * num_cores + core_index, i),
                )
            else:
                bias_spec = pl.BlockSpec(
                    (None, num_groups, block_k),
                    lambda core_index, b, h, i, *_: (b, h, i),
                )

        if num_groups % 8 != 0:
            # Reshape q to hint XLA to pick a <1x128> layout otherwise
            # it will pick a <8x128> layout for a <1x128> memref inside
            # the kernel and error out.
            query = query.reshape(batch_size, num_q_heads, 1, head_dim)
            rearrange_bias_spec = False
            if bias is not None:
                bias = bias.reshape(batch_size, num_q_heads, 1, bias.shape[-1]).astype(jnp.float32)
                block_k = pages_per_compute_block * page_size
                rearrange_bias_spec = True
            if megacore_mode == "kv_head":
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, None, head_dim),
                    lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
                )
                if rearrange_bias_spec:
                    bias_spec = pl.BlockSpec(
                        (None, num_groups, None, block_k),
                        lambda core_index, b, h, i, *_: (b, h * num_cores + core_index, 0, i),
                    )
            elif megacore_mode == "batch":
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, None, head_dim),
                    lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
                )
                if rearrange_bias_spec:
                    bias_spec = pl.BlockSpec(
                        (None, num_groups, None, block_k),
                        lambda core_index, b, h, i, *_: (b * num_cores + core_index, h, 0, i),
                    )
            else:
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, None, head_dim),
                    lambda core_index, b, h, *_: (b, h, 0, 0),
                )
                if rearrange_bias_spec:
                    bias_spec = pl.BlockSpec(
                        (None, num_groups, None, block_k),
                        lambda core_index, b, h, i, *_: (b, h, 0, i),
                    )
            q_dtype_for_kernel_launch = jnp.float32
        else:
            if megacore_mode == "kv_head":
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, head_dim),
                    lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
                )
            elif megacore_mode == "batch":
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, head_dim),
                    lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
                )
            else:
                q_block_spec = pl.BlockSpec(
                    (None, num_groups, head_dim),
                    lambda core_index, b, h, *_: (b, h, 0),
                )
            q_dtype_for_kernel_launch = query.dtype

        dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
        if inline_seq_dim:
            kernel = _paged_flash_attention_kernel_inline_seq_dim
            grid = (
                num_cores,
                per_core_batch,
                per_core_kv_heads,
            )
            dimension_semantics = ("parallel", "arbitrary", "arbitrary")
        else:
            kernel = _paged_flash_attention_kernel
            grid = (
                num_cores,
                per_core_batch,
                per_core_kv_heads,
                pages_per_sequence // pages_per_compute_block,
            )
            dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")

        in_specs = [
            q_block_spec,  # Query
            pl.BlockSpec(memory_space=pltpu.ANY),  # Key pages
            pl.BlockSpec(memory_space=pltpu.ANY),  # Value pages
            bias_spec,  # Bias
        ]
        scratch_shapes = (
            pltpu.VMEM((num_groups, 1), jnp.float32),  # m_i
            pltpu.VMEM((num_groups, 1), jnp.float32),  # l_i
            pltpu.VMEM(
                (
                    2,
                    pages_per_compute_block,
                    page_size,
                    head_dim,
                ),
                key.dtype,
            ),  # k_vmem_buffer
            pltpu.VMEM(
                (
                    2,
                    pages_per_compute_block,
                    page_size,
                    head_dim,
                ),
                value.dtype,
            ),  # v_mem_buffer
            pltpu.SemaphoreType.DMA,  # sem
        )

        out = pl.pallas_call(
            functools.partial(
                kernel,
                batch_size=batch_size,
                pages_per_compute_block=pages_per_compute_block,
                pages_per_sequence=pages_per_sequence,
                softmax_scale=self.cfg.softmax_scale,
                mask_fn=mask_fn,
                megacore_mode=megacore_mode,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                in_specs=in_specs,
                out_specs=q_block_spec,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.TPUCompilerParams(
                dimension_semantics=dimension_semantics,
            ),
            out_shape=jax.ShapeDtypeStruct(query.shape, q_dtype_for_kernel_launch),
            interpret=self.cfg.interpret,
        )(
            lengths,
            page_tables.reshape(-1),
            jnp.zeros((1,), jnp.int32),
            jnp.ones((1,), jnp.int32),
            query.astype(q_dtype_for_kernel_launch),
            key,
            value,
            bias,
        )
        return out.reshape(batch_size, 1, num_q_heads, head_dim).astype(query.dtype)
