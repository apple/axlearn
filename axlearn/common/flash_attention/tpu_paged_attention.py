# Copyright © 2025 Apple Inc.
#
# Some of the code in this file is adapted from:
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implements PagedAttention for TPU.

Compared to base implementation
https://github.com/jax-ml/jax/blob/jax-v0.6.0/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py
We added block-sparse kernel for long context with masking
(particularly for sliding window attention),
we also supports arbitrary logit bias and mask_fn.
"""

import functools
from typing import Literal, Optional, Sequence

import jax
import jax.numpy as jnp
from absl import logging
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.attention_bias import (
    BaseAttentionBias,
    MaskFnAttentionBias,
    SlidingWindowAttentionBias,
    split,
)
from axlearn.common.flash_attention.common import BasePagedAttention
from axlearn.common.flash_attention.tpu_paged_attention_kernel import (
    _make_index_map,
    _paged_flash_attention_kernel,
    _paged_flash_attention_sparse_kernel,
    prepare_block_sparse_map,
)
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
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


class TPUPagedAttention(BasePagedAttention):
    """Wraps TPU paged flash attention kernel."""

    def megacore_mode_heuristic(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Literal["kv_head", "batch", None]:
        """Simple heuristic to enable megacore parallelism on TPUs with 2 cores.

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

    def sparse_mode_heuristic(
        self,
        mask: BaseAttentionBias,
        max_length: int,
    ) -> bool:
        """Simple heuristic of whether to use block-sparse kernel.

        True if we are using sliding window attention and sliding window
        size is smaller than max seq len by sparse ratio specified in config.
        """
        if isinstance(mask, SlidingWindowAttentionBias):
            size = mask.sliding_window_size
            if size < max_length * self.cfg.sparse_ratio:
                return True
        return False

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BasePagedAttention.is_supported`."""
        if not super().is_supported(input_batch=input_batch, kv_cache_type=kv_cache_type):
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
        # Length going out-of-bound may trigger a device halt.
        lengths = jnp.minimum(lengths, pages_per_sequence * page_size)

        sparse_mode = self.sparse_mode_heuristic(mask, page_size * pages_per_sequence)

        bias = explicit_bias.value()
        bias_spec = None
        q_block_spec = pl.BlockSpec(
            (None, num_groups, head_dim),
            _make_index_map(megacore_mode, num_cores, is_rearranged=False, is_query=True),
        )
        q_dtype_for_kernel_launch = query.dtype
        if bias is not None:
            block_k = pages_per_compute_block * page_size
            # TODO(senyut): we don't necessarily need to broadcast 2D bias
            #               (target_len, kv_len) to 4D for.
            bias = jnp.broadcast_to(bias, (batch_size, num_q_heads, 1, bias.shape[-1])).squeeze(2)
            bias_spec = pl.BlockSpec(
                (None, num_groups, block_k),
                _make_index_map(
                    megacore_mode,
                    num_cores,
                    is_rearranged=False,
                    is_query=False,
                    is_sparse=sparse_mode,
                ),
            )
        if num_groups % 8 != 0:
            # Reshape q to hint XLA to pick a <1x128> layout otherwise
            # it will pick a <8x128> layout for a <1x128> memref inside
            # the kernel and error out.
            query = query.reshape(batch_size, num_q_heads, 1, head_dim)
            q_block_spec = pl.BlockSpec(
                (None, num_groups, None, head_dim),
                _make_index_map(megacore_mode, num_cores, is_rearranged=True, is_query=True),
            )

            if bias is not None:
                bias = bias.reshape(batch_size, num_q_heads, 1, bias.shape[-1]).astype(jnp.float32)
                block_k = pages_per_compute_block * page_size
                bias_spec = pl.BlockSpec(
                    (None, num_groups, None, block_k),
                    _make_index_map(
                        megacore_mode,
                        num_cores,
                        is_rearranged=True,
                        is_query=False,
                        is_sparse=sparse_mode,
                    ),
                )
            q_dtype_for_kernel_launch = jnp.float32

        dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
        kernel = (
            _paged_flash_attention_sparse_kernel if sparse_mode else _paged_flash_attention_kernel
        )
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
        args = (
            lengths,
            page_tables.reshape(-1),
            jnp.zeros((1,), jnp.int32),
            jnp.ones((1,), jnp.int32),
            query.astype(q_dtype_for_kernel_launch),
            key,
            value,
            bias,
        )
        num_scalars = 4
        if sparse_mode:
            kv_block_offset, kv_block_offset_size = prepare_block_sparse_map(
                mask,
                lengths=lengths,
                block_size=self.cfg.tpu_block_size,
                seq_len=page_size * pages_per_sequence,
            )
            args = (kv_block_offset, kv_block_offset_size) + args
            num_scalars = 6

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
                num_scalar_prefetch=num_scalars,
                in_specs=in_specs,
                out_specs=q_block_spec,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=dimension_semantics,
            ),
            out_shape=jax.ShapeDtypeStruct(query.shape, q_dtype_for_kernel_launch),
            interpret=self.cfg.interpret,
        )(*args)
        return out.reshape(batch_size, 1, num_q_heads, head_dim).astype(query.dtype)
