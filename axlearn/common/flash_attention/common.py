# Copyright © 2025 Apple Inc.
# Some of the code in this file is adapted from:
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Common utilities across backends."""

from functools import partial
from typing import Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental import pallas as pl

from axlearn.common.attention import compute_gqa_context, compute_gqa_logits, softmax_with_biases
from axlearn.common.attention_bias import BaseAttentionBias, MaskFn, SegmentIdAttentionBias
from axlearn.common.config import Configurable, config_class
from axlearn.common.layers import dropout
from axlearn.common.utils import Nested, Tensor, validate_contains_paths


def build_mask(
    mask_fn: MaskFn, *, q_seq_len: int, kv_seq_len: int, block_q: int, block_k: int
) -> np.ndarray:
    """Builds the block map where True means the block is not fully masked.

    Args:
        mask_fn: The attention mask function.
        q_seq_len: Query sequence length.
        kv_seq_len: Key/Value sequence length.
        block_q: Query block size.
        block_k: Key/Value block size.

    Returns:
        A boolean array of shape (num_q_blocks, num_kv_blocks) where True means the block is not
        fully masked. num_q_blocks * block_q will be larger than q_seq_len if q_seq_len is not
        divisible by block_q. The same holds true for kv blocks.
    """
    # Initialize the iteration map where True means the block is not empty.
    num_q_blocks = pl.cdiv(q_seq_len, block_q)
    num_kv_blocks = pl.cdiv(kv_seq_len, block_k)
    block_mask_map = np.ones(shape=(num_q_blocks, num_kv_blocks), dtype=np.bool_)
    # # Initialize the scan begin and end indices.
    rows = np.arange(q_seq_len, dtype=np.int32)
    cols = np.arange(kv_seq_len, dtype=np.int32)
    # Run a compile-time evaluation to get the mask array.
    # TODO(kelvin-zou): use a block-wise mask function to avoid the compile-time
    # high memory usage.
    with jax.ensure_compile_time_eval():
        mask_array = np.asarray(mask_fn(rows[:, None], cols[None, :]))
    for i in range(0, q_seq_len, block_q):
        for j in range(0, kv_seq_len, block_k):
            # Extract the block
            block = mask_array[i : i + block_q, j : j + block_k]
            # All empty means skipping
            if not block.any():
                block_mask_map[i // block_q, j // block_k] = False
    return block_mask_map


class KVOffsetInfo(NamedTuple):
    """Records the block index of non-empty KV blocks.

    Attributes:
        kv_block_offset: A (num_q_blocks, num_kv_blocks) tensor where `kv_block_offset[i][j]`
            stores the index of the jth non-empty KV block index for the ith query block.
            This tensor may be padded at the end.
        kv_block_offset_size: A (num_q_blocks,) tensor that stores the number of valid entries
            for each row of `kv_block_offset`, i.e. the number of entries before padding.
    """

    kv_block_offset: Tensor
    kv_block_offset_size: Tensor


def query_iterator_indices(block_mask_map: np.ndarray, *, padding: int = 0) -> KVOffsetInfo:
    """Builds `KVOffsetInfo` for block-sparse attention computation in the forward pass.

    Returns:
        A `KVOffsetInfo`. See the attributes of `KVOffsetInfo` for more info.
    """
    num_q_blocks, num_kv_blocks = block_mask_map.shape
    index_offset = np.full((num_q_blocks, num_kv_blocks), padding, dtype=np.int32)
    index_offset_size = np.zeros(shape=(num_q_blocks), dtype=np.int32)
    for i in range(num_q_blocks):
        k = 0
        for j in range(num_kv_blocks):
            if block_mask_map[i, j]:
                index_offset[i, k] = j
                k += 1
        index_offset_size[i] = k
    return KVOffsetInfo(
        kv_block_offset=jnp.asarray(index_offset),
        kv_block_offset_size=jnp.asarray(index_offset_size),
    )


class BaseFlashAttention(Configurable):
    """Common interface of Flash attention for all backends."""

    @config_class
    class Config(Configurable.Config):
        """Configures BaseFlashAttention.

        Attributes:
            is_decoding: Whether we're in decoding/inference mode.
            softmax_scale: Scale factor to apply to QK.
            dropout_rate: Dropout rate for attention probs.
            interpret: Whether to use interpret mode for Pallas kernels.
            tpu_block_size: Block size for TPU pallas kernels.
            gpu_block_size: Block size for GPU pallas kernels.
        """

        is_decoding: bool = False
        softmax_scale: float = 1.0
        dropout_rate: float = 0.0
        interpret: bool = False
        tpu_block_size: int = 512
        gpu_block_size: int = 128

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: BaseFlashAttention.Config = self.config

    def name(self) -> str:
        """Returns the class name."""
        return self.__class__.__name__

    def _log_unsupported(self, reason: str) -> Literal[False]:
        """Logs this class is unsupported with `reason`.

        The log message will be formatted as `Not using {self.name()} because {reason}`.

        This method also conveniently returns False so it could be used like this in `is_supported`
        ```
        if ...:
            return self._log_unsupported(...)
        ```
        """
        logging.warning("Not using %s because %s", self.name(), reason)
        return False

    # Note: Positional arguments are used since some use cases require positional-only args,
    # such as functional transformations.
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """Computes attention context.

        Note: This method is called inside jax.shard_map, so query has the per-device shape.
        Warning: The dtype of key and value may differ from the dtype of query.
        `is_supported` will always validate `input_batch` prior this call,
        so we don't re-check if `input_batch` contains necessary entries for the computation.

        Args:
            input_batch: A dict with the following entries:
                query: A Tensor of shape [batch_size, target_length, num_heads, per_head_dim].
                key: A Tensor of shape [batch_size, source_length, num_kv_heads, per_head_dim];
                value: A Tensor has the same shape with `key`.
                prng_key: An optional Tensor only needed when dropout_rate > 0.0.
                bias: Attention bias to apply.

        Returns:
            The context tensor of shape [batch_size, target_length, num_heads, per_head_dim].
        """
        raise NotImplementedError()

    def _validate_input_batch(self, input_batch: Nested[Tensor | BaseAttentionBias]):
        """Returns whether the input batch is valid for the flash attention call.

        Args:
            input_batch: A dict contains input entries, see __call__ for details.

        Raises:
            ValueError: If query/key/value missing in the input_batch, or the
                shapes of key/value mismatch
        """
        validate_contains_paths(input_batch, paths=["query", "key", "value", "bias"])

        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        if key.shape != value.shape:
            raise ValueError(f"Expects {key.shape=} to be equal to {value.shape=}")

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """Returns whether the attention kernel supports the given configuration.

        Args:
            input_batch: A dict contains input entries, see __call__ for details.

        Returns:
            True if the current configuration is supported. False otherwise.

        Raises:
            ValueError: If the given configuration doesn't logically make sense, e.g. if the
                shapes of q/k/v do not satisfy the requirement of a standard attention.
        """
        self._validate_input_batch(input_batch)
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        if query.shape[0] != key.shape[0]:
            raise ValueError(
                f"Expects query batch size {query.shape[0]} to be equal to key batch size "
                f"{key.shape[0]}"
            )
        if query.shape[-1] != key.shape[-1]:
            raise ValueError(
                f"Expects query head dim {query.shape[-1]} to be equal to key head dim "
                f"{key.shape[-1]}"
            )
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError(
                f"Expects query num heads {query.shape[2]} to be divisible by num key heads "
                f"{key.shape[2]}"
            )
        return True

    def _check_block_size(
        self, input_batch: Nested[Tensor | BaseAttentionBias], *, block_size: int
    ) -> bool:
        """Returns whether the attention kernel supports the given block size.

        Args:
            input_batch: A dict contains input entries, see __call__ for details.
            block_size: An integer value specified backend kernel's block size.

        Returns:
            True if the current block_size is supported. False otherwise.
        """
        self._validate_input_batch(input_batch)
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        q_seq_len = query.shape[1]
        k_seq_len = key.shape[1]
        if q_seq_len % block_size != 0 or k_seq_len % block_size != 0:
            self._log_unsupported(f"{q_seq_len=} or {k_seq_len=} is not divisible by {block_size=}")
            return False
        return True


class BaseSingleStepDecoding(BaseFlashAttention):
    """Wraps the common checks for single step decoding kernels."""

    @classmethod
    def default_config(cls) -> BaseFlashAttention.Config:
        cfg: BaseFlashAttention.Config = super().default_config()
        cfg.is_decoding = True
        return cfg

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch):
            return False
        if not self.cfg.is_decoding:
            return self._log_unsupported("is_decoding=False.")
        query: Tensor = input_batch["query"]
        if query.shape[1] != 1:
            return self._log_unsupported(f"{query.shape[1]=} != 1")
        if self.cfg.dropout_rate != 0.0:
            raise ValueError("Dropout rate cannot be set for decoding!")
        return True


class BasePagedAttention(BaseSingleStepDecoding):
    """Base class for paged attention."""

    def _validate_input_batch(self, input_batch: Nested[Tensor | BaseAttentionBias]):
        super()._validate_input_batch(input_batch)
        validate_contains_paths(input_batch, paths=["page_tables"])

    def _check_block_size(
        self, input_batch: Nested[Tensor | BaseAttentionBias], block_size: int
    ) -> bool:
        self._validate_input_batch(input_batch)
        key: Tensor = input_batch["key"]
        page_size = key.shape[2]

        # In the paged attention kernel, `block_k` for standard flash attention is computed as
        # pages_per_compute_block * page_size.
        if block_size % page_size != 0:
            self._log_unsupported(f"{block_size=} is not divisible by page size {key.shape[2]}")
            return False
        return True

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        """Returns whether paged attention kernel supports the given config.

        Args:
            input_batch: A dict contains input entries, see __call__ for details.

        Returns:
            True if the current configuration is supported in paged attention. False otherwise.
        """
        self._validate_input_batch(input_batch)
        if not self.cfg.is_decoding:
            return self._log_unsupported("is_decoding=False.")
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        page_tables: Tensor = input_batch["page_tables"]
        if query.shape[1] != 1:
            return self._log_unsupported(f"{query.shape[1]=} != 1")
        if self.cfg.dropout_rate != 0.0:
            raise ValueError("Dropout rate cannot be set for decoding!")
        if query.shape[2] % key.shape[0] != 0:
            return self._log_unsupported(
                f"Number of Q heads {query.shape[2]} must be divisible "
                f"by number of kv heads {key.shape[0]}"
            )
        if query.shape[-1] != key.shape[-1]:
            return self._log_unsupported(
                f"head_dim of Q {query.shape[-1]} must be the same as that of K/V {key.shape[-1]}"
            )
        if page_tables.shape[0] != query.shape[0]:
            return self._log_unsupported(
                f"page tables must have the same batch size with query, "
                f"got {page_tables.shape[0]} and {query.shape[0]}"
            )

        return True

    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """Computes attention context.

        Note: This method is called inside jax.shard_map, so query has the per-device shape.
        Warning: The dtype of key and value may differ from the dtype of query.

        Args:
            input_batch: A dict with the following entries:
                query: A Tensor of shape [batch_size, 1, num_heads, per_head_dim].
                key: A Tensor of shape [num_kv_heads, total_num_pages, page_size, per_head_dim]
                    a *physical-page* layout in which every key page
                    stores exactly `page_size` tokens.
                value: A Tensor of the same shape as `key`, holding value pages.
                page_tables: An int Tensor with [batch_size, pages_per_sequence]
                    as the logical to phsyical lookup table mapping `(sequence_idx, page_idx)`
                    logical page to a physical page index in `key`/`value`.
                    Each entry of the table is in the range of [0, batch_size * pages_per_sequence).
                    ```
                    physical_idx = page_tables[b, j]
                    k_page = key[:, physical_idx, :, :]
                    # [num_kv_heads, page_size, per_head_dim]
                    v_page = value[:, physical_idx, :, :]
                    ```
                    Inside the kernel we
                    1. fetch a contiguous slice of this logical table, then
                    2. gather the corresponding physical pages.
                prng_key: An optional tensor used only when `dropout_rate > 0.0`.
                    For paged-attention kernels pass `None`.
                bias: Attention bias to apply.


        Returns:
            The context tensor of shape [batch_size, 1, num_heads, per_head_dim].
        """
        raise NotImplementedError()


def get_segment_ids(
    *, query: Tensor, key: Tensor, segment_ids: SegmentIdAttentionBias
) -> Optional[Tensor]:
    """Return the segment ids Tensor from the sequence of segment ids attention
    biases or None if there are no segment ids.
    """
    if not segment_ids.has_value():
        return None
    if query.shape[1] != key.shape[1]:
        raise ValueError("segment_ids is only supported for query and key with identical lengths.")
    if segment_ids.eval_shape()[0] != query.shape[0]:
        raise ValueError(
            "segment_ids must have matching batch dim: "
            f"{segment_ids.eval_shape()} vs. {query.shape[0]}"
        )
    return segment_ids.segment_ids


def repeat_kv_heads(num_q_heads: int, key_or_value: Tensor) -> Tensor:
    """Repeats key or value heads dim to match the query.

    TODO(dhwang2): optimize computation like GroupedQueryAttention.
    """
    num_head_repeats = num_q_heads // key_or_value.shape[-2]
    if num_head_repeats == 1:
        return key_or_value
    # Repeat along the num_heads dim: [batch, source_length, num_heads, per_head_dim].
    return jnp.repeat(key_or_value, num_head_repeats, axis=-2)


class ReferenceMHA(BaseFlashAttention):
    """The reference implementation of attention in XLA."""

    # The additional argument `dropout_mask` is for unit test only.
    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        dropout_mask: Optional[Tensor] = None,
    ):
        # We apply the scale factor before the attention biases.
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        bias: BaseAttentionBias = input_batch["bias"]
        query *= self.cfg.softmax_scale
        page_tables = input_batch.get("page_tables", None)
        if page_tables is not None:
            key = reconstruct_kv(page_tables, key)
            value = reconstruct_kv(page_tables, value)
        logits = compute_gqa_logits(query, key)
        probs = softmax_with_biases(logits, bias.value())
        if self.cfg.dropout_rate > 0:
            probs = dropout(
                probs,
                prng_key=input_batch.get("prng_key", None),
                rate=self.cfg.dropout_rate,
                mask=dropout_mask,
            )
        return compute_gqa_context(probs, value)

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> bool:
        # @TODO(senyut): Refactor support check.
        if input_batch.get("page_tables") is None:
            return BaseFlashAttention.is_supported(
                self,
                input_batch=input_batch,
            )
        return BasePagedAttention.is_supported(
            self,
            input_batch=input_batch,
        )


def get_cpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """Get the suitable DotAlgorithmPreset for the given dtype for CPU backend.

    CPU doesn't support different compute and accumulation precision. This should only be used
    for CPU emulation and unit tests.
    """
    if dtype == jnp.float32:
        return jax.lax.DotAlgorithmPreset.F32_F32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F16
    if dtype == jnp.bfloat16:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_BF16
    raise ValueError(f"Unsupported dtype {dtype}")


# See https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.DotAlgorithm for information.
def get_gpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """Get the suitable DotAlgorithmPreset for the given dtype."""
    # General rules:
    # 1. Must accumulate in FP32 precision.
    # 2. Must use TensorCore.
    if jax.default_backend() == "cpu":
        return get_cpu_dot_precision(dtype)
    if dtype == jnp.float32:
        # We can use F32_F32_F32, but it disables the use of TensorCore and makes it more than 10x
        # slower on H100, as matmul fallbacks to using CUDA cores.
        return jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F32
    if dtype == jnp.bfloat16:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    raise ValueError(f"Unsupported dtype {dtype}")


# See https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.DotAlgorithm for information.
def get_tpu_dot_precision(dtype) -> jax.lax.Precision:
    """Get the suitable DotAlgorithmPreset for the given dtype.

    TPU Pallas lowering doesn't yet support DotAlgorithmPreset. Use Precision instead.
    """
    if jax.default_backend() == "cpu":
        return get_cpu_dot_precision(dtype)
    if dtype == jnp.float32:
        # HIGHEST uses BF16_BF16_F32_X6, which emulates higher precision with 6 BF16 passes.
        # Note: jax.lax.Precision.HIGH (BF16_BF16_F32_X3) is not yet supported. We should use it
        # when it's supported as it's twice as fast and precision is ok.
        return jax.lax.Precision.HIGHEST
    if dtype == jnp.bfloat16:
        return jax.lax.Precision.DEFAULT
    raise ValueError(f"Unsupported dtype {dtype}")


def reconstruct_kv(page_tables: Tensor, pages: Tensor) -> Tensor:
    """Retrieve key/value from page tables given pages.

    Ported from
    https://github.com/jax-ml/jax/blob/1594d2f30fdbfebf693aba4a2b264e4a3e52acc6/tests/pallas/tpu_paged_attention_kernel_test.py#L62

    Args:
        page_tables: [batch_size, pages_per_sequence], specifying page indices.
        pages: [num_kv_heads, total_num_pages, page_size, head_dim], k/v pages.

    Returns:
        Retrieved actual key / value of shape [batch_size, kv_seq_len, n_kv_heads, head_dim]
    """

    def fn(page_tables: Tensor, pages: Tensor) -> Tensor:
        # page_tables: (pages_per_sequence)
        # pages: (n_kv_heads, total_pages, page_size, head_dim)
        head_dim = pages.shape[-1]
        out = pages.at[page_tables].get(mode="fill", fill_value=0.0)
        return out.reshape(-1, head_dim)

    with_batch = jax.vmap(fn, (0, None), 0)
    attn_fn = jax.vmap(with_batch, (None, 0), 1)

    out = attn_fn(page_tables, pages)
    out = jnp.swapaxes(out, 1, 2)

    return out
