# Copyright © 2025 Apple Inc.
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
from axlearn.common.utils import Tensor


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
    """Common interface for FlashAttention for all backends."""

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

    def __init__(self, cfg):
        super().__init__(cfg)
        # Keep a typed copy of self.config.
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

    def _check_block_size(self, *, query: Tensor, key: Tensor, block_size: int) -> bool:
        q_seq_len = query.shape[1]
        k_seq_len = key.shape[1]
        if q_seq_len % block_size != 0 or k_seq_len % block_size != 0:
            self._log_unsupported(f"{q_seq_len=} or {k_seq_len=} is not divisible by {block_size=}")
            return False
        return True

    def is_supported(
        self, *, query: Tensor, key: Tensor, value: Tensor, bias: BaseAttentionBias
    ) -> bool:
        """Returns whether the attention kernel supports the given configuration.

        Note: This method is called outside of jax.shard_map, so query has the global shape.

        Args:
            query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
            key: Key of shape [batch_size, source_length, num_kv_heads, per_head_dim].
            value: Value of shape [batch_size, source_length, num_kv_heads, per_head_dim].
            bias: Attention bias to apply.

        Returns:
            True if the current configuration is supported. False otherwise.

        Raises:
            ValueError: If the given configuration doesn't logically make sense, e.g. if the
                shapes of q/k/v do not satisfy the requirement of a standard attention.
        """
        del bias
        if key.shape != value.shape:
            raise ValueError(f"Expects {key.shape=} to be equal to {value.shape=}")
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

    # Note: Positional arguments are used since some use cases require positional-only args,
    # such as functional transformations.
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: BaseAttentionBias,
        prng_key: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes attention context.

        Note: This method is called inside jax.shard_map, so query has the per-device shape.
        Warning: The dtype of key and value may differ from the dtype of query.

        Args:
            query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
            key: Key of shape [batch_size, source_length, num_kv_heads, per_head_dim].
            value: Value of shape [batch_size, source_length, num_kv_heads, per_head_dim].
            bias: Attention bias to apply.
            prng_key: PRNG key for dropout. Only needed when dropout_rate > 0.0.

        Returns:
            The context tensor of shape [batch_size, target_length, num_heads, per_head_dim].
        """
        raise NotImplementedError()


class BaseSingleStepDecoding(BaseFlashAttention):
    """Wraps the common checks for single step decoding kernels."""

    @classmethod
    def default_config(cls) -> BaseFlashAttention.Config:
        cfg: BaseFlashAttention.Config = super().default_config()
        cfg.is_decoding = True
        return cfg

    def is_supported(
        self, *, query: Tensor, key: Tensor, value: Tensor, bias: BaseAttentionBias
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(query=query, key=key, value=value, bias=bias):
            return False
        if not self.cfg.is_decoding:
            return self._log_unsupported("is_decoding=False.")
        if query.shape[1] != 1:
            return self._log_unsupported(f"{query.shape[1]=} != 1")
        if self.cfg.dropout_rate != 0.0:
            raise ValueError("Dropout rate cannot be set for decoding!")
        return True


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
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: BaseAttentionBias,
        prng_key: Optional[Tensor] = None,
        dropout_mask: Optional[Tensor] = None,
    ):
        # We apply the scale factor before the attention biases.
        query *= self.cfg.softmax_scale
        logits = compute_gqa_logits(query, key)
        probs = softmax_with_biases(logits, bias.value())
        if self.cfg.dropout_rate > 0:
            probs = dropout(probs, prng_key=prng_key, rate=self.cfg.dropout_rate, mask=dropout_mask)
        return compute_gqa_context(probs, value)
