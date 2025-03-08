# Copyright © 2025 Apple Inc.
"""Common utilities across backends."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from axlearn.common.attention_bias import MaskFn
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
