"""Splash attention mask.

This code is adapted from the jax_ml/jax library, specifically from the
https://github.com/jax-ml/jax/blob/9bcfac6542a330b77f29d5cc5dcf4a57f55b2947/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py
TODO(dhwang2): Delete ComputableMask once JAX is upgraded and LocalMask becomes a computable mask.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import _ComputableMask

from axlearn.common.attention_bias import MaskFnAttentionBias
from axlearn.common.utils import Tensor


class ComputableMask(_ComputableMask):
    """Computable mask for splash attention that supports custom mask functions.

    This mask accepts any Jax/Numpy exchangeable mask function following the MaskFn protocol
    from attention_bias.py, such as causal_mask or sliding_window_causal_mask.

    Attributes:
      mask_fn: A callable mask function that takes query_position and key_position
        tensors and returns a boolean mask tensor.
    """

    mask_fn: Callable[[Tensor, Tensor], Tensor]

    def __init__(
        self,
        shape: tuple[int, int],
        mask_fn: Callable[[Tensor, Tensor], Tensor],
        shard_count: int = 1,
    ):
        """Initialize ComputableMask.

        Args:
            shape: The shape of the attention mask (q_len, kv_len).
            mask_fn: A callable that implements the MaskFn protocol from attention_bias.py.
                Takes (query_position, key_position) and returns a boolean mask.
            shard_count: Number of shards.
        """
        self.mask_fn = mask_fn

        def mask_function(q_ids, kv_ids):
            """Computes the attention mask using the provided mask_fn."""
            assert q_ids.ndim == 2
            assert kv_ids.ndim == 2
            return self.mask_fn(q_ids, kv_ids)

        super().__init__(
            shape=shape,
            mask_function=mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return False

        return (
            self.shape == other.shape
            and self.mask_fn == other.mask_fn
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                id(self.mask_fn),
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


def classify_blocks(
    mask: MaskFnAttentionBias,
    q_positions: np.ndarray | jax.Array,
    block_shape: tuple[int, int],
    *,
    kv_seq_len: int,
    downcast_smem_data: bool = True,
    head_shards: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Computes block-level sparsity via tiled mask function evaluation. Classify Q/KV block pairs
    as fully masked, partially masked, or fully unmasked.

    Args:
        mask: The attention mask.
        q_positions: Shape (batch, q_len). Absolute positions for each Q token per batch element.
        block_shape: (block_q, block_kv) tile sizes for the kernel grid.
        kv_seq_len: The total KV sequence length.
        downcast_smem_data: If True, downcast block_mask and data_next for scalar memory.
        head_shards: Number of head shards.

    Returns:
        (block_mask, data_next) arrays for the splash attention kernel.
        block_mask: Shape (batch, head_shards, q_blocks, kv_blocks).
            Values: 0=skip, 1=partial, 2=full.
        data_next: Shape (batch, head_shards, q_blocks, kv_blocks). KV block indices for iteration.
    """
    assert isinstance(mask, MaskFnAttentionBias)
    block_q, block_kv = block_shape
    q_seq_len = q_positions.shape[1]

    q_blocks, q_mod = divmod(q_seq_len, block_q)
    kv_blocks, kv_mod = divmod(kv_seq_len, block_kv)
    if q_mod != 0:
        raise ValueError(f"{block_q=} should divide {q_seq_len=}.")
    if kv_mod != 0:
        raise ValueError(f"{block_kv=} should divide {kv_seq_len=}.")

    def compute_block_mask(q_seq):
        def classify_q_row(q_block_idx):
            q_pos = jax.lax.dynamic_slice(q_seq, (q_block_idx * block_q,), (block_q,))

            def classify_kv_col(kv_block_idx):
                kv_pos = kv_block_idx * block_kv + jnp.arange(block_kv, dtype=jnp.int32)
                tile = mask.mask(q_pos[:, None], kv_pos[None, :])
                is_all = jnp.all(tile)
                is_any = jnp.any(tile)
                return jnp.where(is_all, 2, jnp.where(is_any, 1, 0)).astype(jnp.int32)

            return jax.vmap(classify_kv_col)(jnp.arange(kv_blocks, dtype=jnp.int32))

        return jax.lax.map(classify_q_row, jnp.arange(q_blocks, dtype=jnp.int32))

    per_batch_block_mask = jax.vmap(compute_block_mask)(q_positions)
    block_mask = jnp.broadcast_to(
        per_batch_block_mask[:, None, :, :],
        (q_positions.shape[0], head_shards, q_blocks, kv_blocks),
    )

    data_next = jnp.broadcast_to(
        jnp.arange(kv_blocks, dtype=jnp.int32)[None, None, None, :],
        block_mask.shape,
    )
    data_next = jnp.where(block_mask == 0, 0, data_next)

    if downcast_smem_data:
        block_mask = block_mask.astype(jnp.int8)
        if kv_blocks <= jnp.iinfo(jnp.int8).max:
            data_next = data_next.astype(jnp.int8)
        elif kv_blocks <= jnp.iinfo(jnp.int16).max:
            data_next = data_next.astype(jnp.int16)

    return block_mask, data_next
