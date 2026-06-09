# Copyright © 2025 Apple Inc.

"""A KVCache layer that Manages a fixed cached_kv_length kv_cache using a FIFO approach."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from axlearn.common.attention import MultiheadAttention
from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.module import nowrap
from axlearn.common.utils import Nested, Tensor, maybe_shard


class SlidingWindowKVCache(BaseKVCache):
    """KV cache for sliding window attention.

    Manages a fixed cached_kv_length kv_cache using a FIFO approach.
    This KV cache falls back to standard attention when using flash decoding.

    Note: `SlidingWindowKVCache` updates the cache using a ring buffer. To make `extend_step` run
    in O(1) instead of O(window), the stored sequence order may not match the original order
    (e.g., `[10, 11, 6, 7, 8, 9]`). Since `key_positions` tracks the true positions, this does not
    affect attention computation results.
    Note: Because of this, flash attention kernels like SplashAttention, which assume monotonic
    sequence order, cannot be used. `SlidingWindowKVCache` therefore uses the standard attention
    fallback during flash decoding. For window sizes below 4k, there is little benefit to flash
    decoding anyway, and on TPU benchmarks, flash decoding was over 50% slower.
    """

    @config_class
    class Config(BaseKVCache.Config):
        """Configures SlidingWindowKVCache."""

        # Specifies the size of the key-value cache. `init_states()` ignores `shape[1]`.
        cached_kv_length: Required[int] = REQUIRED

    def _invaild_position(self) -> int:
        # Out of window position.
        return -(self.config.cached_kv_length + 1)

    @nowrap
    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        # NB: key and value in init_state are transposed so that source_length is in the last
        # dimension as a TPU fusion optimization for one-hot matmul. See KVCache.
        cfg = self.config
        shape = (shape.batch_size, shape.num_kv_heads, shape.per_head_dim, cfg.cached_kv_length)
        # kv_partition_spec is in BTNH layout. Rotate to BNHT for key/value,
        # and use batch-only spec for key_positions [B, T].
        bnht_spec = None
        batch_spec = None
        if cfg.kv_partition_spec is not None:
            b, t, n, h = cfg.kv_partition_spec
            bnht_spec = (b, n, h, t)
            batch_spec = (b, None)
        return dict(
            key=maybe_shard(jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)), bnht_spec),
            value=maybe_shard(jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)), bnht_spec),
            key_positions=maybe_shard(
                jnp.full(
                    (shape[0], cfg.cached_kv_length), self._invaild_position(), dtype=jnp.int32
                ),
                batch_spec,
            ),
        )

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        *,
        k_proj: Tensor,
        v_proj: Tensor,
        key_positions: Tensor,
        segment_ids: Optional[Tensor] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Tensor], BaseKVCache.Output]:
        """Updates the sliding window KV cache per extend step.

        Args:
            cached_states: A `Nested[Tensor]` object containing KV cache such as key and value.
            k_proj: A Tensor of shape [batch, step_length, num_kv_heads, per_head_dim].
            v_proj: A Tensor of shape [batch, step_length, num_kv_heads, per_head_dim].
            key_positions: An optional Tensor of shape [1|batch, step_length].
            segment_ids: An optional Tensor of shape [batch, step_length]. `segment_ids == 0`
                represents padding tokens that should not be cached. Padding tokens are masked out
                and marked as invalid positions.

        Returns:
            A tuple (updated_state, output):
            * updated_state: A `Nested[Tensor]` object containing KV cache such as key and value.
            * output: The output k_proj, v_proj, and key_positions, which are merged with
                KV cache, resulting in a length of `cached_kv_length + step_size`.
        """
        assert page_pool is None
        cfg = self.config
        cached_key: Tensor = cached_states["key"]
        cached_value: Tensor = cached_states["value"]
        cached_pos: Tensor = cached_states["key_positions"]
        batch, step_len = k_proj.shape[:2]
        invalid = self._invaild_position()

        # [1|batch, step_length] -> [batch, step_length]
        key_positions = jnp.broadcast_to(key_positions, (batch, step_len))
        if segment_ids is not None:
            # Mark padding positions as invalid so they are not stored in the ring buffer.
            key_positions = jnp.where(segment_ids != 0, key_positions, invalid)

        # [B, T, N, H] --> [B, N, H, T].
        k_proj = jnp.einsum("btnh->bnht", k_proj)
        v_proj = jnp.einsum("btnh->bnht", v_proj)

        # Update the KV entries in the ring buffer.
        def update_cache(k_proj, v_proj, key_positions):
            cache_len = cfg.cached_kv_length
            chex.assert_shape(cached_key, (*k_proj.shape[:3], cache_len))
            updated_state = dict()
            max_idx = key_positions.max(initial=0, axis=1, keepdims=True)
            min_idx = jnp.maximum(max_idx - (cache_len - 1), 0)
            ring_positions = jnp.where(key_positions >= min_idx, key_positions % cache_len, invalid)
            oh_indices = jax.nn.one_hot(ring_positions, cache_len, dtype=cached_key.dtype)
            pos_scattered = jnp.einsum("bt,bts->bs", key_positions, oh_indices)
            k_scattered = jnp.einsum("b...t,bts->b...s", k_proj, oh_indices)
            v_scattered = jnp.einsum("b...t,bts->b...s", v_proj, oh_indices)
            keep_mask = ~oh_indices.any(axis=1)  # [B, S]
            updated_state["key_positions"] = cached_pos * keep_mask + pos_scattered.astype(
                cached_pos.dtype
            )
            keep_mask = keep_mask[:, None, None, :]  # [B, 1, 1, S]
            updated_state["key"] = cached_key * keep_mask + k_scattered.astype(cached_key.dtype)
            updated_state["value"] = cached_value * keep_mask + v_scattered.astype(
                cached_value.dtype
            )
            chex.assert_equal_shape((updated_state["key"], cached_key))
            chex.assert_equal_shape((updated_state["value"], cached_value))
            chex.assert_equal_shape((updated_state["key_positions"], cached_pos))
            return updated_state

        updated_state = update_cache(k_proj, v_proj, key_positions)

        # This KV is used only for this attention computation. Since key_positions indicates KV
        # positions, simply concatenation is sufficient.
        def prepare_proj(k_proj, v_proj, key_positions):
            key_positions = jnp.concat((cached_pos, key_positions), axis=1)
            k_proj = jnp.concat((cached_key, k_proj), axis=3)
            v_proj = jnp.concat((cached_value, v_proj), axis=3)
            # [B, S, N, H]
            k_proj = jnp.einsum("bnhs->bsnh", k_proj)
            v_proj = jnp.einsum("bnhs->bsnh", v_proj)
            return self.Output(k_proj=k_proj, v_proj=v_proj, key_positions=key_positions)

        outputs = prepare_proj(k_proj, v_proj, key_positions)
        return updated_state, outputs


def enable_sliding_window_attention(
    cfg: MultiheadAttention.Config, sliding_window_size: int
) -> MultiheadAttention.Config:
    """Enable sliding window attention.

    Args:
        cfg: MultiheadAttention Config.
        sliding_window_size: Sliding window size.

    Returns:
        The in-place modified MultiheadAttention Config with sliding window attention enabled.
    """
    cfg.set(
        kv_cache=SlidingWindowKVCache.default_config().set(
            cached_kv_length=sliding_window_size,
            cache_dtype=cfg.kv_cache.cache_dtype,
            kv_partition_spec=cfg.kv_cache.kv_partition_spec,
        ),
        mask=SlidingWindowAttentionBias.default_config(sliding_window_size=sliding_window_size),
    )
    return cfg
