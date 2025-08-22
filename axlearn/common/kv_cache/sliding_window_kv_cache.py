# Copyright © 2025 Apple Inc.

"""A KVCache layer that Manages a fixed cached_kv_length kv_cache using a FIFO approach."""

import typing
from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.utils import Nested, Tensor, sequence_mask

if typing.TYPE_CHECKING:
    from axlearn.common.attention import MultiheadAttention


class SlidingWindowKVCache(BaseKVCache):
    """KV cache for sliding window attention.

    Manages a fixed cached_kv_length kv_cache using a FIFO approach.
    """

    @config_class
    class Config(BaseKVCache.Config):
        """Configures SlidingWindowKVCache."""

        # Specifies the size of the key-value cache. `init_states()` ignores `shape[1]`.
        cached_kv_length: Required[int] = REQUIRED

    def _invaild_position(self) -> int:
        # Out of window position.
        return -(self.config.cached_kv_length + 1)

    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        cfg = self.config
        shape = (shape.batch_size, cfg.cached_kv_length, shape.num_kv_heads, shape.per_head_dim)
        return dict(
            key=jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)),
            value=jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)),
            key_positions=jnp.full(
                shape=shape[:2], fill_value=self._invaild_position(), dtype=jnp.int32
            ),
        )

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        *,
        k_proj: Tensor,
        v_proj: Tensor,
        key_positions: Tensor,
        unpadded_len: Optional[Tensor] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Tensor], BaseKVCache.Output]:
        """Updates the sliding window KV cache per extend step.

        Args:
            cached_states: A `Nested[Tensor]` object containing KV cache such as key and value.
            k_proj: A Tensor of shape [batch, step_length, num_kv_heads, per_head_dim].
            v_proj: A Tensor of shape [batch, step_length, num_kv_heads, per_head_dim].
            key_positions: An optional Tensor of shape [1|batch, step_length].
            unpadded_len: An optional Tensor of shape [batch]. Specifies the number of
                non-padding tokens per sequence. When provided, only the first `unpadded_len[i]`
                tokens of sequence `i` are considered valid and will be cached. Padding tokens
                are masked out and marked as invalid positions.

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
        cached_positions: Tensor = cached_states["key_positions"]
        batch, step_len = k_proj.shape[:2]
        assert cached_key.shape == (batch, cfg.cached_kv_length, *k_proj.shape[2:])

        # [1|batch, step_length] -> [batch, step_length]
        key_positions = jnp.broadcast_to(key_positions, (batch, step_len))
        if unpadded_len is not None:
            if unpadded_len.shape[0] != batch:
                raise ValueError(f"{unpadded_len.shape=} must be [{batch}].")
            steps = unpadded_len
            seq_mask = sequence_mask(lengths=steps, max_len=step_len, dtype=key_positions.dtype)
            # update_single rolls key_positions, so mark invalid positions.
            key_positions = jnp.where(seq_mask, key_positions, self._invaild_position())
        else:
            steps = jnp.full([batch], fill_value=step_len)

        # Ensure that we accumulate using the original dtype.
        k_proj = k_proj.astype(cached_key.dtype)
        v_proj = v_proj.astype(cached_value.dtype)

        # Function to update the cache for a single batch element.
        def update_single(cached_kv_slice, kv_proj_slice, steps_slice):
            new_kv_slice = jnp.concatenate((cached_kv_slice, kv_proj_slice), axis=0)  # [T, N, H]
            shift = kv_proj_slice.shape[0] - steps_slice
            new_kv_slice = jnp.roll(new_kv_slice, shift, axis=0)
            return new_kv_slice

        # Use jax.vmap to vectorize over the batch dimension.
        vmap_update = jax.vmap(update_single)
        # [B, Lc, N, H], [B, S, N, H] -> [B, Lc+S, N, H]
        new_key = vmap_update(cached_key, k_proj, steps)
        new_value = vmap_update(cached_value, v_proj, steps)
        new_key_positions = vmap_update(cached_positions, key_positions, steps)  # [batch, Lc+S]
        updated_state = dict(
            key=new_key[:, step_len:],
            value=new_value[:, step_len:],
            key_positions=new_key_positions[:, step_len:],
        )
        assert updated_state["key"].shape == cached_key.shape
        assert updated_state["value"].shape == cached_value.shape
        return updated_state, self.Output(
            k_proj=new_key, v_proj=new_value, key_positions=new_key_positions
        )


def enable_sliding_window_attention(
    cfg: "MultiheadAttention.Config", sliding_window_size: int
) -> "MultiheadAttention.Config":
    """Enable sliding window attention.

    Args:
        cfg: MultiheadAttention Config.
        sliding_window_size: Sliding window size.

    Returns:
        The in-place modified MultiheadAttention Config with sliding window attention enabled.
    """
    if cfg.kv_cache is not None:
        cache_dtype = cfg.kv_cache.cache_dtype
    else:
        cache_dtype = None
    cfg.set(
        kv_cache=SlidingWindowKVCache.default_config().set(
            cache_dtype=cache_dtype, cached_kv_length=sliding_window_size
        ),
        mask=SlidingWindowAttentionBias.default_config(sliding_window_size=sliding_window_size),
    )
    return cfg
