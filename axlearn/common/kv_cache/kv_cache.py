# Copyright © 2025 Apple Inc.

"""A KVCache layer that stores KV cache in a tensor of shape
[batch_size, kv_heads, head_dim, max_seq_len]."""

from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.module import nowrap
from axlearn.common.utils import Nested, Tensor, maybe_shard

# Sentinel value for padding/unused key positions. Must exceed any reachable query position so
# that `causal_mask = query_pos >= key_pos` returns False for them.
_INVALID_KV_POSITION = jnp.iinfo(jnp.int32).max


class KVCache(BaseKVCache):
    """Default KV cache.

    Manages the kv_cache provided with max_len and updates it at each time_step.
    Padding tokens (segment_ids == 0) are not written to the cache; the causal attention mask
    naturally excludes unwritten slots via `query_pos >= slot_index`.
    """

    @nowrap
    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        # NB: key and value in init_state are transposed so that source_length is in the last
        # dimension as a TPU fusion optimization for one-hot matmul.
        # Reference:
        # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.
        cfg = self.config
        shape_kv = (shape.batch_size, shape.num_kv_heads, shape.per_head_dim, shape.kv_len)
        # kv_partition_spec is in BTNH layout. Rotate to BNHT to match stored tensor layout.
        bnht_spec = None
        if cfg.kv_partition_spec is not None:
            b, t, n, h = cfg.kv_partition_spec
            bnht_spec = (b, n, h, t)
        return dict(
            key=maybe_shard(jnp.zeros(shape=shape_kv, dtype=self._cache_dtype(dtype)), bnht_spec),
            value=maybe_shard(jnp.zeros(shape=shape_kv, dtype=self._cache_dtype(dtype)), bnht_spec),
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
        assert page_pool is None
        if k_proj.shape != v_proj.shape:
            raise ValueError(f"{k_proj.shape=} != {v_proj.shape=}")
        if k_proj.shape[1] != key_positions.shape[1]:
            raise ValueError(f"{k_proj.shape[1]=} != {key_positions.shape[1]=}")

        cached_key: Tensor = cached_states["key"]
        cached_value: Tensor = cached_states["value"]
        batch, step_size = k_proj.shape[:2]

        # [1|batch, step_length] -> [batch, step_length]
        key_positions = jnp.broadcast_to(key_positions, (batch, step_size))
        # Padding tokens get _INVALID_KV_POSITION so they never occupy a valid cache slot.
        if segment_ids is not None:
            key_positions = jnp.where(segment_ids != 0, key_positions, _INVALID_KV_POSITION)

        # [B, T, N, H] --> [B, N, H, T].
        k_proj = jnp.einsum("btnh->bnht", k_proj)
        v_proj = jnp.einsum("btnh->bnht", v_proj)

        # On GPU, dynamic_update_slice_in_dim becomes faster when step size ≈ 32.
        # On TPU, dynamic_update_slice_in_dim becomes faster when step size ≈ 1024.
        threshold = 32 if jax.default_backend() != "tpu" else 1024

        # dynamic_update_slice_in_dim is typically used for updating tensors, but we found that
        # when step_size is small, one-hot matmul is 10-20% faster on both TPU and GPU.
        if step_size < threshold:
            source_len = cached_key.shape[-1]
            # Padding positions (mapped to _INVALID_KV_POSITION ≥ source_len) produce all-zero
            # one_hot rows, leaving every cache slot untouched.
            oh_indices = jax.nn.one_hot(key_positions, source_len, dtype=cached_key.dtype)
            keep_mask = ~oh_indices.any(axis=1)  # [B, S]

            k_scattered = jnp.einsum("b...t,bts->b...s", k_proj, oh_indices)
            v_scattered = jnp.einsum("b...t,bts->b...s", v_proj, oh_indices)
            cached_key = cached_key * keep_mask[:, None, None, :] + k_scattered.astype(
                cached_key.dtype
            )
            cached_value = cached_value * keep_mask[:, None, None, :] + v_scattered.astype(
                cached_value.dtype
            )
        else:
            # Note: KV transpose is an optimization for one-hot matmul and is not related to
            # dynamic_update_slice_in_dim. As a result, KV transpose only adds overhead for it.
            # Since small step_size scenarios are more frequent, we accept slowdown in this case.

            def update_single(cached_kv_slice, kv_proj_slice, time_idx):
                return jax.lax.dynamic_update_slice_in_dim(
                    cached_kv_slice, kv_proj_slice, time_idx, axis=-1
                )

            vmap_update = jax.vmap(update_single)
            # Use the first (valid) position as the write offset. `segment_ids` must be contiguous
            # at the beginning (e.g., [1,1,1,0,0,0]).
            time_step = jnp.broadcast_to(key_positions[:, 0], [batch])
            cached_key = vmap_update(cached_key, k_proj.astype(cached_key.dtype), time_step)
            cached_value = vmap_update(cached_value, v_proj.astype(cached_key.dtype), time_step)

        updated_state = dict(key=cached_key, value=cached_value)
        assert updated_state["key"].shape == cached_key.shape
        assert updated_state["value"].shape == cached_value.shape

        # [B, S, N, H]
        k_proj = jnp.einsum("bnhs->bsnh", cached_key)
        v_proj = jnp.einsum("bnhs->bsnh", cached_value)
        key_positions_out = jnp.arange(k_proj.shape[1])[None]  # [1, source_length]
        return updated_state, self.Output(
            k_proj=k_proj, v_proj=v_proj, key_positions=key_positions_out
        )
