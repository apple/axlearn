# Copyright © 2025 Apple Inc.

"""A KVCache layer that stores KV cache in a tensor of shape
[batch_size, kv_heads, head_dim, max_seq_len]."""

from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.utils import Nested, Tensor


class KVCache(BaseKVCache):
    """Default KV cache.

    Manages the kv_cache provided with max_len and updates it at each time_step.
    """

    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        # NB: key and value in init_state are transposed so that source_length is in the last
        # dimension as a TPU fusion optimization for one-hot matmul.
        # Reference:
        # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.
        shape = (shape.batch_size, shape.num_kv_heads, shape.per_head_dim, shape.kv_len)
        init_states = dict(
            key=jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)),
            value=jnp.zeros(shape=shape, dtype=self._cache_dtype(dtype)),
        )
        return init_states

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        *,
        k_proj: Tensor,
        v_proj: Tensor,
        key_positions: Tensor,
        live_step_len: Optional[Tensor] = None,
    ) -> tuple[Nested[Tensor], BaseKVCache.Output]:
        # TODO(dhwang2): By returning only the valid portions of the KV (by live_step_len),
        # the attention complexity can be reduced from O(max_len²) to O(live_step_len²), especially
        # in prefill.
        # The remaining part after `live_step_len` is considered padding.
        del live_step_len
        if k_proj.shape != v_proj.shape:
            raise ValueError(f"{k_proj.shape=} != {v_proj.shape=}")
        if k_proj.shape[1] != key_positions.shape[1]:
            raise ValueError(f"{k_proj.shape[1]=} != {key_positions.shape[1]=}")

        cached_key: Tensor = cached_states["key"]
        cached_value: Tensor = cached_states["value"]
        batch, step_size = k_proj.shape[:2]
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
            # Create a dispatch matrix of shape [B, T=step, S].
            oh_indices = jax.nn.one_hot(key_positions, source_len, dtype=cached_key.dtype)
            # Create a mask of shape [B, 1, 1, S].
            negated_oh_indices = (1 - oh_indices.sum(axis=1))[:, None, None, :]

            k_proj = jnp.einsum("b...t,bts->b...s", k_proj, oh_indices)
            v_proj = jnp.einsum("b...t,bts->b...s", v_proj, oh_indices)

            # Ensure that we accumulate using the original dtype.
            cached_key = cached_key * negated_oh_indices + k_proj.astype(cached_key.dtype)
            cached_value = cached_value * negated_oh_indices + v_proj.astype(cached_value.dtype)
        else:
            # Note: KV transpose is an optimization for one-hot matmul and is not related to
            # dynamic_update_slice_in_dim. As a result, KV transpose only adds overhead for it.
            # Since small step_size scenarios are more frequent, we accept slowdown in this case.

            # Function to update the cache for a single batch element.
            def update_single(cached_kv_slice, kv_proj_slice, time_idx):
                return jax.lax.dynamic_update_slice_in_dim(
                    cached_kv_slice, kv_proj_slice, time_idx, axis=-1
                )

            # Use jax.vmap to vectorize over the batch dimension.
            vmap_update = jax.vmap(update_single)
            time_step = jnp.broadcast_to(key_positions[:, 0], [batch])
            cached_key = vmap_update(cached_key, k_proj.astype(cached_key.dtype), time_step)
            cached_value = vmap_update(cached_value, v_proj.astype(cached_key.dtype), time_step)

        updated_state = dict(key=cached_key, value=cached_value)
        assert updated_state["key"].shape == cached_key.shape
        assert updated_state["value"].shape == cached_value.shape

        # [B, S, N, H]
        k_proj = jnp.einsum("bnhs->bsnh", cached_key)
        v_proj = jnp.einsum("bnhs->bsnh", cached_value)
        # Currently, the part larger than live_step_len is also being overwritten in the KV cache,
        # and this part is filtered out by the causal mask through key_positions.
        key_positions = jnp.arange(k_proj.shape[1])[None]  # [1, source_length]
        return updated_state, self.Output(k_proj=k_proj, v_proj=v_proj, key_positions=key_positions)
