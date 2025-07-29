# Copyright © 2025 Apple Inc.

"""A KVCache stores KV tokens in paged format."""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax._src.mesh import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from axlearn.common.kv_cache.base_kv_cache import KVState
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache_gpu_kernel import gpu_scatter_update_pages_shmap_fn
from axlearn.common.kv_cache.paged_kv_cache_tpu_kernel import tpu_scatter_update_pages_shmap_fn
from axlearn.common.utils import Nested, Tensor


def reconstruct_kv(page_tables: Tensor, pages: Tensor) -> Tensor:
    """Retrieve key/value from page tables given pages.

    Args:
        page_tables: [batch_size, pages_per_sequence], specifying page indices.
        pages: [num_kv_heads, total_num_pages, page_size, head_dim], k/v pages.

    Returns:
        Retrieved actual key / value of shape [batch_size, kv_seq_len, n_kv_heads, head_dim],
            where kv_seq_len = pages_per_sequence * page_size.
    """
    temp = jnp.einsum("nbpsh->bpsnh", pages.at[:, page_tables].get(mode="fill", fill_value=0))
    b, _, _, n, h = temp.shape
    return temp.reshape(b, -1, n, h)


def scatter_update_pages(
    kv_pages: Tensor, kv_proj: Tensor, page_indices: Tensor, key_positions: Tensor
) -> Tensor:
    """Scatter kv_proj into kv_pages according to key_positions.

    Args:
        kv_pages: A tensor of shape [num_heads, num_pages, page_size, head_dim].
        kv_proj: A tensor of shape [num_heads, batch_size, 1, head_dim].
        page_indices: A tensor of shape [batch_size, pages_per_batch].
        key_positions: A tensor of shape [batch_size, 1].

    Returns:
        A tensor with the same shape as `kv_pages`.
    """
    page_size = kv_pages.shape[-2]
    offset_in_page = key_positions % page_size  # (batch, step)
    page_idx = key_positions // page_size  # (batch, step)
    page_idx = jnp.take_along_axis(page_indices, page_idx, axis=1)
    # unique_indices=True provides 10x faster perf for the scatter kernel on GPU.
    kv_pages = kv_pages.at[:, page_idx, offset_in_page].set(
        kv_proj, unique_indices=True, mode="drop"
    )
    return kv_pages


def scatter_update_pages_kernel(
    *,
    kv_pages: Tensor,
    kv_proj: Tensor,
    page_indices: Tensor,
    key_positions: Tensor,
    shmap_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Equivalent to `scatter_update_pages` but much faster on TPU and GPU.

    Args:
        kv_pages: A tensor of shape [num_heads, num_pages, page_size, head_dim].
        kv_proj: A tensor of shape [num_heads, batch_size, 1, head_dim].
        page_indices: A tensor of shape [batch_size, pages_per_batch].
        key_positions: A tensor of shape [batch_size, 1].
        shmap_fn: A callable that takes in the tensors above and returns an updated `kv_pages` in
            a shard_map context.

    Returns:
        A tensor with the same shape as `kv_pages`.
    """
    key_positions = key_positions.squeeze(1)
    mesh = thread_resources.env.physical_mesh
    model_axis = "model"
    if model_axis not in mesh.axis_names:
        model_axis = None
    return shard_map(
        shmap_fn,
        mesh=mesh,
        in_specs=(
            PartitionSpec(model_axis, None, None, None),
            PartitionSpec(model_axis, None, None, None),
            PartitionSpec(None, None),
            PartitionSpec(None),
        ),
        out_specs=PartitionSpec(model_axis, None, None, None),
        check_rep=False,
    )(kv_pages, kv_proj, page_indices, key_positions)


class PagedKVCache(KVCache):
    """Paged KV cache.

    During Prefill:
    1. init_states does nothing.
    2. page_indices is not passed in, extend_step would return return k_proj and v_proj directly.

    During Decode:
    1. Only extend_step is called.
    2. Basically step len is 1 (unless we want to support speculative decoding in the future).
       We assume the last page for each sequence is not full. Fill the new k/v_proj into the
       last token in the last page for each sequence.
    """

    PADDING_PAGE_ID = 0

    def init_states(self, shape: KVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        """Initialize the KV States.

        For PagedKVCache, page indices and actual kv pages should be populated outside the class
        because prefill would be done for each request while the global page allocation is done
        only once for all the requests, and all pages are stored in a batch-agnostic centralized
        kv pages tensor.
        """
        return {}

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        *,
        k_proj: Tensor,
        v_proj: Tensor,
        key_positions: Tensor,
        unpadded_len: Optional[Tensor] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Tensor], KVCache.Output]:
        """Extend the cache with the new key and value.

        cached_states contains the following keys:
        * key: A Tensor of shape [num_heads, max_pages_global, page_size, head_dim] representing
          physical key pages.
        * value: A Tensor of shape [num_heads, max_pages_global, page_size, head_dim] representing
          physical value pages.
        * page_indices: A Tensor of shape [batch_size, max_pages_per_request] indicating the page
          indices of the current batch. Each value is an index into physical pages with the range
          [0, max_pages_global).
            None is treated as prefill.
            The caller is responsible for populating this Tensor.

        Basically does the same thing as the following for-loop:

        for i in range(batch):
            for j in range(steps):
                page_idx = key_positions[i, j] // page_size
                page_offset = key_positions[i, j] % page_size
                actual_page_idx = page_indices[i, page_idx]
                for k in range(num_heads):
                    k_pages = k_pages.at[k, actual_page_idx, page_offset].set(k_proj[i, j, k, :])
                    v_pages = v_pages.at[k, actual_page_idx, page_offset].set(v_proj[i, j, k, :])
        """
        del unpadded_len

        if k_proj.shape != v_proj.shape:
            raise ValueError(f"{k_proj.shape=} != {v_proj.shape=}")
        if k_proj.shape[1] != key_positions.shape[1]:
            raise ValueError(f"{k_proj.shape[1]=} != {key_positions.shape[1]=}")

        if "page_indices" not in cached_states:
            assert page_pool is None
            # Prefill, return kv cache directly
            cached_states["key"] = k_proj
            cached_states["value"] = v_proj
            return cached_states, self.Output(
                k_proj=k_proj, v_proj=v_proj, key_positions=key_positions
            )

        page_indices: Tensor = cached_states["page_indices"]

        # kv_pages shape: [num_heads, max_pages_global, page_size, head_dim]. Also refer to
        # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L388
        if page_pool is not None:
            # We use `group_info` to index into `page_pool` to get the paged KV pool for this
            # layer.
            group_info = cached_states["group_info"]
            # HACK(hanzhi-zhou): we store the indices as dict keys to workaround them being
            # converted to tracers.
            group_idx = list(group_info["group_idx"].keys())[0]
            repeat_idx = list(group_info["repeat_idx"].keys())[0]
            pool = page_pool[group_idx][repeat_idx]
            k_pages: Tensor = pool.k_pages  # type: ignore
            v_pages: Tensor = pool.v_pages  # type: ignore
        else:
            k_pages: Tensor = cached_states["key"]
            v_pages: Tensor = cached_states["value"]

        assert k_pages.shape == v_pages.shape

        batch = page_indices.shape[0]
        page_size, _ = k_pages.shape[-2:]
        steps = key_positions.shape[1]

        # TODO(@xiyou): enable efficient multi-step extend step
        assert steps == 1, "Currently only support single time step."

        if key_positions.shape[0] == 1:
            key_positions = jnp.broadcast_to(key_positions, (batch, steps))
        else:
            assert key_positions.shape[0] == batch, (key_positions.shape[0], batch)

        def update_kv_pages(kv_pages, page_indices, key_positions, kv_proj):
            # Compute global page indices and offsets.
            kv_proj = jnp.einsum("bsnd->nbsd", kv_proj)
            update_fn = scatter_update_pages
            if jax.default_backend() == "tpu":
                update_fn = partial(
                    scatter_update_pages_kernel, shmap_fn=tpu_scatter_update_pages_shmap_fn
                )
            elif jax.default_backend() == "gpu":
                update_fn = partial(
                    scatter_update_pages_kernel, shmap_fn=gpu_scatter_update_pages_shmap_fn
                )
            kv_pages = update_fn(
                kv_pages=kv_pages,
                kv_proj=kv_proj,
                page_indices=page_indices,
                key_positions=key_positions,
            )
            return kv_pages

        updated_k_pages = update_kv_pages(
            k_pages, page_indices, key_positions, k_proj.astype(k_pages.dtype)
        )
        updated_v_pages = update_kv_pages(
            v_pages, page_indices, key_positions, v_proj.astype(v_pages.dtype)
        )

        if page_pool is not None:
            page_pool[group_idx][repeat_idx] = type(pool)(updated_k_pages, updated_v_pages)

            # Updates are already performed through mutable arrays above. We don't perform state
            # updates through `updated_state`.
            updated_state = dict(key=None, value=None, page_indices=None)
        else:
            updated_state = dict(
                key=updated_k_pages,
                value=updated_v_pages,
                page_indices=page_indices,
            )

        assert updated_k_pages.shape == k_pages.shape
        assert updated_v_pages.shape == v_pages.shape

        # total time step length is num_pages_per_request x page_size = kv_len
        key_positions = jnp.arange(page_indices.shape[1] * page_size)[None]

        return updated_state, self.Output(
            k_proj=updated_k_pages,
            v_proj=updated_v_pages,
            key_positions=key_positions,
            page_indices=page_indices,
        )

    @classmethod
    def maybe_normalize_kv(cls, kv_state: KVState) -> tuple[Tensor, Tensor]:
        """See `BaseKVCache.maybe_normalize_kv`."""
        if kv_state.page_indices is None:
            # page_indices is None during prefill.
            return kv_state.k_proj, kv_state.v_proj
        k_proj = reconstruct_kv(kv_state.page_indices, kv_state.k_proj)
        v_proj = reconstruct_kv(kv_state.page_indices, kv_state.v_proj)
        return k_proj, v_proj
