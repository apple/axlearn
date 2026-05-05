# Copyright © 2025 Apple Inc.

"""Tests PagedKVCacheTest."""

from functools import partial

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.kv_cache.base_kv_cache import KVState
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache import PagedKVCache
from axlearn.common.kv_cache.paged_kv_cache_gpu_kernel import gpu_scatter_update_pages_shmap_fn
from axlearn.common.kv_cache.paged_kv_cache_tpu_kernel import tpu_scatter_update_pages_shmap_fn
from axlearn.common.kv_cache.paged_kv_storage import (
    Bf16PagedStorage,
    reconstruct_kv,
    scatter_update_pages,
    scatter_update_pages_kernel,
)
from axlearn.common.test_utils import TestCase

test_fns = []
if jax.default_backend() in ("cpu", "gpu"):
    test_fns.append(gpu_scatter_update_pages_shmap_fn)
elif jax.default_backend() in ("cpu", "tpu"):
    test_fns.append(tpu_scatter_update_pages_shmap_fn)


class ScatterUpdatePagesTest(TestCase):
    @parameterized.product(
        batch_size=[1, 16],
        page_size=[4, 16, 64],
        num_heads=[1, 4],
        fn=test_fns,
    )
    def test_scatter_update(self, batch_size, page_size, num_heads, fn):
        k2, k3, k4 = jax.random.split(jax.random.PRNGKey(batch_size + num_heads), 3)
        head_dim = 128
        pages_per_seq = 32
        # Any number larger than pages_per_seq * batch_size should be fine.
        num_pages = pages_per_seq * batch_size + 11

        dtype = jnp.bfloat16

        kv_pages = jnp.zeros((num_heads, num_pages, page_size, head_dim), dtype=dtype)
        kv_proj = jax.random.normal(k2, (num_heads, batch_size, 1, head_dim), dtype=dtype)
        page_indices = jax.random.choice(
            k3, jnp.arange(1, num_pages), shape=(batch_size, pages_per_seq), replace=False
        )
        key_positions = jax.random.choice(
            k4,
            jnp.arange(0, pages_per_seq * page_size + 512),
            shape=(batch_size, 1),
            replace=True,
        )

        with jax.make_mesh((1,), ("data",), devices=jax.devices()[0:1]):
            ref = scatter_update_pages(kv_pages, kv_proj, page_indices, key_positions)
            out = jax.jit(partial(scatter_update_pages_kernel, shmap_fn=fn))(
                kv_pages=kv_pages.copy(),
                kv_proj=kv_proj,
                page_indices=page_indices,
                key_positions=key_positions,
            )
            self.assertNestedAllClose(out, ref)


class PagedKVCacheTest(TestCase):
    @parameterized.product(
        time_step_value=[5],
        cache_dtype=[jnp.bfloat16],
        max_pages_each_request=[64],
        page_size=[64],
    )
    def test_paged_kv_cache(
        self,
        time_step_value,
        cache_dtype,
        max_pages_each_request,
        page_size,
    ):
        with jax.make_mesh((1, 1), ("data", "model"), devices=[jax.devices()[0]]):
            test_layer = (
                PagedKVCache.default_config()
                .set(name="test", cache_dtype=cache_dtype)
                .instantiate(parent=None)
            )
            ref_layer = (
                KVCache.default_config()
                .set(name="ref", cache_dtype=cache_dtype)
                .instantiate(parent=None)
            )

            prng_key = jax.random.PRNGKey(42)
            batch, step_len = 32, 1
            heads, dim = 2, 128
            # As long as this is >= batch * max_pages_each_request it's fine.
            total_pages = batch * max_pages_each_request + 97

            step_shape = (batch, step_len, heads, dim)
            k_proj = jax.random.normal(prng_key, shape=step_shape, dtype=cache_dtype)
            v_proj = jax.random.normal(prng_key, shape=step_shape, dtype=cache_dtype)
            key_positions = jnp.full((batch, 1), time_step_value, dtype=jnp.int32)

            # TODO(xiyou): consider segment_ids when it's supported
            segment_ids = None

            kv_shape = KVCache.Shape(batch, max_pages_each_request * page_size, heads, dim)
            ref_states = ref_layer.init_states(kv_shape, dtype=k_proj.dtype)
            test_states = test_layer.init_states(kv_shape, dtype=k_proj.dtype)
            self.assertEmpty(test_states, "PagedKVCache init states should be empty.")
            page_indices = jnp.zeros((batch, max_pages_each_request), dtype=jnp.int32)

            # Not all batches could be active. Test when only some batches are active.
            start_batch = 1
            end_batch = 3
            page_indices = page_indices.at[start_batch:end_batch].set(
                jnp.arange((end_batch - start_batch) * max_pages_each_request).reshape(
                    (-1, max_pages_each_request)
                )
                + 1
            )
            key_positions = key_positions.at[:start_batch].set(0).at[end_batch:].set(0)

            # This parts are done outside PagedKVCache Layer for performance.
            page_shape = (heads, total_pages, page_size, dim)
            test_states["page_indices"] = page_indices
            test_states["key"] = jnp.zeros(page_shape, dtype=cache_dtype)
            test_states["value"] = jnp.zeros(page_shape, dtype=cache_dtype)

            @partial(jax.jit, static_argnums=(0,))
            def jit_extend_step(
                layer: KVCache, test_states, k_proj, v_proj, key_positions, segment_ids
            ):
                _, test_output = layer.extend_step(
                    test_states,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    key_positions=key_positions,
                    segment_ids=segment_ids,
                )
                return test_output

            ref_out: KVState = jit_extend_step(
                ref_layer, ref_states, k_proj, v_proj, key_positions, segment_ids
            )
            test_out: KVState = jit_extend_step(
                test_layer, test_states, k_proj, v_proj, key_positions, segment_ids
            )

            test_k_proj = reconstruct_kv(page_indices, test_out.k_proj)
            test_v_proj = reconstruct_kv(page_indices, test_out.v_proj)
            self.assertNestedEqual(
                ref_out.k_proj[start_batch:end_batch], test_k_proj[start_batch:end_batch]
            )
            self.assertNestedEqual(
                ref_out.v_proj[start_batch:end_batch], test_v_proj[start_batch:end_batch]
            )


class PagedKVCacheAsDenseKvTest(TestCase):
    """Freeze `PagedKVCache.as_dense_kv` dispatch.

    Downstream callers (e.g. `attention.py`'s `_compute_attention`) rely on
    this method to hand back dense `(k, v)` regardless of whether
    `extend_step` emitted a legacy `KVState` or a `PagedKVStorage` variant.
    """

    def _make_bf16_storage(
        self,
        *,
        batch: int = 2,
        num_heads: int = 3,
        num_pages: int = 8,
        page_size: int = 4,
        head_dim: int = 16,
        pages_per_request: int = 3,
        seed: int = 0,
    ) -> Bf16PagedStorage:
        k_key, v_key = jax.random.split(jax.random.PRNGKey(seed), 2)
        k_pages = jax.random.normal(
            k_key, (num_heads, num_pages, page_size, head_dim), dtype=jnp.bfloat16
        )
        v_pages = jax.random.normal(
            v_key, (num_heads, num_pages, page_size, head_dim), dtype=jnp.bfloat16
        )
        page_indices = jnp.arange(1, batch * pages_per_request + 1, dtype=jnp.int32).reshape(
            batch, pages_per_request
        )
        key_positions = jnp.broadcast_to(
            jnp.arange(pages_per_request * page_size, dtype=jnp.int32),
            (batch, pages_per_request * page_size),
        )
        return Bf16PagedStorage(
            k_proj=k_pages,
            v_proj=v_pages,
            page_indices=page_indices,
            key_positions=key_positions,
        )

    def test_dense_kv_state_falls_through_to_base(self):
        k = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2, 8), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 2, 8), dtype=jnp.bfloat16)
        kp = jnp.arange(4, dtype=jnp.int32)[None]
        state = KVState(k_proj=k, v_proj=v, key_positions=kp)

        k_out, v_out = PagedKVCache.as_dense_kv(state)

        # `page_indices is None` → identity, no reconstruction.
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_kv_state_with_page_indices_is_reconstructed(self):
        # Legacy `PagedKVCache.extend_step` decode emission: `KVState` with
        # `k_proj` / `v_proj` holding the page pool and `page_indices`
        # populated. `as_dense_kv` must reconstruct to dense.
        num_heads, num_pages, page_size, head_dim = 3, 8, 4, 16
        batch, pages_per_request = 2, 3
        k_pages = jax.random.normal(
            jax.random.PRNGKey(0),
            (num_heads, num_pages, page_size, head_dim),
            dtype=jnp.bfloat16,
        )
        v_pages = jax.random.normal(
            jax.random.PRNGKey(1),
            (num_heads, num_pages, page_size, head_dim),
            dtype=jnp.bfloat16,
        )
        page_indices = jnp.arange(1, batch * pages_per_request + 1, dtype=jnp.int32).reshape(
            batch, pages_per_request
        )
        key_positions = jnp.broadcast_to(
            jnp.arange(pages_per_request * page_size, dtype=jnp.int32),
            (batch, pages_per_request * page_size),
        )
        state = KVState(
            k_proj=k_pages,
            v_proj=v_pages,
            key_positions=key_positions,
            page_indices=page_indices,
        )

        k_out, v_out = PagedKVCache.as_dense_kv(state)

        self.assertEqual(k_out.shape, (batch, pages_per_request * page_size, num_heads, head_dim))
        self.assertEqual(v_out.shape, (batch, pages_per_request * page_size, num_heads, head_dim))
        self.assertTrue(jnp.array_equal(k_out, reconstruct_kv(page_indices, k_pages)))
        self.assertTrue(jnp.array_equal(v_out, reconstruct_kv(page_indices, v_pages)))

    def test_paged_storage_returns_dense_reconstruction(self):
        storage = self._make_bf16_storage()

        k_out, v_out = PagedKVCache.as_dense_kv(storage)

        # Must materialise dense tensors from the page pool — NOT hand back
        # the pool (`[num_kv_heads, total_num_pages, page_size, head_dim]`)
        # as-is. That was the silent-corruption failure mode flagged in review.
        batch, pages_per_request = storage.page_indices.shape
        num_kv_heads, _, page_size, head_dim = storage.k_proj.shape
        self.assertEqual(
            k_out.shape, (batch, pages_per_request * page_size, num_kv_heads, head_dim)
        )
        self.assertEqual(
            v_out.shape, (batch, pages_per_request * page_size, num_kv_heads, head_dim)
        )
        self.assertTrue(
            jnp.array_equal(k_out, reconstruct_kv(storage.page_indices, storage.k_proj))
        )
        self.assertTrue(
            jnp.array_equal(v_out, reconstruct_kv(storage.page_indices, storage.v_proj))
        )


if __name__ == "__main__":
    absltest.main()
