# Copyright © 2025 Apple Inc.

"""Tests PagedKVCacheTest."""

from functools import partial

import jax
import jax.numpy as jnp
from absl.testing import parameterized

from axlearn.common.kv_cache.base_kv_cache import KVState
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache import (
    PagedKVCache,
    reconstruct_kv,
    scatter_update_pages,
    scatter_update_pages_kernel,
)
from axlearn.common.kv_cache.paged_kv_cache_gpu_kernel import gpu_scatter_update_pages_shmap_fn
from axlearn.common.kv_cache.paged_kv_cache_tpu_kernel import tpu_scatter_update_pages_shmap_fn
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

            # TODO(xiyou): consider unpadded_len when it's supported
            unpadded_len = None

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
                layer: KVCache, test_states, k_proj, v_proj, key_positions, unpadded_len
            ):
                _, test_output = layer.extend_step(
                    test_states,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    key_positions=key_positions,
                    unpadded_len=unpadded_len,
                )
                return test_output

            ref_out: KVState = jit_extend_step(
                ref_layer, ref_states, k_proj, v_proj, key_positions, unpadded_len
            )
            test_out: KVState = jit_extend_step(
                test_layer, test_states, k_proj, v_proj, key_positions, unpadded_len
            )

            test_k_proj = reconstruct_kv(page_indices, test_out.k_proj)
            test_v_proj = reconstruct_kv(page_indices, test_out.v_proj)
            self.assertNestedEqual(
                ref_out.k_proj[start_batch:end_batch], test_k_proj[start_batch:end_batch]
            )
            self.assertNestedEqual(
                ref_out.v_proj[start_batch:end_batch], test_v_proj[start_batch:end_batch]
            )
