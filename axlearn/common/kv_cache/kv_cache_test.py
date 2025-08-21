# Copyright © 2025 Apple Inc.

"""Tests KVCache."""

import jax
import jax.numpy as jnp
from absl.testing import parameterized

from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.test_utils import TestCase, assert_allclose


class KVCacheTest(TestCase):
    """Tests KVCache."""

    @parameterized.product(
        cached_kv_length=[8],
        time_step_value=[2, 4],
        cache_dtype=[None, jnp.bfloat16],
        unpadded_len=[-1, 2, 4],
    )
    def test_kv_cache(self, cached_kv_length, time_step_value, cache_dtype, unpadded_len):
        test_layer = (
            KVCache.default_config()
            .set(name="ref", cache_dtype=cache_dtype)
            .instantiate(parent=None)
        )

        prng_key = jax.random.PRNGKey(2)
        batch, step_len = 2, 4
        heads, dim = 2, 2
        step_shape = (batch, step_len, heads, dim)
        k_proj = jax.random.normal(prng_key, shape=step_shape)
        v_proj = jax.random.normal(prng_key, shape=step_shape)
        key_positions = jnp.arange(step_len)[None] + time_step_value
        if unpadded_len < 0:
            valid_step_len = step_len
            unpadded_len = None
        else:
            valid_step_len = unpadded_len
            unpadded_len = jnp.full([batch], fill_value=unpadded_len, dtype=jnp.int32)

        kv_shape = KVCache.Shape(batch, cached_kv_length, heads, dim)
        test_states = test_layer.init_states(kv_shape, dtype=k_proj.dtype)
        expect_dtype = cache_dtype or k_proj.dtype

        _, test_output = test_layer.extend_step(
            test_states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            unpadded_len=unpadded_len,
        )

        def check(input_kv, output_kv):
            self.assertEqual(output_kv.shape, kv_shape)
            self.assertEqual(output_kv.dtype, expect_dtype)
            assert_allclose(output_kv[:, :time_step_value], 0)
            assert_allclose(
                output_kv[:, time_step_value : time_step_value + valid_step_len],
                input_kv.astype(expect_dtype)[:, :valid_step_len],
            )

        check(k_proj, test_output.k_proj)
        check(v_proj, test_output.v_proj)
        key_positions = jnp.arange(cached_kv_length)[None]
        assert_allclose(test_output.key_positions, key_positions)
        # Currently, the part larger than unpadded_len is also being overwritten in the KV cache.
        # TODO(dhwang2): remove this check when KVCache updates only valid part.
        assert_allclose(
            test_output.k_proj[:, time_step_value : time_step_value + step_len],
            k_proj.astype(expect_dtype)[:, :step_len],
        )

    @parameterized.product(cache_dtype=[None, jnp.bfloat16])
    def test_kv_cache_onehot_vs_dynamic(self, cache_dtype):
        test_layer = (
            KVCache.default_config()
            .set(name="test", cache_dtype=cache_dtype)
            .instantiate(parent=None)
        )

        kv_len = 64
        kv_shape = KVCache.Shape(2, kv_len, 2, 2)
        onehot_states = test_layer.init_states(kv_shape, dtype=jnp.float32)
        dynamic_states = test_layer.init_states(kv_shape, dtype=jnp.float32)

        prng_key = jax.random.PRNGKey(2)
        k_proj = jax.random.normal(prng_key, shape=kv_shape)
        v_proj = jax.random.normal(prng_key, shape=kv_shape)

        def extend_step(step_size, cached_states):
            for i in range(0, kv_len, step_size):
                k_step = k_proj[:, i : i + step_size]
                v_step = v_proj[:, i : i + step_size]
                key_positions = jnp.arange(step_size)[None] + i
                cached_states, test_output = test_layer.extend_step(
                    cached_states, k_proj=k_step, v_proj=v_step, key_positions=key_positions
                )
            return cached_states, test_output

        onehot_states, onehot_output = extend_step(step_size=1, cached_states=onehot_states)
        dynamic_states, dynamic_output = extend_step(step_size=32, cached_states=dynamic_states)

        expect_dtype = cache_dtype or k_proj.dtype
        assert_allclose(onehot_states["key"], dynamic_states["key"])
        assert_allclose(onehot_states["value"], dynamic_states["value"])
        self.assertEqual(onehot_states["key"].dtype, dynamic_states["key"].dtype)
        self.assertEqual(onehot_states["value"].dtype, dynamic_states["value"].dtype)
        self.assertEqual(onehot_states["key"].dtype, expect_dtype)
        self.assertEqual(onehot_states["value"].dtype, expect_dtype)

        assert_allclose(onehot_output.k_proj, dynamic_output.k_proj)
        assert_allclose(onehot_output.v_proj, dynamic_output.v_proj)
        self.assertEqual(onehot_output.k_proj.dtype, dynamic_output.k_proj.dtype)
        self.assertEqual(onehot_output.v_proj.dtype, dynamic_output.v_proj.dtype)

        assert_allclose(onehot_output.key_positions, dynamic_output.key_positions)
