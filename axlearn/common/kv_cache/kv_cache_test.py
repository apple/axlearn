# Copyright © 2025 Apple Inc.

"""Tests KVCache."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import sequence_mask


class KVCacheTest(TestCase):
    """Tests KVCache."""

    @parameterized.product(
        cached_kv_length=[8],
        time_step_value=[2, 4],
        cache_dtype=[None, jnp.bfloat16],
        step_len=[-1, 2, 4],
    )
    def test_kv_cache(self, cached_kv_length, time_step_value, cache_dtype, step_len):
        test_layer = (
            KVCache.default_config()
            .set(name="ref", cache_dtype=cache_dtype)
            .instantiate(parent=None)
        )

        prng_key = jax.random.PRNGKey(2)
        batch, step_size = 2, 4
        heads, dim = 2, 2
        step_shape = (batch, step_size, heads, dim)
        k_proj = jax.random.normal(prng_key, shape=step_shape)
        v_proj = jax.random.normal(prng_key, shape=step_shape)
        key_positions = jnp.arange(step_size)[None] + time_step_value
        if step_len < 0:
            step_len = step_size
            segment_ids = None
        else:
            segment_ids = sequence_mask(
                lengths=jnp.full([batch], fill_value=step_len, dtype=jnp.int32),
                max_len=step_size,
                dtype=jnp.int32,
            )

        kv_shape = KVCache.Shape(batch, cached_kv_length, heads, dim)
        test_states = test_layer.init_states(kv_shape, dtype=k_proj.dtype)
        expect_dtype = cache_dtype or k_proj.dtype

        test_states, test_output = test_layer.extend_step(
            test_states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            segment_ids=segment_ids,
        )

        def check(input_kv, output_kv):
            self.assertEqual(output_kv.shape, kv_shape)
            self.assertEqual(output_kv.dtype, expect_dtype)
            assert_allclose(output_kv[:, :time_step_value], 0)
            assert_allclose(
                output_kv[:, time_step_value : time_step_value + step_len],
                input_kv.astype(expect_dtype)[:, :step_len],
            )

        check(k_proj, test_output.k_proj)
        check(v_proj, test_output.v_proj)
        # Output key_positions are always slot indices [0, 1, ..., source_len-1].
        assert_allclose(test_output.key_positions, jnp.arange(cached_kv_length)[None])

    def test_segment_ids(self):
        """segment_ids=[0,1,1,1,0] must not evict a previously cached valid token.

        Step 1 writes valid tokens at positions 0-4.
        Step 2 has segment_ids=[0,1,1,1,0] covering positions 3-7:
        - Position 3 (valid in step 1, padding in step 2): prior KV must survive.
        - Position 7 (fresh slot, padding in step 2): KV must remain zero (not written).
        - Positions 4,5,6 (valid in step 2): overwritten with new KV.
        """
        layer = KVCache.default_config().set(name="test").instantiate(parent=None)
        batch, heads, dim = 2, 2, 2
        cached_kv_length = 16

        prng_key = jax.random.PRNGKey(0)

        # Step 1: write 5 valid tokens at positions 0..4.
        step1_size = 5
        k1 = jax.random.normal(prng_key, shape=(batch, step1_size, heads, dim))
        v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(batch, step1_size, heads, dim))
        pos1 = jnp.arange(step1_size)[None]  # [0,1,2,3,4]
        states = layer.init_states(
            KVCache.Shape(batch, cached_kv_length, heads, dim), dtype=k1.dtype
        )
        states, _ = layer.extend_step(states, k_proj=k1, v_proj=v1, key_positions=pos1)

        # Step 2: positions [3,4,5,6,7] with padding at indices 0 and 4 (positions 3 and 7).
        step2_size = 5
        k2 = jax.random.normal(jax.random.PRNGKey(2), shape=(batch, step2_size, heads, dim))
        v2 = jax.random.normal(jax.random.PRNGKey(3), shape=(batch, step2_size, heads, dim))
        pos2 = jnp.arange(step2_size)[None] + 3  # [3,4,5,6,7]
        seg2 = jnp.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], dtype=jnp.int32)
        states, out2 = layer.extend_step(
            states, k_proj=k2, v_proj=v2, key_positions=pos2, segment_ids=seg2
        )

        # Position 3 was valid in step 1; padding in step 2 must NOT evict it.
        assert_allclose(out2.k_proj[:, 3], k1[:, 3].astype(out2.k_proj.dtype))
        assert_allclose(out2.v_proj[:, 3], v1[:, 3].astype(out2.v_proj.dtype))
        # Positions 4,5,6 overwritten by valid tokens in step 2.
        assert_allclose(out2.k_proj[:, 4:7], k2[:, 1:4].astype(out2.k_proj.dtype))
        # Position 7: padding in step 2, slot was never written — KV remains zero.
        assert_allclose(out2.k_proj[:, 7], 0)

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
        assert_allclose(onehot_output.key_positions, dynamic_output.key_positions)
        self.assertEqual(onehot_output.k_proj.dtype, dynamic_output.k_proj.dtype)
        self.assertEqual(onehot_output.v_proj.dtype, dynamic_output.v_proj.dtype)


if __name__ == "__main__":
    absltest.main()
