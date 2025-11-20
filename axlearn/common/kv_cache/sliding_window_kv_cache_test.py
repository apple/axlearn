# Copyright © 2025 Apple Inc.

"""Tests SlidingWindowKVCacheTest."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.kv_cache.sliding_window_kv_cache import SlidingWindowKVCache
from axlearn.common.test_utils import TestCase, assert_allclose


class SlidingWindowKVCacheTest(TestCase):
    """Tests SlidingWindowKVCache."""

    @parameterized.product(cached_kv_length=[8], time_step_value=[2, 4, 6], unpadded_len=[None, 2])
    def test_sliding_window_kv_cache(self, cached_kv_length, time_step_value, unpadded_len):
        test_layer = (
            SlidingWindowKVCache.default_config()
            .set(name="ref", cached_kv_length=cached_kv_length)
            .instantiate(parent=None)
        )
        batch, step_len = 2, 4
        step_shape = (batch, step_len, 2, 2)
        test_states = test_layer.init_states(
            shape=SlidingWindowKVCache.Shape(*step_shape), dtype=jnp.float32
        )
        prng_key = jax.random.PRNGKey(2)
        k_proj = jax.random.normal(prng_key, shape=step_shape)
        v_proj = jax.random.normal(prng_key, shape=step_shape)
        key_positions = jnp.arange(step_len)[None] + time_step_value
        valid_out_len = unpadded_len or step_len
        unpadded_len = (
            jnp.full([batch], fill_value=unpadded_len) if unpadded_len is not None else None
        )
        updated_states, test_output = test_layer.extend_step(
            test_states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            unpadded_len=unpadded_len,
        )
        kv_shape = (2, cached_kv_length + step_len, 2, 2)
        self.assertEqual(test_output.key_positions.shape, kv_shape[:2])
        test_key_positions = test_output.key_positions[:, -step_len:][:, :valid_out_len]
        assert_allclose(
            test_key_positions,
            jnp.broadcast_to(key_positions[:, :valid_out_len], test_key_positions.shape),
        )
        self.assertEqual(test_output.k_proj.shape, kv_shape)
        test_k_proj = test_output.k_proj[:, -step_len:][:, :valid_out_len]
        assert_allclose(test_k_proj, k_proj[:, :valid_out_len])
        test_v_proj = test_output.v_proj[:, -step_len:][:, :valid_out_len]
        assert_allclose(test_v_proj, v_proj[:, :valid_out_len])

        updated_pos = updated_states["key_positions"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_pos, test_key_positions[:, : updated_pos.shape[1]])
        updated_states["key"] = jnp.einsum("bnhs->bsnh", updated_states["key"])
        updated_key = updated_states["key"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_key, test_k_proj[:, : updated_key.shape[1]])
        updated_states["value"] = jnp.einsum("bnhs->bsnh", updated_states["value"])
        updated_value = updated_states["value"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_value, test_v_proj[:, : updated_value.shape[1]])


if __name__ == "__main__":
    absltest.main()
