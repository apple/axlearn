# Copyright © 2025 Apple Inc.

"""Tests SlidingWindowKVCacheTest."""

import jax
import jax.numpy as jnp
from absl.testing import parameterized

from axlearn.common.kv_cache.sliding_window_kv_cache import SlidingWindowKVCache
from axlearn.common.test_utils import TestCase, assert_allclose


class SlidingWindowKVCacheTest(TestCase):
    """Tests SlidingWindowKVCache."""

    @parameterized.product(cached_kv_length=[8], time_step_value=[2, 4, 6], live_step_len=[None, 2])
    def test_sliding_window_kv_cache(self, cached_kv_length, time_step_value, live_step_len):
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
        valid_out_len = live_step_len or step_len
        live_step_len = (
            jnp.full([batch], fill_value=live_step_len) if live_step_len is not None else None
        )
        _, test_output = test_layer.extend_step(
            test_states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            live_step_len=live_step_len,
        )
        kv_shape = (2, cached_kv_length + step_len, 2, 2)
        self.assertEqual(test_output.key_positions.shape, kv_shape[:2])
        test_key_positions = test_output.key_positions[:, -valid_out_len:]
        assert_allclose(
            test_key_positions,
            jnp.broadcast_to(key_positions[:, :valid_out_len], test_key_positions.shape),
        )
        test_mask = (test_output.key_positions[:, :-valid_out_len] >= 0)[:, :, None, None]
        self.assertEqual(test_output.k_proj.shape, kv_shape)
        assert_allclose(test_output.k_proj[:, :-valid_out_len] * test_mask, 0)
        assert_allclose(test_output.k_proj[:, -valid_out_len:], k_proj[:, :valid_out_len])
        self.assertEqual(test_output.v_proj.shape, kv_shape)
        assert_allclose(test_output.v_proj[:, :-valid_out_len] * test_mask, 0)
        assert_allclose(test_output.v_proj[:, -valid_out_len:], v_proj[:, :valid_out_len])
