# Copyright © 2025 Apple Inc.

"""Tests SlidingWindowKVCacheTest."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.kv_cache.sliding_window_kv_cache import SlidingWindowKVCache
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import sequence_mask


class SlidingWindowKVCacheTest(TestCase):
    """Tests SlidingWindowKVCache."""

    @parameterized.product(cached_kv_length=[8], time_step_value=[2, 4, 6], step_len=[None, 2])
    def test_sliding_window_kv_cache(self, cached_kv_length, time_step_value, step_len):
        test_layer = (
            SlidingWindowKVCache.default_config()
            .set(name="ref", cached_kv_length=cached_kv_length)
            .instantiate(parent=None)
        )
        batch, step_size = 2, 4
        step_shape = (batch, step_size, 2, 2)
        test_states = test_layer.init_states(
            shape=SlidingWindowKVCache.Shape(*step_shape), dtype=jnp.float32
        )
        prng_key = jax.random.PRNGKey(2)
        k_proj = jax.random.normal(prng_key, shape=step_shape)
        v_proj = jax.random.normal(prng_key, shape=step_shape)
        key_positions = jnp.arange(step_size)[None] + time_step_value
        valid_out_len = step_len or step_size
        segment_ids = (
            sequence_mask(
                lengths=jnp.full([batch], fill_value=step_len, dtype=jnp.int32),
                max_len=step_size,
                dtype=jnp.int32,
            )
            if step_len is not None
            else None
        )
        updated_states, test_output = test_layer.extend_step(
            test_states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            segment_ids=segment_ids,
        )
        kv_shape = (2, cached_kv_length + step_size, 2, 2)
        self.assertEqual(test_output.key_positions.shape, kv_shape[:2])
        test_key_positions = test_output.key_positions[:, -step_size:][:, :valid_out_len]
        assert_allclose(
            test_key_positions,
            jnp.broadcast_to(key_positions[:, :valid_out_len], test_key_positions.shape),
        )
        self.assertEqual(test_output.k_proj.shape, kv_shape)
        test_k_proj = test_output.k_proj[:, -step_size:][:, :valid_out_len]
        assert_allclose(test_k_proj, k_proj[:, :valid_out_len])
        test_v_proj = test_output.v_proj[:, -step_size:][:, :valid_out_len]
        assert_allclose(test_v_proj, v_proj[:, :valid_out_len])

        updated_pos = updated_states["key_positions"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_pos, test_key_positions[:, : updated_pos.shape[1]])
        updated_states["key"] = jnp.einsum("bnhs->bsnh", updated_states["key"])
        updated_key = updated_states["key"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_key, test_k_proj[:, : updated_key.shape[1]])
        updated_states["value"] = jnp.einsum("bnhs->bsnh", updated_states["value"])
        updated_value = updated_states["value"][:, time_step_value:][:, :valid_out_len]
        assert_allclose(updated_value, test_v_proj[:, : updated_value.shape[1]])

    def test_segment_ids(self):
        """segment_ids=[0,1,1,1,0]: leading/trailing padding must not enter the ring buffer.

        Contiguous trailing padding ([1,1,0,0]) is covered by `test_sliding_window_kv_cache`.
        This test verifies that a leading padding token is also excluded from the ring buffer,
        i.e. the two padding slots must remain at the invalid sentinel value.
        """
        cached_kv_length, batch, step_size, time_step = 8, 2, 5, 3
        layer = (
            SlidingWindowKVCache.default_config()
            .set(name="test", cached_kv_length=cached_kv_length)
            .instantiate(parent=None)
        )
        invalid = -(cached_kv_length + 1)
        states = layer.init_states(
            shape=SlidingWindowKVCache.Shape(batch, step_size, 2, 2), dtype=jnp.float32
        )
        prng_key = jax.random.PRNGKey(42)
        k_proj = jax.random.normal(prng_key, shape=(batch, step_size, 2, 2))
        v_proj = jax.random.normal(prng_key, shape=(batch, step_size, 2, 2))
        key_positions = jnp.arange(step_size)[None] + time_step  # [3, 4, 5, 6, 7]
        segment_ids = jnp.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], dtype=jnp.int32)

        _, output = layer.extend_step(
            states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            segment_ids=segment_ids,
        )

        # output.key_positions[-step_size:] reflects the current step (cached part precedes it).
        out_pos = output.key_positions[:, -step_size:]
        # Valid tokens at indices 1,2,3 must carry real positions.
        for i in range(1, 4):
            assert_allclose(out_pos[:, i], time_step + i)
        # Padding tokens at indices 0 and 4 must be the invalid sentinel.
        # They are not stored in the ring buffer.
        assert_allclose(out_pos[:, 0], invalid)
        assert_allclose(out_pos[:, 4], invalid)


if __name__ == "__main__":
    absltest.main()
