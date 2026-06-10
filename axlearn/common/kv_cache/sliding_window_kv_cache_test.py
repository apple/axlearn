# Copyright © 2025 Apple Inc.

"""Tests SlidingWindowKVCacheTest."""

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized
from jax.sharding import PartitionSpec

from axlearn.common.attention import MultiheadAttention
from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.sliding_window_kv_cache import (
    SlidingWindowKVCache,
    enable_sliding_window_attention,
)
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

    @pytest.mark.for_8_devices
    def test_init_states_kv_partition_spec(self):
        """Verify init_states applies kv_partition_spec sharding to key/value and batch-shards
        key_positions."""
        if jax.device_count() < 8:
            self.skipTest("Requires 8+ devices")
        batch, heads, dim, cached_kv_length = 4, 8, 4, 16
        kv_partition_spec = (("data",), None, "model", None)
        layer = (
            SlidingWindowKVCache.default_config()
            .set(
                name="test", cached_kv_length=cached_kv_length, kv_partition_spec=kv_partition_spec
            )
            .instantiate(parent=None)
        )
        shape = SlidingWindowKVCache.Shape(
            batch_size=batch, kv_len=32, num_kv_heads=heads, per_head_dim=dim
        )

        with jax.make_mesh((4, 2), ("data", "model")):

            @jax.jit
            def f():
                return layer.init_states(shape=shape, dtype=jnp.float32)

            state = f()

        for name in ("key", "value"):
            self.assertEqual(state[name].shape, (batch, heads, dim, cached_kv_length))
            spec = state[name].sharding.spec
            self.assertEqual(spec, PartitionSpec("data", "model"))

        self.assertEqual(state["key_positions"].shape, (batch, cached_kv_length))
        kp_spec = state["key_positions"].sharding.spec
        self.assertEqual(kp_spec, PartitionSpec("data"))

    def test_key_positions_precision_with_bfloat16_cache(self):
        """Regression test: positions ≥ cache_len must be stored exactly with bfloat16 cache."""
        cached_kv_length = 4096
        prompt_len = 5000  # > cached_kv_length to trigger ring wrap
        max_seq_len = 6000
        batch, n_heads, head_dim = 1, 1, 32
        layer = (
            SlidingWindowKVCache.default_config()
            .set(name="bf16", cached_kv_length=cached_kv_length)
            .instantiate(parent=None)
        )
        states = layer.init_states(
            shape=SlidingWindowKVCache.Shape(
                batch_size=batch, kv_len=max_seq_len, num_kv_heads=n_heads, per_head_dim=head_dim
            ),
            dtype=jnp.bfloat16,  # must trigger bf16 path
        )
        key_positions = jnp.arange(max_seq_len)[None, :]
        segment_ids = jnp.where(jnp.arange(max_seq_len) < prompt_len, 1, 0)[None, :]
        k_proj = jnp.zeros((batch, max_seq_len, n_heads, head_dim), dtype=jnp.bfloat16)
        v_proj = jnp.zeros_like(k_proj)
        new_state, _ = layer.extend_step(
            states,
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=key_positions,
            segment_ids=segment_ids,
        )
        # Expected: each ring slot s holds the most-recent valid position p with p % cache_len == s
        # and p in [max(0, prompt_len - cache_len), prompt_len - 1].
        stored = new_state["key_positions"][0]
        # Slot 0: positions {0, 4096} ∩ [904, 4999] = {4096}.
        self.assertEqual(int(stored[0]), 4096)
        # Slot 1: positions {1, 4097} ∩ [904, 4999] = {4097} (would be 4096 if bf16 corrupts).
        self.assertEqual(int(stored[1]), 4097)
        # Slot 100: {100, 4196} ∩ [904, 4999] = {4196} (would be 4192 if bf16 corrupts).
        self.assertEqual(int(stored[100]), 4196)
        # Slot 903: {903, 4999} ∩ [904, 4999] = {4999} (would be 4992 if bf16 corrupts).
        self.assertEqual(int(stored[903]), 4999)
        # Slot 904: {904} ∩ [904, 4999] = {904}.
        self.assertEqual(int(stored[904]), 904)


class FunctionsTest(TestCase):
    """Tests `enable_sliding_window_attention` config rewriting."""

    def test_enable_sliding_window_attention(self):
        partition_spec = (("data",), None, "model", None)
        cfg = MultiheadAttention.default_config().set(
            name="attn", query_dim=8, key_dim=8, value_dim=8, num_heads=2
        )
        cfg.kv_cache = KVCache.default_config().set(
            cache_dtype=jnp.bfloat16, kv_partition_spec=partition_spec
        )
        sliding_window_size = 4
        out = enable_sliding_window_attention(cfg, sliding_window_size)

        self.assertIs(out, cfg)  # in-place modification.
        self.assertEqual(out.kv_cache.klass, SlidingWindowKVCache)
        self.assertEqual(out.kv_cache.cached_kv_length, sliding_window_size)
        self.assertEqual(out.kv_cache.cache_dtype, jnp.bfloat16)
        self.assertEqual(out.kv_cache.kv_partition_spec, partition_spec)
        self.assertEqual(out.mask.klass, SlidingWindowAttentionBias)
        self.assertEqual(out.mask.sliding_window_size, sliding_window_size)


if __name__ == "__main__":
    absltest.main()
