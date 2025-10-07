# Copyright © 2025 Apple Inc.
"""Tests utils.py.

XLA_FLAGS="--xla_force_host_platform_device_count=8" \
    pytest -m "for_8_devices" axlearn/common/flash_attention/utils_test.py -n auto

This test is expected to run on CPU and is designed to validate GPU/TPU code from a CPU environment.
It allows quick verification in CI and local environments to ensure that code changes do not break
GPU/TPU Flash Attention.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention import common, utils
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache import PagedKVCache
from axlearn.common.kv_cache.sliding_window_kv_cache import SlidingWindowKVCache
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape


def setUpModule():
    # Uncomment for local debugging.
    # import chex
    # chex.set_n_cpu_devices(8)
    if jax.default_backend() in ("gpu", "tpu"):
        pytest.skip(reason="This is a CPU only test.", allow_module_level=True)


def _get_inputs(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    per_head_dim: int,
    input_dtype: jnp.dtype = jnp.bfloat16,
):
    query = jax.random.normal(
        jax.random.PRNGKey(0),
        [batch, seq_len, num_heads, per_head_dim],
        dtype=input_dtype,
    )
    key = jax.random.normal(
        jax.random.PRNGKey(1),
        [batch, seq_len, num_kv_heads, per_head_dim],
        dtype=input_dtype,
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2),
        [batch, seq_len, num_kv_heads, per_head_dim],
        dtype=input_dtype,
    )
    return query, key, value


def _get_paged_inputs(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    per_head_dim: int,
    page_size: int,
    input_dtype: jnp.dtype = jnp.bfloat16,
):
    total_pages = batch * seq_len // page_size
    pages_per_sequence = seq_len // page_size
    query = jax.random.normal(
        jax.random.PRNGKey(0),
        [batch, seq_len, num_heads, per_head_dim],
        dtype=input_dtype,
    )
    key = jax.random.normal(
        jax.random.PRNGKey(1),
        [num_kv_heads, total_pages, page_size, per_head_dim],
        dtype=input_dtype,
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2),
        [num_kv_heads, total_pages, page_size, per_head_dim],
        dtype=input_dtype,
    )
    page_tables = jnp.arange(batch * pages_per_sequence, dtype=jnp.int32)
    page_tables = jax.random.permutation(jax.random.PRNGKey(3), page_tables, independent=True)
    page_tables = page_tables.reshape(batch, pages_per_sequence)

    return query, key, value, page_tables


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=128,
            mesh=(1, 1, 8, 1),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=1,
            per_head_dim=128,
            mesh=(2, 1, 2, 2),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
    ]

    @parameterized.product(
        _TEST_CONFIGS,
        backend=["cpu", "gpu", "tpu"],
        bias_type=["full", "causal", "sliding"],
        input_dtype=[jnp.float32],
    )
    @pytest.mark.for_8_devices
    def test_forward(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        backend,
        bias_type,
        input_dtype,
    ):
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        if bias_type == "full":
            bias = ZeroAttentionBias()
        elif bias_type == "causal":
            bias = CausalAttentionBias(
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )
        else:
            assert bias_type == "sliding"
            bias = SlidingWindowAttentionBias(
                mask=sliding_window_causal_mask(sliding_window_size=4),
                sliding_window_size=4,
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )

        query, key, value = _get_inputs(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads or num_heads,
            per_head_dim=per_head_dim,
            input_dtype=input_dtype,
        )
        with patch("axlearn.common.flash_attention.utils._interpret", return_value=True):
            ref_fn = common.ReferenceMHA.default_config().instantiate()
            test_fn = utils.flash_attention_implementation(
                backend,
                query=query,
                key=key,
                value=value,
                bias=bias,
                tpu_block_size=128,
            )
            with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
                prng_key = jax.random.PRNGKey(0)
                input_batch = dict(
                    query=query,
                    key=key,
                    value=value,
                    prng_key=prng_key,
                    bias=bias,
                    logit_sink=None,
                )
                ref_out = ref_fn(input_batch)
                test_out = test_fn(input_batch)
                self.assertNestedAllClose(ref_out, test_out, atol=0.015)
        jax.clear_caches()

    @parameterized.product(
        _TEST_CONFIGS,
        backend=["cpu", "gpu", "tpu"],
        bias_type=["causal", "sliding"],
        input_dtype=[jnp.float32],
        # TODO(hanzhi_zhou): support multi step gpu decoding.
        step_len=[1],
        page_size=[16, None],
    )
    @pytest.mark.for_8_devices
    def test_decoding(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        backend,
        bias_type,
        input_dtype,
        step_len,
        page_size,
    ):
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        if bias_type == "causal":
            bias = CausalAttentionBias(
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )
            kv_cache_type = KVCache
        else:
            assert bias_type == "sliding"
            bias = SlidingWindowAttentionBias(
                mask=sliding_window_causal_mask(sliding_window_size=4),
                sliding_window_size=4,
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )
            kv_cache_type = SlidingWindowKVCache

        if page_size is not None:
            query, key, value, page_tables = _get_paged_inputs(
                batch=batch,
                seq_len=seq_len,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                per_head_dim=per_head_dim,
                page_size=page_size,
                input_dtype=input_dtype,
            )

            key_for_forward = common.reconstruct_kv(page_tables, key)
            value_for_forward = common.reconstruct_kv(page_tables, value)
            kv_cache_type = PagedKVCache
        else:
            query, key, value = _get_inputs(
                batch=batch,
                seq_len=seq_len,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                per_head_dim=per_head_dim,
                input_dtype=input_dtype,
            )
            page_tables = None
            key_for_forward, value_for_forward = key, value

        with patch("axlearn.common.flash_attention.utils._interpret", return_value=True):
            fwd_fn = utils.flash_attention_implementation(
                backend,
                query=query,
                key=key_for_forward,
                value=value_for_forward,
                bias=bias,
                tpu_block_size=128,
            )
            dummy_query_step = query[:, :1]
            decode_fn = utils.flash_attention_implementation(
                backend,
                query=dummy_query_step,
                key=key,
                value=value,
                bias=bias,
                kv_cache_type=kv_cache_type,
                tpu_block_size=128,
                page_tables=page_tables,
            )
            if decode_fn is None:
                self.assertEqual(kv_cache_type, SlidingWindowKVCache)
                return
            with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
                prng_key = jax.random.PRNGKey(0)
                forward_batch = dict(
                    query=query,
                    key=key_for_forward,
                    value=value_for_forward,
                    prng_key=prng_key,
                    bias=bias,
                    logit_sink=None,
                )
                fwd_out = fwd_fn(forward_batch)
                # Limit generation length to 16 to save test time.
                query_len = 16
                query = query[:, :query_len]
                fwd_out = fwd_out[:, :query_len]

                decoding_output = []
                for t in range(0, query_len, step_len):
                    if bias_type == "causal":
                        bias_step = CausalAttentionBias(
                            target_positions=jnp.arange(step_len)[None] + t,
                            source_positions=jnp.arange(seq_len)[None],
                        )
                    else:
                        assert bias_type == "sliding"
                        bias_step = SlidingWindowAttentionBias(
                            mask=sliding_window_causal_mask(sliding_window_size=4),
                            sliding_window_size=4,
                            target_positions=jnp.arange(step_len)[None] + t,
                            source_positions=jnp.arange(seq_len)[None],
                        )

                    query_step = query[:, t : t + step_len]
                    decode_batch = dict(
                        query=query_step,
                        key=key,
                        value=value,
                        prng_key=prng_key,
                        page_tables=page_tables,
                        bias=bias_step,
                        logit_sink=None,
                    )
                    decoding_out = decode_fn(input_batch=decode_batch)
                    decoding_output.append(decoding_out)
                decoding_output = jnp.concatenate(decoding_output, axis=1)
                self.assertNestedAllClose(fwd_out, decoding_output, atol=0.02)
        jax.clear_caches()


if __name__ == "__main__":
    absltest.main()
