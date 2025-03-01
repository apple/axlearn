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
from axlearn.common.flash_attention import utils
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
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

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

        with patch("axlearn.common.flash_attention.utils._interpret", return_value=True):
            with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
                xla_fn = utils.flash_attention_implementation("xla")
                test_fn = utils.flash_attention_implementation(backend)

                query, key, value = _get_inputs(
                    batch=batch,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads or num_heads,
                    per_head_dim=per_head_dim,
                    input_dtype=input_dtype,
                )
                prng_key = jax.random.PRNGKey(0)

                ref_out = xla_fn(query, key, value, bias, prng_key)
                test_out = test_fn(query, key, value, bias, prng_key)
                self.assertNestedAllClose(ref_out, test_out, atol=0.01)
        jax.clear_caches()

    @parameterized.product(
        _TEST_CONFIGS,
        backend=["cpu", "gpu", "tpu"],
        bias_type=["causal", "sliding"],
        input_dtype=[jnp.float32],
        # TODO(hanzhi_zhou): support multi step gpu decoding.
        step_len=[1],
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
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

        if bias_type == "causal":
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

        with patch("axlearn.common.flash_attention.utils._interpret", return_value=True):
            with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
                fwd_fn = utils.flash_attention_implementation(backend)
                decode_fn = utils.flash_attention_implementation(backend, is_decoding=True)

                query, key, value = _get_inputs(
                    batch=batch,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads or num_heads,
                    per_head_dim=per_head_dim,
                    input_dtype=input_dtype,
                )
                prng_key = jax.random.PRNGKey(0)

                fwd_out = fwd_fn(query, key, value, bias, prng_key)
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
                    decoding_out = decode_fn(query_step, key, value, bias_step, prng_key)
                    decoding_output.append(decoding_out)
                decoding_output = jnp.concatenate(decoding_output, axis=1)
                self.assertNestedAllClose(fwd_out, decoding_output, atol=0.01)
        jax.clear_caches()


if __name__ == "__main__":
    absltest.main()
