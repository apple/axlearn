# Copyright © 2025 Apple Inc.
"""Tests GPU and TPU decoding."""
from contextlib import nullcontext
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.attention_bias import sliding_window_causal_mask
from axlearn.common.flash_attention.gpu_decoding import NEG_INF
from axlearn.common.flash_attention.gpu_decoding import flash_decoding as gpu_decoding
from axlearn.common.flash_attention.tpu_decoding import tpu_decoding
from axlearn.common.flash_attention.utils import mha_reference
from axlearn.common.test_utils import TestCase, Tolerance

if jax.default_backend() == "gpu":
    decoding_fns = [gpu_decoding]
    dtypes = [jnp.float32, jnp.float16]
elif jax.default_backend() == "tpu":
    decoding_fns = [tpu_decoding]
    dtypes = [jnp.bfloat16]
elif jax.default_backend() == "cpu":
    # CPU emulation of pallas kernels.
    decoding_fns = [gpu_decoding, tpu_decoding]
    dtypes = [jnp.float32]
else:
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


class DecodingTest(TestCase):
    """Tests GPU and TPU decoding."""

    @parameterized.product(
        [
            dict(zip(["batch_size", "seq_len", "num_heads", "per_head_dim"], args))
            for args in [
                (1, 1024, 32, 64),
                (4, 512, 48, 64),
                (2, 1024, 16, 128),
                (1, 4096, 8, 128),
                (2, 734, 48, 64),
            ]
        ],
        attention_bias_type=[None],
        input_dtype=dtypes,
        padding=[0, 111],
        kv_head_factor=[1, 4, 8],
        window_len=[-1, 16, 127],
        decoding_fn=decoding_fns,
    )
    def test_decode_against_ref(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        per_head_dim: int,
        attention_bias_type: Literal["2d", "4d", None],
        input_dtype: jnp.dtype,
        padding: int,
        kv_head_factor: int,
        window_len: int,
        decoding_fn,
    ):
        if seq_len % 512 != 0 and decoding_fn is tpu_decoding:
            self.skipTest("TPU decoding doesn't support seq_len % block_size != 0")
        self.assertEqual(num_heads % kv_head_factor, 0)
        assert num_heads % kv_head_factor == 0
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        q = jax.random.normal(k1, (batch_size, 1, num_heads, per_head_dim), dtype=input_dtype)
        k = jax.random.normal(
            k2,
            (batch_size, seq_len, num_heads // kv_head_factor, per_head_dim),
            dtype=input_dtype,
        )
        v = jax.random.normal(
            k3,
            (batch_size, seq_len, num_heads // kv_head_factor, per_head_dim),
            dtype=input_dtype,
        )

        if attention_bias_type == "4d":
            bias = jax.random.normal(k4, (batch_size, num_heads, 1, seq_len), dtype=input_dtype)
        elif attention_bias_type == "2d":
            bias = jax.random.normal(k4, (1, 1, 1, seq_len), dtype=input_dtype)
        else:
            bias = None

        softmax_scale = per_head_dim**0.5
        mask_fn = None
        if window_len > 0:
            mask_fn = sliding_window_causal_mask(window_len)
        o = decoding_fn(
            q,
            k,
            v,
            bias=bias,
            softmax_scale=softmax_scale,
            kv_seq_len=seq_len - padding,
            mask_fn=mask_fn,
            interpret=(jax.default_backend() == "cpu"),
        )
        if bias is not None:
            bias = bias[:, :, :, : seq_len - padding]
        if window_len > 0:
            if bias is None:
                bias = jnp.zeros((1, 1, 1, seq_len - padding), dtype=input_dtype)
            bias = bias.at[:, :, :, : -window_len - 1].set(NEG_INF)
        with jax.default_matmul_precision(
            "highest"
        ) if input_dtype is jnp.float32 else nullcontext():
            o_ref = mha_reference(
                q,
                k[:, : seq_len - padding],
                v[:, : seq_len - padding],
                bias,
                None,
                causal=False,
                softmax_scale=softmax_scale,
            )
        if input_dtype is jnp.float32:
            self.assertNestedAllClose(o, o_ref, rtol=0.001, atol=0.0005)
        # bfloat16 and float16 have occasional outliers that require relaxed tolerances.
        elif input_dtype is jnp.bfloat16:
            self.assertAllCloseWithOutliers(
                o,
                o_ref,
                tolerance_map={
                    1.0: Tolerance(rtol=0.05, atol=1.25),
                    0.99: Tolerance(rtol=0.05, atol=0.4),
                    0.95: Tolerance(rtol=0.05, atol=0.2),
                    0.9: Tolerance(rtol=0.05, atol=0.1),
                    0.8: Tolerance(rtol=0.05, atol=0.05),
                },
            )
        elif input_dtype is jnp.float16:
            self.assertAllCloseWithOutliers(
                o,
                o_ref,
                tolerance_map={
                    1.0: Tolerance(rtol=0.01, atol=0.2),
                    0.98: Tolerance(rtol=0.01, atol=0.05),
                    0.9: Tolerance(rtol=0.01, atol=0.025),
                },
            )
        else:
            raise ValueError(f"Unsupported dtype {input_dtype}")
