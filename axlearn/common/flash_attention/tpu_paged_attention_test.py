# Copyright © 2025 Apple Inc.

"""Tests TPUPagedAttention kernel.

Adapted from JAX paged attention kernel test:
https://github.com/jax-ml/jax/blob/jax-v0.8.1/tests/pallas/tpu_paged_attention_kernel_test.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import generate_paged_attention_data
from axlearn.common.flash_attention.tpu_paged_attention import TPUPagedAttention
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor


class PagedAttentionKernelTest(TestCase):
    @parameterized.product(
        dtype=(jnp.float32, jnp.bfloat16),
        page_size=(16, 32, 64),
        num_kv_heads=(1, 8),
        q_kv_head_ratio=(1, 4, 8),
        head_dim=(128, 256),
        sliding_window_size=(None, 128),
        megacore_mode=(None, "batch", "kv_head"),
    )
    @pytest.mark.skip(reason="Fails in CI due to OOM.")
    def test_paged_attention(
        self,
        dtype,
        page_size,
        num_kv_heads,
        q_kv_head_ratio,
        head_dim,
        sliding_window_size,
        megacore_mode,
    ):
        max_kv_len = 2048
        block_size = 512
        seq_lens = jnp.array([1, 3, 256, 513, 1023, 2048])
        batch_size = len(seq_lens)

        q, k_pages, v_pages, page_indices, bias = generate_paged_attention_data(
            batch_size=batch_size,
            query_len=1,  # Single-step decoding
            kv_len=max_kv_len,
            num_heads=num_kv_heads * q_kv_head_ratio,
            per_head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            mask_fn=causal_mask if not sliding_window_size else None,
            sliding_window_sz=sliding_window_size,
            dtype=dtype,
            query_offset=seq_lens - 1,
        )

        paged_attn = (
            TPUPagedAttention.default_config()
            .set(
                interpret=jax.default_backend() == "cpu",
                tpu_block_size=block_size,
                megacore_mode=megacore_mode,
            )
            .instantiate()
        )
        ref_paged_attn = ReferenceMHA.default_config().set(tpu_block_size=block_size).instantiate()

        input_batch = {
            "query": q,
            "key": k_pages,
            "value": v_pages,
            "page_tables": page_indices,
            "bias": bias,
        }
        output = paged_attn(input_batch)

        def to_float32(x):
            if isinstance(x, Tensor) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.float32)
            return x

        input_batch_ref = jax.tree.map(to_float32, input_batch)
        output_ref = ref_paged_attn(input_batch_ref)

        atol, rtol = 1e-2, 2e-2
        np.testing.assert_allclose(
            output[np.where(seq_lens > 0)].astype(jnp.float32),
            output_ref[np.where(seq_lens > 0)].astype(jnp.float32),
            atol=atol,
            rtol=rtol,
        )
