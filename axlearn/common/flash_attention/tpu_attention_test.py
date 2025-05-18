# Copyright © 2023 Apple Inc.

"""Tests TPU FlashAttention kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    MaskFnAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    and_masks,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention import tpu_attention
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor


def setUpModule():
    if jax.default_backend() not in ("tpu", "cpu"):
        pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def jax_fn_mask(sliding_window_size: int) -> Tensor:
    """A MaskFn that calls jax.

    The mask is the same as `sliding_window_causal_mask`.

    However, this implementation requires specially handling to use with
    SplashAttention since `tpu_flash_attention()` needs to wrap this function
    to return numpy values if the input is numpy. (Otherwise we get tracer errors in jit.)
    """

    def mask(query_position: Tensor, key_position: Tensor):
        return query_position - key_position <= sliding_window_size

    fun = and_masks(causal_mask, mask)
    return fun


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch_size=2,
            kv_len=256,
            num_heads=4,
        ),
        dict(
            batch_size=8,
            kv_len=2048,
            num_heads=4,
        ),
    ]

    @parameterized.product(seq_len=[8, 16, 32, 128], sliding_window_size=[4, 8, 16])
    def test_sliding_window_mask_equivalence(self, seq_len, sliding_window_size):
        shape = (seq_len, seq_len)
        ref_mask = splash_attention_mask.LocalMask(
            shape=shape, window_size=(sliding_window_size, 0), offset=0
        )

        mask_fn = sliding_window_causal_mask(sliding_window_size=sliding_window_size)
        mask_array = np.asarray(mask_fn(np.arange(seq_len)[:, None], np.arange(seq_len)[None, :]))

        test_mask = splash_attention_mask.NumpyMask(array=mask_array)

        for i in range(seq_len):
            self.assertNestedAllClose(ref_mask[i:, i:], test_mask[i:, i:])

    @parameterized.parameters(
        [ZeroAttentionBias(), splash_attention_mask.FullMask((8, 8))],
        [
            CausalAttentionBias(
                target_positions=jnp.arange(8)[None], source_positions=jnp.arange(8)[None]
            ),
            splash_attention_mask.CausalMask(shape=(8, 8)),
        ],
        [
            SlidingWindowAttentionBias(
                sliding_window_causal_mask(4),
                sliding_window_size=4,
                target_positions=jnp.arange(8)[None],
                source_positions=jnp.arange(8)[None],
            ),
            splash_attention_mask.LocalMask(shape=(8, 8), window_size=(4, 0), offset=0),
        ],
        [
            MaskFnAttentionBias(
                jax_fn_mask(5),
                target_positions=jnp.arange(8)[None],
                source_positions=jnp.arange(8)[None],
            ),
            splash_attention_mask.NumpyMask(
                array=np.array(jax_fn_mask(5)(jnp.arange(8)[:, None], jnp.arange(8)[None, :]))
            ),
        ],
    )
    def test_to_splash_mask(self, mask, expected):
        # _to_splash_mask must work well during jax tracing as it runs in shard_map.
        @jax.jit
        def inside_tracing(mask):
            # pylint: disable-next=protected-access
            splash_mask = tpu_attention._to_splash_mask(mask, mask_shape=(8, 8))
            self.assertEqual(splash_mask, expected)

        inside_tracing(mask)

    @parameterized.product(
        _TEST_CONFIGS,
        query_length_multiplier=[0.5, 1, 2],
        mask=[None, causal_mask, jax_fn_mask(5)],
        attention_bias_type=[None, "2d", "4d"],
        with_segment_ids=[False, True],
        per_head_dim=[32, 64, 128, 256],
    )
    def test_forward_and_backward(
        self,
        batch_size,
        kv_len,
        num_heads,
        per_head_dim,
        query_length_multiplier,
        mask,
        attention_bias_type,
        with_segment_ids,
    ):
        if jax.default_backend() == "cpu":
            # TODO(dhwang2): this has been broken for a while on CPU.
            pytest.skip(reason="Backward path is broken on CPU")
        if mask not in (None, causal_mask) and query_length_multiplier > 1:
            pytest.skip(reason="Sliding window attention does not make sense when q_len != kv_len.")
        # pylint: disable=protected-access
        fallback_to_legacy = per_head_dim % 128 != 0 or (attention_bias_type is not None)
        q, k, v, bias = generate_attention_data(
            batch_size,
            int(kv_len * query_length_multiplier),
            kv_len,
            num_heads,
            per_head_dim,
            mask_fn=mask,
            attention_bias_type=attention_bias_type,
            with_segment_ids=with_segment_ids,
        )
        cfg = dict(
            interpret=jax.default_backend() == "cpu",
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=128,
        )
        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
        )
        if not fn.is_supported(input_batch=input_batch):
            # Check splash attention is used when it should be.
            self.assertEqual(fallback_to_legacy, True)
            fn = tpu_attention.LegacyTPUFlashAttention.default_config().set(**cfg).instantiate()
            self.assertEqual(fn.is_supported(input_batch=input_batch), True)

        # Compare outputs.
        out = fn(input_batch)
        ref_out = ref_fn(input_batch)
        self.assertNestedAllClose(out, ref_out, atol=0.05)

        # Compare grads.
        def grad_fn(float_inputs, aux_inputs, f):
            full_batch = {**float_inputs, **aux_inputs}
            return f(full_batch).mean()

        float_inputs = dict(query=q, key=k, value=v)
        aux_inputs = dict(bias=bias)
        grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, fn)
        ref_grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, ref_fn)
        self.assertNestedAllClose(grad_out, ref_grad_out, atol=0.05)


if __name__ == "__main__":
    absltest.main()
