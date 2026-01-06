# Copyright © 2025 Apple Inc.

"""Tests for common utilities"""

import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.attention_bias import sliding_window_causal_mask
from axlearn.common.flash_attention.common import (
    build_mask,
    build_sliding_window_mask,
    maybe_pad_inputs,
)
from axlearn.common.test_utils import TestCase


class BuildMaskTest(TestCase):
    @parameterized.product(
        sliding_window_sz=[127, 128, 129],
        seq_len=[128, 256, 512],
        block_size=[64, 128],
    )
    def test_sliding_window_fast_path(self, sliding_window_sz, seq_len, block_size):
        args = dict(q_seq_len=seq_len, kv_seq_len=seq_len, block_k=block_size, block_q=block_size)
        mask = build_mask(sliding_window_causal_mask(sliding_window_sz), **args)
        sliding_mask = build_sliding_window_mask(**args, sliding_window_size=sliding_window_sz)
        self.assertNestedEqual(sliding_mask, mask)


class UtilsTest(TestCase):
    @parameterized.parameters(
        dict(block_size=8, input_len=7, output_len=8),
        dict(block_size=8, input_len=16, output_len=16),
        dict(block_size=16, input_len=15, output_len=16),
        dict(block_size=64, input_len=63, output_len=64),
    )
    def test_maybe_pad_inputs(self, block_size, input_len, output_len):
        query = jnp.ones((2, input_len, 4, 64))
        key = jnp.ones((2, input_len, 2, 64))
        value = jnp.ones((2, input_len, 2, 64))
        segment_id = jnp.ones((2, input_len), dtype=jnp.int32) * 5

        orig_q_len, orig_k_len = query.shape[1], key.shape[1]
        padded_query, padded_key, padded_value, padded_segment_id = maybe_pad_inputs(
            block_size, query, key, value, segment_id
        )

        self.assertEqual(padded_query.shape[1], output_len)
        self.assertEqual(padded_key.shape[1], output_len)
        self.assertEqual(padded_value.shape[1], output_len)
        self.assertEqual(padded_segment_id.shape[1], output_len)
        self.assertEqual(orig_q_len, input_len)
        self.assertEqual(orig_k_len, input_len)
        self.assertEqual(padded_query.shape[1] % block_size, 0)

        self.assertNestedAllClose(padded_query[:, :input_len], query)
        self.assertNestedAllClose(padded_segment_id[:, :input_len], segment_id)

        if output_len > input_len:
            self.assertNestedAllClose(
                padded_query[:, input_len:], jnp.zeros_like(padded_query[:, input_len:])
            )
            self.assertNestedAllClose(
                padded_segment_id[:, input_len:], jnp.zeros_like(padded_segment_id[:, input_len:])
            )

        padded_query2, _, _, padded_segment_id2 = maybe_pad_inputs(
            block_size, query, key, value, None
        )
        self.assertEqual(padded_query2.shape[1], output_len)
        self.assertIsNone(padded_segment_id2)


if __name__ == "__main__":
    absltest.main()
