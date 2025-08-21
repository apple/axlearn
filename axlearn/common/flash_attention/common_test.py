# Copyright © 2025 Apple Inc.

"""Tests for common utilities"""

from absl.testing import parameterized

from axlearn.common.attention_bias import sliding_window_causal_mask
from axlearn.common.flash_attention.common import build_mask, build_sliding_window_mask
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
