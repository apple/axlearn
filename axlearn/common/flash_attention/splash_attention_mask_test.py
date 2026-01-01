"""Tests for splash attention mask.

This code is adapted from the jax_ml/jax library, specifically from the
https://github.com/jax-ml/jax/blob/9bcfac6542a330b77f29d5cc5dcf4a57f55b2947/tests/pallas/tpu_splash_attention_mask_test.py
TODO(dhwang2): Delete this file once JAX is upgraded and LocalMask becomes a computable mask.
"""

import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

from axlearn.common.attention_bias import causal_mask, sliding_window_causal_mask
from axlearn.common.flash_attention.splash_attention_mask import ComputableMask
from axlearn.common.test_utils import TestCase


class SplashAttentionMaskTest(TestCase):
    @parameterized.parameters(
        [
            ((256, 256), (1024, 1024)),
            ((256, 128), (1024, 1024)),
            ((128, 256), (1024, 1024)),
            ((256, 256), (1024, 2048)),
            ((256, 128), (1024, 2048)),
            ((128, 256), (1024, 2048)),
            ((256, 256), (2048, 1024)),
            ((256, 128), (2048, 1024)),
            ((128, 256), (2048, 1024)),
        ]
    )
    def test_causal_mask_fn(self, block_size, shape):
        """Test ComputableMask with causal_mask function from attention_bias.py."""
        # Create expected dense causal mask
        q_len, kv_len = shape
        q_ids = np.arange(q_len)[:, None]
        kv_ids = np.arange(kv_len)[None, :]
        dense_mask = causal_mask(q_ids, kv_ids)

        # Create ComputableMask with mask_fn
        lazy_mask = ComputableMask(shape=shape, mask_fn=causal_mask)

        self._compare_masks(dense_mask, lazy_mask, block_size)

    @parameterized.parameters(
        [
            ((256, 256), (1024, 1024), 128),
            ((256, 128), (1024, 1024), 128),
            ((128, 256), (1024, 1024), 128),
            ((256, 256), (1024, 2048), 256),
            ((256, 128), (1024, 2048), 256),
            ((128, 256), (1024, 2048), 256),
            ((256, 256), (2048, 1024), 512),
            ((256, 128), (2048, 1024), 512),
            ((128, 256), (2048, 1024), 512),
        ]
    )
    def test_sliding_window_causal_mask_fn(self, block_size, shape, window_size):
        """Test ComputableMask with sliding_window_causal_mask from attention_bias.py."""
        # Create expected dense sliding window causal mask
        q_len, kv_len = shape
        q_ids = np.arange(q_len)[:, None]
        kv_ids = np.arange(kv_len)[None, :]
        mask_fn = sliding_window_causal_mask(sliding_window_size=window_size)
        dense_mask = mask_fn(q_ids, kv_ids)

        # Create ComputableMask with mask_fn
        lazy_mask = ComputableMask(shape=shape, mask_fn=mask_fn)

        self._compare_masks(dense_mask, lazy_mask, block_size)

    def _compare_masks(
        self,
        dense_mask: np.ndarray,
        lazy_mask: mask_lib.Mask,
        block_size: tuple[int, int],
    ):
        self.assertEqual(dense_mask.shape, lazy_mask.shape)

        *prefix, width, height = dense_mask.shape

        assert width % block_size[0] == 0
        assert height % block_size[1] == 0

        full_lazy_mask = lazy_mask[(*[slice(p) for p in prefix], slice(None), slice(None))]
        self.assertNestedEqual(dense_mask, full_lazy_mask)
        for i, j in np.ndindex(width // block_size[0], height // block_size[1]):
            indexer = (
                *[slice(p) for p in prefix],
                slice(i * block_size[0], (i + 1) * block_size[0]),
                slice(j * block_size[1], (j + 1) * block_size[1]),
            )
            dense_chunk = dense_mask[indexer]
            lazy_chunk = lazy_mask[indexer]
            self.assertNestedEqual(dense_chunk, lazy_chunk)


if __name__ == "__main__":
    absltest.main()
