"""Tests for splash attention mask.

This code is adapted from the jax_ml/jax library, specifically from the
https://github.com/jax-ml/jax/blob/9bcfac6542a330b77f29d5cc5dcf4a57f55b2947/tests/pallas/tpu_splash_attention_mask_test.py
TODO(dhwang2): Delete this file once JAX is upgraded and LocalMask becomes a computable mask.
"""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from axlearn.common.attention_bias import (
    MaskFnAttentionBias,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention.splash_attention_mask import ComputableMask, classify_blocks
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

    @parameterized.product(
        mask_fn_factory=[
            lambda: causal_mask,
            lambda: sliding_window_causal_mask(sliding_window_size=256),
        ],
        block_shape=[(128, 128), (256, 256)],
        shape=[(1024, 1024), (512, 2048)],
        q_offsets=[[0, 0], [0, 512]],
        downcast_smem_data=[True, False],
    )
    def test_classify_blocks(
        self, mask_fn_factory, block_shape, shape, q_offsets, downcast_smem_data
    ):
        """Test classify_blocks block_mask against process_mask as reference."""
        mask_fn = mask_fn_factory()
        num_heads = 2
        block_kv = block_shape[1]
        kv_blocks = shape[1] // block_kv
        q_offsets = jnp.array(q_offsets)

        ref_block_masks = []
        for offset in q_offsets:

            def offset_mask_fn(q, k, offset=int(offset)):
                return mask_fn(q + offset, k)

            ref_mask = ComputableMask(shape=shape, mask_fn=offset_mask_fn)
            ref_mhm = mask_lib.MultiHeadMask(masks=tuple(ref_mask for _ in range(num_heads)))
            ref_info, _ = mask_info_lib.process_mask(
                ref_mhm, block_shape, head_shards=1, shrink_grid=False
            )
            ref_block_masks.append(np.asarray(ref_info.block_mask).astype(np.int32))
        ref_block_mask = np.stack(ref_block_masks, axis=0)

        # Under test: classify_blocks.
        q_positions = jnp.arange(shape[0], dtype=jnp.int32)[None, :] + q_offsets[:, None]
        mask = MaskFnAttentionBias(
            mask_fn,
            target_positions=q_positions,
            source_positions=jnp.arange(shape[1], dtype=jnp.int32)[None, :],
        )
        block_mask, data_next = classify_blocks(
            mask=mask,
            q_positions=q_positions,
            block_shape=block_shape,
            kv_seq_len=shape[1],
            head_shards=1,
            downcast_smem_data=downcast_smem_data,
        )

        np.testing.assert_array_equal(
            np.asarray(block_mask).astype(np.int32),
            ref_block_mask,
        )

        # Verify data_next.
        expected_data_next = np.broadcast_to(
            np.arange(kv_blocks, dtype=np.int32)[None, None, None, :],
            block_mask.shape,
        )
        expected_data_next = np.where(np.asarray(block_mask) == 0, 0, expected_data_next)
        np.testing.assert_array_equal(
            np.asarray(data_next).astype(np.int32),
            expected_data_next,
        )

        if downcast_smem_data:
            self.assertEqual(block_mask.dtype, jnp.int8)
            expected_dn_dtype = jnp.int8 if kv_blocks <= 127 else jnp.int16
            self.assertEqual(data_next.dtype, expected_dn_dtype)
        else:
            self.assertEqual(block_mask.dtype, jnp.int32)
            self.assertEqual(data_next.dtype, jnp.int32)

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
