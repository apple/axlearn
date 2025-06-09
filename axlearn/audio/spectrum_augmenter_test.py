# Copyright © 2023 Apple Inc.

"""Tests spectrum augmentation utilities."""
# pylint: disable=protected-access

import contextlib
from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

from axlearn.audio.spectrum_augmenter import MaskSampler, SpectrumAugmenter
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, dummy_padding_mask
from axlearn.common.utils import Tensor, safe_not


class MaskSamplerTest(TestCase):
    """Tests MaskSampler."""

    @parameterized.product(
        [
            dict(batch_size=3, max_length=20, max_num_masks=1, max_mask_length=5),
            dict(batch_size=3, max_length=20, max_num_masks=5, max_mask_length=1),
            dict(batch_size=8, max_length=30, max_num_masks=8, max_mask_length=15),
            # Test where the configured values are larger than max_length.
            dict(batch_size=4, max_length=20, max_num_masks=50, max_mask_length=50),
        ],
        max_num_masks_ratio=[1.0, 0.6, 0.1],
        max_mask_length_ratio=[1.0, 0.6, 0.1],
    )
    def test_mask_sampler(
        self,
        batch_size: int,
        max_length: int,
        max_num_masks: int,
        max_mask_length: int,
        max_num_masks_ratio: float = 1.0,
        max_mask_length_ratio: float = 1.0,
    ):
        cfg = MaskSampler.default_config().set(
            max_num_masks=max_num_masks,
            max_num_masks_ratio=max_num_masks_ratio,
            max_mask_length=max_mask_length,
            max_mask_length_ratio=max_mask_length_ratio,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        input_lengths = jax.random.randint(
            jax.random.PRNGKey(321), shape=[batch_size], minval=1, maxval=max_length
        )
        effective_max_mask_length = jnp.minimum(
            max_mask_length, max_mask_length_ratio * input_lengths
        )
        effective_max_num_masks = jnp.minimum(max_num_masks, max_num_masks_ratio * input_lengths)
        # Enforce between [1, max_length].
        effective_max_mask_length = jnp.minimum(
            max_length, jnp.maximum(effective_max_mask_length, 1)
        )
        effective_max_num_masks = jnp.minimum(max_length, jnp.maximum(effective_max_num_masks, 1))

        @jax.jit
        def jit_forward(input_lengths):
            out, _ = F(
                layer,
                inputs=dict(input_lengths=input_lengths, max_length=max_length),
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
                state={},
            )
            return out

        # [batch_size, max_length].
        masks = jit_forward(input_lengths)
        # Test the shape of the mask.
        self.assertEqual(masks.shape, (batch_size, max_length))
        # Test that total masked positions does not exceed effective max.
        self.assertTrue(
            jnp.all(jnp.sum(masks, axis=-1) <= effective_max_mask_length * effective_max_num_masks)
        )
        # Test that masks are within valid input range.
        self.assertTrue(jnp.all(jnp.argmax(masks, axis=-1) < input_lengths))

    @parameterized.parameters(
        # Valid cases.
        dict(max_num_masks=1, max_mask_length=1),
        dict(max_num_masks_ratio=1, max_mask_length_ratio=1),
        # Invalid cases.
        dict(max_num_masks=None, max_mask_length=None, expected=ValueError("at least one of")),
        dict(max_num_masks=3, max_mask_length=0, expected=ValueError("greater than 0")),
        dict(max_num_masks=0, max_mask_length=3, expected=ValueError("greater than 0")),
        dict(max_num_masks_ratio=0, max_mask_length=3, expected=ValueError(r"in \(0, 1]")),
        dict(max_num_masks_ratio=2, max_mask_length=3, expected=ValueError(r"in \(0, 1]")),
        dict(max_num_masks=3, max_mask_length_ratio=0, expected=ValueError(r"in \(0, 1]")),
        dict(max_num_masks=3, max_mask_length_ratio=2, expected=ValueError(r"in \(0, 1]")),
    )
    def test_input_validation(
        self,
        max_num_masks: Optional[int] = None,
        max_mask_length: Optional[int] = None,
        max_num_masks_ratio: Optional[float] = None,
        max_mask_length_ratio: Optional[float] = None,
        expected: Optional[Exception] = None,
    ):
        cfg = MaskSampler.default_config().set(
            max_num_masks=max_num_masks,
            max_num_masks_ratio=max_num_masks_ratio,
            max_mask_length=max_mask_length,
            max_mask_length_ratio=max_mask_length_ratio,
        )
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            cfg.set(name="test").instantiate(parent=None)


class SpectrumAugmenterTest(TestCase):
    """Tests SpectrumAugmenter."""

    def _generate_masks(
        self,
        inputs: dict[str, Tensor],
        *,
        max_freq_masks: int,
        max_freq_length: int,
        max_time_masks: int,
        max_time_length: int,
        max_time_masks_ratio: float,
        max_time_length_ratio: float,
        is_training: bool,
    ):
        cfg = SpectrumAugmenter.default_config().set(
            freq_mask_sampler=MaskSampler.default_config().set(
                max_num_masks=max_freq_masks,
                max_mask_length=max_freq_length,
            ),
            time_mask_sampler=MaskSampler.default_config().set(
                max_num_masks=max_time_masks,
                max_num_masks_ratio=max_time_masks_ratio,
                max_mask_length=max_time_length,
                max_mask_length_ratio=max_time_length_ratio,
            ),
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        @jax.jit
        def jit_forward(inputs) -> Tensor:
            out, _ = F(
                layer,
                inputs=inputs,
                is_training=is_training,
                prng_key=jax.random.PRNGKey(0),
                state={},
            )
            return out

        return jit_forward(inputs)

    @parameterized.product(
        [
            dict(
                max_freq_masks=3,
                max_freq_length=4,
                max_time_masks=20,
                max_time_masks_ratio=0.2,
                max_time_length=10,
                max_time_length_ratio=0.1,
            ),
        ],
        input_shape=[(2, 8, 8, 3), (4, 32, 16, 3)],
        is_training=[True, False],
    )
    def test_spectrum_augmenter(self, input_shape: Sequence[int], is_training: bool, **kwargs):
        if is_training:
            inputs = jnp.ones(input_shape)
        else:
            inputs = jax.random.normal(jax.random.PRNGKey(123), input_shape)
        paddings = safe_not(
            dummy_padding_mask(batch_size=inputs.shape[0], max_seq_len=inputs.shape[1])
        )
        outputs = self._generate_masks(
            dict(inputs=inputs, paddings=paddings), is_training=is_training, **kwargs
        )
        if is_training:
            self.assertEqual(inputs.shape, outputs.shape)
            # Tests that the mask is the same for all channels.
            self.assertTrue(jnp.all(jnp.diff(outputs, axis=-1) == 0))
            # Different samples should get different masks with high probability.
            self.assertFalse(jnp.any(jnp.all(jnp.diff(outputs, axis=0) == 0, axis=(1, 2, 3))))
        else:
            # Tests that inputs remains the same.
            self.assertTrue(jnp.allclose(inputs, outputs))

    @parameterized.product(
        [
            dict(
                max_freq_masks=3,
                max_freq_length=4,
                max_time_masks=4,
                max_time_masks_ratio=0.2,
                max_time_length=10,
                max_time_length_ratio=0.1,
            ),
        ],
        input_shape=[(4, 32, 16, 1)],
    )
    @pytest.mark.skip(reason="Comment out to run manually.")
    def test_visualize(self, input_shape: Sequence[int], **kwargs):
        inputs = jnp.ones(input_shape)
        paddings = jnp.zeros([input_shape[0], input_shape[1]], jnp.bool)
        outputs = self._generate_masks(
            dict(inputs=inputs, paddings=paddings), is_training=True, **kwargs
        )
        # pylint: disable-next=import-outside-toplevel
        import matplotlib.pyplot as plt  # pytype: disable=import-error

        _, plots = plt.subplots(outputs.shape[0], 1)
        for plot, output in zip(plots, outputs):
            # Show time on x axis and freq on y.
            plot.imshow(jnp.moveaxis(output, 0, 1), cmap=plt.cm.gray)
        plt.show()


if __name__ == "__main__":
    absltest.main()
