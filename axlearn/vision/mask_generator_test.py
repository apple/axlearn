# Copyright © 2023 Apple Inc.

"""Tests mask generator."""
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.vision import mask_generator


class MaskGeneratorTest(parameterized.TestCase):
    """Tests MaskingGenerator."""

    @parameterized.product(
        num_masking_patches=(0, 118, 196),
        max_aspect=(None, 0.8, 10.0),
        max_mask_patches=(None, 100),
    )
    def test_mask_generation(self, num_masking_patches, max_aspect, max_mask_patches):
        input_size = 14
        model = mask_generator.MaskingGenerator(
            input_size=(input_size, input_size),
            num_masking_patches=num_masking_patches,
            max_aspect=max_aspect,
            min_mask_patches=16,
            max_mask_patches=max_mask_patches,
        )
        mask = model()
        self.assertEqual(mask.sum(), num_masking_patches)
        if num_masking_patches == 0:
            np.testing.assert_array_equal(mask, np.zeros(shape=mask.shape, dtype=np.int32))
        if num_masking_patches == input_size * input_size:
            np.testing.assert_array_equal(mask, np.ones(shape=mask.shape, dtype=np.int32))


if __name__ == "__main__":
    absltest.main()
