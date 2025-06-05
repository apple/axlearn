# Copyright © 2023 Apple Inc.

"""Tests feature tokenizers."""
import jax
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.vision import feature_tokenizer
from axlearn.vision.clip import set_vision_encoder_config


class ModelTest(TestCase):
    """Tests CLIPFeatureTokenizer."""

    def test_clip_tokenizer_forward(self):
        batch_size = 2
        image_size = 48
        patch_size = 16
        output_dim = 4
        projection_dim = 100

        seq_len = (image_size // patch_size) ** 2
        model_cfg = set_vision_encoder_config(
            num_layers=1,
            model_dim=output_dim,
            num_heads=4,
            projection_dim=projection_dim,
            encoder_cls=feature_tokenizer.CLIPFeatureTokenizer,
        )
        model = model_cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Generate inputs.
        image = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(np.float32)
        (output, _), _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=as_tensor(image)),
        )
        self.assertEqual((batch_size, seq_len, projection_dim), output.shape)


if __name__ == "__main__":
    absltest.main()
