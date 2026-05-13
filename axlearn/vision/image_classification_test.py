# Copyright © 2023 Apple Inc.

"""Tests image classification models."""
import jax.random
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.metrics import WeightedSummary
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, set_threefry_partitionable
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.image_classification import ImageClassificationModel


class ViTClassificationModelTest(TestCase):
    @parameterized.parameters(0, 1, 2)
    @set_threefry_partitionable(False)  # TODO(markblee): update for threefry_partitionable True
    def test_padded_examples(self, num_padded_examples):
        vit_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            patch_size=(16, 16),
        )
        cfg = ImageClassificationModel.default_config().set(
            name="test", backbone=vit_cfg, num_classes=1000
        )
        model = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size = 2
        num_valid_examples = batch_size - num_padded_examples
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.asarray(
                [99] * num_valid_examples + [-1] * num_padded_examples, dtype=np.int32
            ),
        }

        (loss, aux), output_collection = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=as_tensor(inputs)),
        )
        if num_valid_examples:
            self.assertGreater(loss, 0)
        else:
            self.assertEqual(loss, 0)
        self.assertIn("logits", aux)
        self.assertAlmostEqual(
            output_collection.summaries["metric"]["loss"], WeightedSummary(loss, num_valid_examples)
        )
        self.assertAlmostEqual(
            output_collection.summaries["metric"]["accuracy"],
            WeightedSummary(0.0, num_valid_examples),
        )


if __name__ == "__main__":
    absltest.main()
