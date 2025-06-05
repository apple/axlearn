# Copyright © 2023 Apple Inc.

"""Tests ViTDet transformer layers."""
import math

import jax
import numpy as np
from absl.testing import parameterized

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import VisionTransformer
from axlearn.vision.vitdet_transformer import build_vitdet_model_config


# pylint: disable=no-self-use
class ViTDetTransformerTest(TestCase):
    """Tests ViTDetTransformer."""

    @parameterized.parameters(True, False)
    def test_model_forward(self, is_training):
        cfg = build_vitdet_model_config(
            num_layers=3,
            model_dim=8,
            num_heads=4,
            image_size=(32, 32),
            patch_size=(16, 16),
            pretrain_image_size=224,
            window_size=14,
            window_block_indexes=(list(range(0, 2))),
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 32, 32, 3]).astype(np.float32)

        F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )

    def test_stride_error(self):
        with self.assertRaises(ValueError):
            build_vitdet_model_config(
                num_layers=1, model_dim=8, num_heads=4, patch_size=(16, 16), stride=(3, 3)
            )

    @parameterized.parameters(32, 64)
    def test_model_input_image_size(self, image_size):
        cfg = build_vitdet_model_config(
            num_layers=3,
            model_dim=8,
            num_heads=4,
            image_size=(image_size, image_size),
            patch_size=(16, 16),
            pretrain_image_size=224,
            window_size=14,
            window_block_indexes=(list(range(0, 2))),
            peak_stochastic_depth_rate=0.1,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(
            np.float32
        )
        test_output, _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )
        ref_shape_2d = (batch_size, image_size // 16, image_size // 16, 8)
        self.assertEqual(test_output["4"].shape, ref_shape_2d)

    @parameterized.parameters(16, 32)
    def test_model_patch_size(self, patch_size):
        cfg = build_vitdet_model_config(
            num_layers=3,
            model_dim=8,
            num_heads=4,
            image_size=(32, 32),
            patch_size=(patch_size, patch_size),
            pretrain_image_size=224,
            window_size=14,
            window_block_indexes=(list(range(0, 2))),
            peak_stochastic_depth_rate=0.1,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 32, 32, 3]).astype(np.float32)
        test_output, _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )
        ref_shape_2d = (batch_size, 32 // patch_size, 32 // patch_size, 8)
        level = int(math.log2(patch_size))
        self.assertEqual(test_output[str(level)].shape, ref_shape_2d)

    @parameterized.parameters(6, 12)
    def test_model_num_layers(self, num_layers):
        cfg = build_vitdet_model_config(
            num_layers=num_layers,
            model_dim=8,
            num_heads=4,
            image_size=(32, 32),
            patch_size=(16, 16),
            pretrain_image_size=224,
            window_size=14,
            window_block_indexes=(
                list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11))
            ),
            peak_stochastic_depth_rate=0.1,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 32, 32, 3]).astype(np.float32)
        test_output, _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )
        ref_shape_2d = (batch_size, 32 // 16, 32 // 16, 8)
        self.assertEqual(test_output["4"].shape, ref_shape_2d)

    def test_model_window_block_indexes(self):
        cfg = build_vitdet_model_config(
            num_layers=24,
            model_dim=8,
            num_heads=4,
            image_size=(32, 32),
            patch_size=(16, 16),
            pretrain_image_size=224,
            window_size=14,
            window_block_indexes=(
                list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
            ),
            peak_stochastic_depth_rate=0.1,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 32, 32, 3]).astype(np.float32)

        F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )
