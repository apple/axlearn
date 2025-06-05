# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests mobile networks."""
import re

import jax.random
import numpy as np
import timm
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.module import functional as F
from axlearn.common.utils import count_model_params
from axlearn.vision.image_classification import ImageClassificationModel
from axlearn.vision.mobilenets import (
    EfficientNetEmbedding,
    EndpointsMode,
    MobileNets,
    MobileNetV3Embedding,
    ModelNames,
    named_model_configs,
)


class MobileNetsTest(parameterized.TestCase):
    """Tests MobileNets."""

    # TODO(edaxberger): Unify this test with test_mobilenets in image_classification_test.py.
    @parameterized.parameters(
        (ModelNames.MOBILENETV3, "small-minimal-100", None),
        (ModelNames.MOBILENETV3, "small-075", None),
        (ModelNames.MOBILENETV3, "small-100", None),
        (ModelNames.MOBILENETV3, "large-minimal-100", None),
        (ModelNames.MOBILENETV3, "large-075", None),
        (ModelNames.MOBILENETV3, "large-100", None),
        (ModelNames.EFFICIENTNET, "B0", "V1"),
        (ModelNames.EFFICIENTNET, "B1", "V1"),
        (ModelNames.EFFICIENTNET, "B2", "V1"),
        (ModelNames.EFFICIENTNET, "B3", "V1"),
        (ModelNames.EFFICIENTNET, "B4", "V1"),
        (ModelNames.EFFICIENTNET, "B5", "V1"),
        (ModelNames.EFFICIENTNET, "B6", "V1"),
        (ModelNames.EFFICIENTNET, "B7", "V1"),
        (ModelNames.EFFICIENTNET, "B8", "V1"),
        (ModelNames.EFFICIENTNET, "lite0", "V1"),
        (ModelNames.EFFICIENTNET, "lite1", "V1"),
        (ModelNames.EFFICIENTNET, "lite2", "V1"),
        (ModelNames.EFFICIENTNET, "lite3", "V1"),
        (ModelNames.EFFICIENTNET, "lite4", "V1"),
        (ModelNames.EFFICIENTNET, "B0", "V2"),
        (ModelNames.EFFICIENTNET, "B1", "V2"),
        (ModelNames.EFFICIENTNET, "B2", "V2"),
        (ModelNames.EFFICIENTNET, "B3", "V2"),
        (ModelNames.EFFICIENTNET, "S", "V2"),
        (ModelNames.EFFICIENTNET, "M", "V2"),
        (ModelNames.EFFICIENTNET, "L", "V2"),
    )
    def test_param_count(self, model_name, variant, version):
        cfg = named_model_configs(model_name, variant, efficientnet_version=version)
        classification_model = (
            ImageClassificationModel.default_config()
            .set(
                name="test",
                backbone=cfg,
                num_classes=1000,
            )
            .instantiate(parent=None)
        )
        init_params_classifier = classification_model.initialize_parameters_recursively(
            jax.random.PRNGKey(1)
        )

        timm_model_name = {
            (ModelNames.EFFICIENTNET, "V1"): "efficientnet",
            (ModelNames.EFFICIENTNET, "V2"): "efficientnetv2",
            (ModelNames.MOBILENETV3, None): "mobilenetv3",
        }[(model_name, version)]
        timm_variant = variant.replace("-", "_").lower()

        ref_model = timm.create_model(f"tf_{timm_model_name}_{timm_variant}", pretrained=False)
        expected_param_count = 0
        for name, param in ref_model.named_parameters(recurse=True):
            if re.match(r".*bn\d.bias", name) is not None:
                # moving_mean and moving_variance of BatchNorm are not considered parameters
                # in the reference model. They are however counted in the jax model.
                # Thus multiply number of bias parameters by 3 to account for this.
                multiplier = 3
            else:
                multiplier = 1
            expected_param_count += param.numel() * multiplier

        self.assertEqual(count_model_params(init_params_classifier), expected_param_count)

    @parameterized.product(
        model_variant=(
            (ModelNames.MOBILENETV3, "small-minimal-100", None),
            (ModelNames.EFFICIENTNET, "B0", "V1"),
            (ModelNames.EFFICIENTNET, "lite0", "V1"),
            (ModelNames.EFFICIENTNET, "B0", "V2"),
        ),
        is_training=(False, True),
        endpoints_mode=tuple(EndpointsMode),
    )
    def test_shapes(self, model_variant, is_training, endpoints_mode):
        model_name, variant, version = model_variant
        cfg = named_model_configs(model_name, variant, efficientnet_version=version)
        model: MobileNets = cfg.set(name="backbone", endpoints_mode=endpoints_mode).instantiate(
            parent=None
        )
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=init_params,
            inputs=dict(input_batch=jax.tree.map(jnp.asarray, inputs)),
        )
        self.assertListEqual(
            sorted(outputs.keys()),
            sorted(model.endpoints_dims.keys()),
        )
        for k in outputs:
            self.assertEqual(outputs[k].shape[-1], model.endpoints_dims[k])
        self.assertEqual(outputs["embedding"].shape[-1], model.config.output_dim)

    @parameterized.product(
        endpoints_mode=tuple(EndpointsMode),
        model_variant=(
            (ModelNames.MOBILENETV3, "small-minimal-100", None),
            (ModelNames.EFFICIENTNET, "B0", "V1"),
            (ModelNames.EFFICIENTNET, "lite0", "V1"),
            (ModelNames.EFFICIENTNET, "B0", "V2"),
        ),
    )
    def test_feature_maps(self, endpoints_mode, model_variant):
        model_name, variant, version = model_variant
        cfg = named_model_configs(model_name, variant, efficientnet_version=version)
        cfg.embedding_layer = None
        model: MobileNets = cfg.set(name="backbone", endpoints_mode=endpoints_mode).instantiate(
            parent=None
        )
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        outputs, _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=init_params,
            inputs=dict(input_batch=jax.tree.map(jnp.asarray, inputs)),
        )
        expected_feature_dims = {
            (ModelNames.MOBILENETV3, "small-minimal-100", None): {
                EndpointsMode.DEFAULT: {
                    "2": 16,
                    "3": 24,
                    "4": 24,
                    "5": 40,
                    "6": 40,
                    "7": 40,
                    "8": 48,
                    "9": 48,
                    "10": 96,
                    "11": 96,
                    "12": 96,
                    "13": 576,
                },
                EndpointsMode.LASTBLOCKS: {"stem": 16, "1": 16, "2": 24, "3": 48, "4": 576},
            },
            (ModelNames.EFFICIENTNET, "B0", "V1"): {
                EndpointsMode.DEFAULT: {
                    "2": 16,
                    "3": 24,
                    "4": 24,
                    "5": 40,
                    "6": 40,
                    "7": 80,
                    "8": 80,
                    "9": 80,
                    "10": 112,
                    "11": 112,
                    "12": 112,
                    "13": 192,
                    "14": 192,
                    "15": 192,
                    "16": 192,
                    "17": 320,
                },
                EndpointsMode.LASTBLOCKS: {"1": 16, "2": 24, "3": 40, "4": 112, "5": 320},
            },
            (
                ModelNames.EFFICIENTNET,
                "lite0",
                "V1",
            ): {  # Identical to (ModelNames.EFFICIENTNET, "B0", "V1").
                EndpointsMode.DEFAULT: {
                    "2": 16,
                    "3": 24,
                    "4": 24,
                    "5": 40,
                    "6": 40,
                    "7": 80,
                    "8": 80,
                    "9": 80,
                    "10": 112,
                    "11": 112,
                    "12": 112,
                    "13": 192,
                    "14": 192,
                    "15": 192,
                    "16": 192,
                    "17": 320,
                },
                EndpointsMode.LASTBLOCKS: {"1": 16, "2": 24, "3": 40, "4": 112, "5": 320},
            },
            (ModelNames.EFFICIENTNET, "B0", "V2"): {
                EndpointsMode.DEFAULT: {
                    "2": 16,
                    "3": 32,
                    "4": 32,
                    "5": 48,
                    "6": 48,
                    "7": 96,
                    "8": 96,
                    "9": 96,
                    "10": 112,
                    "11": 112,
                    "12": 112,
                    "13": 112,
                    "14": 112,
                    "15": 192,
                    "16": 192,
                    "17": 192,
                    "18": 192,
                    "19": 192,
                    "20": 192,
                    "21": 192,
                    "22": 192,
                },
                EndpointsMode.LASTBLOCKS: {"1": 16, "2": 32, "3": 48, "4": 112, "5": 192},
            },
        }
        # Ensure equality between actual output and registered endpoint.
        self.assertListEqual(
            sorted(outputs.keys()),
            sorted(model.endpoints_dims.keys()),
        )
        # Compare with expected outputs.
        self.assertSetEqual(
            set(expected_feature_dims[model_variant][endpoints_mode].keys()),
            set(model.endpoints_dims.keys()),
        )
        # Compare expected output dims.
        for k in outputs:
            self.assertEqual(
                expected_feature_dims[model_variant][endpoints_mode][k], model.endpoints_dims[k]
            )


class MobileNetV3EmbeddingTest(parameterized.TestCase):
    """Tests MobileNetV3Embedding."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        width = 16
        height = 24
        hidden_dim = 8
        output_dim = 13

        inputs = np.random.uniform(-1, 1, [batch_size, height, width, hidden_dim]).astype(
            np.float32
        )
        cfg = MobileNetV3Embedding.default_config().set(
            name="test",
            input_dim=hidden_dim,
            output_dim=output_dim,
        )
        model: MobileNetV3Embedding = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(x=inputs),
        )

        expected_shape = (batch_size, output_dim)
        self.assertEqual(expected_shape, output.shape)


class EfficientNetEmbeddingTest(parameterized.TestCase):
    """Tests EfficientNetEmbedding."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        width = 16
        height = 24
        hidden_dim = 8
        output_dim = 13

        inputs = np.random.uniform(-1, 1, [batch_size, height, width, hidden_dim]).astype(
            np.float32
        )
        cfg = EfficientNetEmbedding.default_config().set(
            name="test",
            input_dim=hidden_dim,
            output_dim=output_dim,
        )
        model: EfficientNetEmbedding = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(x=inputs),
        )

        expected_shape = (batch_size, output_dim)
        self.assertEqual(expected_shape, output.shape)


if __name__ == "__main__":
    absltest.main()
