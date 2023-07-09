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


# TODO(e_daxberger): Compare model outputs against reference implementation,
# e.g. https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
class MobileNetsTest(parameterized.TestCase):
    """Tests MobileNets."""

    @parameterized.parameters(
        (ModelNames.MOBILENETV3, "small-minimal-100"),
        (ModelNames.MOBILENETV3, "small-100"),
        (ModelNames.MOBILENETV3, "large-minimal-100"),
        (ModelNames.MOBILENETV3, "large-100"),
        # (ModelNames.MOBILENETV3, "small-minimal-075"), # not available in timm
        (ModelNames.MOBILENETV3, "small-075"),
        # (ModelNames.MOBILENETV3, "large-minimal-075"), # not available in timm
        (ModelNames.MOBILENETV3, "large-075"),
        (ModelNames.EFFICIENTNET, "B0"),
        (ModelNames.EFFICIENTNET, "B1"),
        (ModelNames.EFFICIENTNET, "B2"),
        (ModelNames.EFFICIENTNET, "B3"),
        (ModelNames.EFFICIENTNET, "B4"),
        (ModelNames.EFFICIENTNET, "B5"),
        (ModelNames.EFFICIENTNET, "B6"),
        (ModelNames.EFFICIENTNET, "B7"),
        (ModelNames.EFFICIENTNET, "B8"),
        (ModelNames.EFFICIENTNET, "lite0"),
        (ModelNames.EFFICIENTNET, "lite1"),
        (ModelNames.EFFICIENTNET, "lite2"),
        (ModelNames.EFFICIENTNET, "lite3"),
        (ModelNames.EFFICIENTNET, "lite4"),
    )
    def test_param_count(self, model_name, variant):
        cfg = named_model_configs(model_name, variant)
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

        if model_name == ModelNames.EFFICIENTNET:
            timm_model_name = "efficientnet"
        elif model_name == ModelNames.MOBILENETV3:
            timm_model_name = "mobilenetv3"
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
        model=(
            (ModelNames.MOBILENETV3, "small-minimal-100"),
            (ModelNames.EFFICIENTNET, "B0"),
            (ModelNames.EFFICIENTNET, "lite0"),
        ),
        is_training=(False, True),
        endpoints_mode=tuple(EndpointsMode),
    )
    def test_shapes(self, model, is_training, endpoints_mode):
        model_name, variant = model
        cfg = named_model_configs(model_name, variant)
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
            inputs=dict(input_batch=jax.tree_util.tree_map(jnp.asarray, inputs)),
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
            (ModelNames.MOBILENETV3, "small-minimal-100"),
            (ModelNames.EFFICIENTNET, "B0"),
            (ModelNames.EFFICIENTNET, "lite0"),
        ),
    )
    def test_feature_maps(self, endpoints_mode, model_variant):
        cfg = named_model_configs(*model_variant)
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
            inputs=dict(input_batch=jax.tree_util.tree_map(jnp.asarray, inputs)),
        )
        expected_feature_dims = {
            (ModelNames.MOBILENETV3, "small-minimal-100"): {
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
            (ModelNames.EFFICIENTNET, "B0"): {
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
            ): {  # identical to (ModelNames.EFFICIENTNET, "B0")
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
        }
        # ensure equality between actual output and registered endpoint
        self.assertListEqual(
            sorted(outputs.keys()),
            sorted(model.endpoints_dims.keys()),
        )
        # compare with expected outputs
        self.assertSetEqual(
            set(expected_feature_dims[model_variant][endpoints_mode].keys()),
            set(model.endpoints_dims.keys()),
        )
        # compare expected output dims
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
