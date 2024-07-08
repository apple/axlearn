# Copyright © 2023 Apple Inc.

"""Tests feature pyramid network implementations."""
import jax.numpy as jnp
import jax.random
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose, flatten_items
from axlearn.vision.fpn import (
    FPN,
    BiFPN,
    BiFPNLayer,
    DepthwiseSeparableConvolution,
    FusionMethod,
    LayerType,
    ResampleFeatures,
    RescaleImageMethod,
    SimpleFPN,
    WeightedFeatureFusion,
    bifpn_config,
    bifpn_layer_config,
)


class FPNTest(parameterized.TestCase):
    """Tests FPN."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        hidden_dim = 8
        min_level = 3
        max_level = 7
        backbone_max_level = 5
        image_size = 256
        fpn_hidden_dim = 256

        inputs = {}
        input_dims = {}
        for level in range(0, backbone_max_level + 1):
            inputs[str(level)] = np.random.uniform(
                -1, 1, [batch_size, image_size, image_size, hidden_dim]
            ).astype(np.float32)
            input_dims[str(level)] = hidden_dim
            hidden_dim *= 2
            image_size //= 2

        cfg = FPN.default_config().set(
            name="test",
            min_level=min_level,
            max_level=max_level,
            hidden_dim=fpn_hidden_dim,
            input_dims=input_dims,
        )
        model: FPN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (batch_size, 256 // 2**level, 256 // 2**level, fpn_hidden_dim)
            self.assertEqual(expected_shape, outputs[level].shape)


class SimpleFPNTest(parameterized.TestCase):
    """Tests SimpleFPN."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        hidden_dim = 8
        min_level = 2
        max_level = 6
        image_size = 128
        patch_size = 16
        feature_size = image_size // patch_size
        fpn_hidden_dim = 256

        inputs = {
            "4": np.random.uniform(
                -1, 1, [batch_size, feature_size, feature_size, hidden_dim]
            ).astype(np.float32)
        }
        input_dims = {"4": 8}
        cfg = SimpleFPN.default_config().set(
            input_dims=input_dims,
            name="test",
            min_level=min_level,
            max_level=max_level,
            hidden_dim=fpn_hidden_dim,
        )
        model: SimpleFPN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                fpn_hidden_dim,
            )
            self.assertEqual(expected_shape, outputs[level].shape)

    @parameterized.product(
        fpn_min_level=(2, 3),
        fpn_max_level=(6, 7),
    )
    def test_model_min_max_level(self, fpn_min_level, fpn_max_level):
        batch_size = 2
        hidden_dim = 8
        min_level = fpn_min_level
        max_level = fpn_max_level
        image_size = 128
        patch_size = 16
        feature_size = image_size // patch_size
        fpn_hidden_dim = 256

        inputs = {
            "4": np.random.uniform(
                -1, 1, [batch_size, feature_size, feature_size, hidden_dim]
            ).astype(np.float32)
        }
        input_dims = {"4": 8}
        cfg = SimpleFPN.default_config().set(
            input_dims=input_dims,
            name="test",
            min_level=min_level,
            max_level=max_level,
            hidden_dim=fpn_hidden_dim,
        )
        model: SimpleFPN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                fpn_hidden_dim,
            )
            self.assertEqual(expected_shape, outputs[level].shape)


class WeightedFeatureFusionTest(parameterized.TestCase):
    """Tests WeightedFeatureFusion."""

    _weights = jnp.array([[0.75], [1.3], [-0.5]])
    _expected_weights = {
        FusionMethod.SUM: jnp.array([[0.75], [1.3], [-0.5]]),
        FusionMethod.ATTENTION: jnp.array([[0.331152], [0.573970], [0.094876]]),
        FusionMethod.FASTATTENTION: jnp.array([[0.365836], [0.634115], [0.0]]),
    }

    @parameterized.product(
        is_training=(False, True),
        fusion_method=(FusionMethod.SUM, FusionMethod.ATTENTION, FusionMethod.FASTATTENTION),
    )
    def test_model_forward(self, is_training, fusion_method):
        num_inputs = 3
        batch_size = 2
        width = 16
        height = 24
        hidden_dim = 8

        inputs = []
        for _ in range(num_inputs):
            inputs.append(
                np.random.uniform(-1, 1, [batch_size, height, width, hidden_dim]).astype(np.float32)
            )
        cfg = WeightedFeatureFusion.default_config().set(
            name="test",
            num_input_tensors=num_inputs,
        )
        state = {"weight": self._weights}

        cfg = cfg.set(method=fusion_method)
        model: WeightedFeatureFusion = cfg.instantiate(parent=None)
        output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(x=inputs),
        )

        expected_shape = (batch_size, height, width, hidden_dim)
        self.assertEqual(expected_shape, output.shape)

        assert_allclose(
            output,
            jax.lax.squeeze(
                jnp.stack(inputs, axis=-1) @ self._expected_weights[fusion_method],
                dimensions=[-1],
            ),
        )


class DepthwiseSeparableConvolutionTest(parameterized.TestCase):
    """Tests DepthwiseSeparableConvolution."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        input_dim = 8
        output_dim = 12
        dw_kernel_size = 3

        batch_size = 2
        width = 16
        height = 24

        input_tensor = np.random.uniform(-1, 1, [batch_size, height, width, input_dim]).astype(
            np.float32
        )

        cfg = DepthwiseSeparableConvolution.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            depthwise_kernel_size=dw_kernel_size,
        )
        model: DepthwiseSeparableConvolution = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=input_tensor),
        )
        expected_shape = (batch_size, height, width, output_dim)
        self.assertEqual(expected_shape, output.shape)

    @parameterized.product(
        input_dim=(2, 9),
        output_dim=(13, 8),
        dw_kernel_size=(1, 3),
    )
    def test_num_params(self, input_dim, output_dim, dw_kernel_size):
        cfg = DepthwiseSeparableConvolution.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            depthwise_kernel_size=dw_kernel_size,
        )
        model: DepthwiseSeparableConvolution = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        num_params = 0
        for _, param in flatten_items(state):
            num_params += param.size

        # num_params = num_params(dw_conv) + num_params(pw_conv)
        expected_num_params = dw_kernel_size * dw_kernel_size * input_dim + input_dim * output_dim
        self.assertEqual(expected_num_params, num_params)


class ResampleFeaturesTest(parameterized.TestCase):
    """Tests ResampleFeatures."""

    @parameterized.product(
        is_training=(False, True),
        rescale_op=tuple(RescaleImageMethod),
    )
    def test_model_forward(self, is_training, rescale_op):
        input_dim = 8
        output_dim = 12

        batch_size = 2
        width = 17
        height = 24

        input_tensor = np.random.uniform(-1, 1, [batch_size, height, width, input_dim]).astype(
            np.float32
        )

        # Only project
        cfg = ResampleFeatures.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            rescale_op=rescale_op,
        )

        model: ResampleFeatures = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=input_tensor),
        )

        expected_shapes = {
            RescaleImageMethod.IDENTITY: (batch_size, height, width, output_dim),
            RescaleImageMethod.DOUBLE: (batch_size, height * 2, width * 2, output_dim),
            RescaleImageMethod.HALVE: (
                batch_size,
                np.ceil(height / 2),
                np.ceil(width / 2),
                output_dim,
            ),
        }
        self.assertEqual(expected_shapes[rescale_op], output.shape)


class BiFPNLayerTest(parameterized.TestCase):
    """Tests BiFPNLayer."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        hidden_dim = 8
        min_level = 3
        max_level = 5
        image_size = 640
        input_dims = {3: 16, 4: 32, 5: 64}

        inputs = {}
        for level in range(min_level, max_level + 1):
            inputs[level] = np.random.uniform(
                -1,
                1,
                [batch_size, image_size // 2**level, image_size // 2**level, input_dims[level]],
            ).astype(np.float32)

        cfg = bifpn_layer_config(
            hidden_dim=hidden_dim,
            min_level=min_level,
            max_level=max_level,
            layer_type=LayerType.FIRSTLAYER,
        )
        cfg.name = "test"
        cfg.input_dims = input_dims

        model: BiFPNLayer = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                hidden_dim,
            )
            self.assertEqual(expected_shape, outputs[level].shape)


class BiFPNTest(parameterized.TestCase):
    """Tests BiFPN."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        input_dims = {"3": 256, "4": 192, "5": 123}
        batch_size = 2
        hidden_dim = 8
        min_level = 3
        max_level = 7
        num_bifpn_layers = 3
        image_size = 640

        max_input_level = max(int(l) for l in input_dims if l.isdigit())
        min_input_level = min(int(l) for l in input_dims if l.isdigit())
        inputs = {}
        for level in range(min_input_level, max_input_level + 1):
            inputs[str(level)] = np.random.uniform(
                -1,
                1,
                [
                    batch_size,
                    image_size // 2**level,
                    image_size // 2**level,
                    input_dims[str(level)],
                ],
            ).astype(np.float32)

        cfg = bifpn_config(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            min_level=min_level,
            max_level=max_level,
            num_bifpn_layers=num_bifpn_layers,
        )
        cfg = cfg.set(name="test")
        model: BiFPN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                hidden_dim,
            )
            self.assertEqual(expected_shape, outputs[level].shape)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
