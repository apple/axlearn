# Copyright © 2024 Apple Inc.

"""Tests for audio adapters."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from axlearn.audio.adapter import ASRModelAdapter, AudioModelAdapter
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose


class AudioModelAdapterTest(TestCase):
    """Tests AudioModelAdapter."""

    def test_forward_basic(self):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(jnp.isfinite(outputs).all())

    def test_forward_with_layer_norm(self):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            use_layer_norm=True,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)

    def test_forward_without_residual(self):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            residual=False,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)
        with self.assertRaises(AssertionError):
            assert_allclose(outputs, inputs)

    def test_forward_with_scaling(self):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32
        adapter_scale = 0.5

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_scale=adapter_scale,
            residual=False,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)

    @parameterized.parameters(["relu", "gelu"])
    def test_forward_with_activation(self, activation: str):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(jnp.isfinite(outputs).all())

    def test_parameter_counts(self):
        input_dim, bottleneck_dim = 256, 64

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            use_layer_norm=True,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        layer_params = layer.initialize_parameters_recursively(prng_key)

        down_proj_weight = layer_params["down_proj"]["weight"]
        down_proj_bias = layer_params["down_proj"]["bias"]
        up_proj_weight = layer_params["up_proj"]["weight"]
        up_proj_bias = layer_params["up_proj"]["bias"]
        layer_norm_scale = layer_params["layer_norm"]["scale"]

        self.assertEqual(down_proj_weight.shape, (input_dim, bottleneck_dim))
        self.assertEqual(down_proj_bias.shape, (bottleneck_dim,))
        self.assertEqual(up_proj_weight.shape, (bottleneck_dim, input_dim))
        self.assertEqual(up_proj_bias.shape, (input_dim,))
        self.assertEqual(layer_norm_scale.shape, (input_dim,))

        total_params = np.prod(down_proj_weight.shape)
        total_params += np.prod(down_proj_bias.shape)
        total_params += np.prod(up_proj_weight.shape)
        total_params += np.prod(up_proj_bias.shape)
        total_params += np.prod(layer_norm_scale.shape)

        self.assertEqual(total_params, 82368)

    @parameterized.parameters([True, False])
    def test_training_vs_eval_mode(self, is_training: bool):
        batch_size, seq_len, input_dim, bottleneck_dim = 4, 10, 128, 32

        cfg = AudioModelAdapter.default_config().set(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        inputs = jax.random.normal(input_key, (batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            inputs=inputs,
            is_training=is_training,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(outputs.shape, inputs.shape)


class ASRModelAdapterTest(TestCase):
    """Tests ASRModelAdapter."""

    def test_encoder_adapter_only(self):
        encoder_dim = 256
        encoder_bottleneck_dim = 64
        batch_size, seq_len = 4, 100

        cfg = ASRModelAdapter.default_config().set(
            encoder_dim=encoder_dim,
            encoder_bottleneck_dim=encoder_bottleneck_dim,
            adapt_encoder=True,
            adapt_decoder=False,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        encoder_features = jax.random.normal(input_key, (batch_size, seq_len, encoder_dim))

        adapted_features = layer.adapt_encoder_features(
            encoder_features,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(adapted_features.shape, encoder_features.shape)

    def test_decoder_adapter_only(self):
        decoder_dim = 256
        decoder_bottleneck_dim = 64
        batch_size, seq_len = 4, 50

        cfg = ASRModelAdapter.default_config().set(
            encoder_dim=128,
            encoder_bottleneck_dim=32,
            decoder_dim=decoder_dim,
            decoder_bottleneck_dim=decoder_bottleneck_dim,
            adapt_encoder=False,
            adapt_decoder=True,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        decoder_features = jax.random.normal(input_key, (batch_size, seq_len, decoder_dim))

        adapted_features = layer.adapt_decoder_features(
            decoder_features,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(adapted_features.shape, decoder_features.shape)

    def test_both_encoders_and_decoders(self):
        encoder_dim, encoder_bottleneck_dim = 256, 64
        decoder_dim, decoder_bottleneck_dim = 256, 64
        batch_size, enc_seq_len, dec_seq_len = 4, 100, 50

        cfg = ASRModelAdapter.default_config().set(
            encoder_dim=encoder_dim,
            encoder_bottleneck_dim=encoder_bottleneck_dim,
            decoder_dim=decoder_dim,
            decoder_bottleneck_dim=decoder_bottleneck_dim,
            adapt_encoder=True,
            adapt_decoder=True,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key1, input_key2 = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        encoder_features = jax.random.normal(input_key1, (batch_size, enc_seq_len, encoder_dim))
        decoder_features = jax.random.normal(input_key2, (batch_size, dec_seq_len, decoder_dim))

        adapted_enc_features = layer.adapt_encoder_features(
            encoder_features,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )
        adapted_dec_features = layer.adapt_decoder_features(
            decoder_features,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        self.assertEqual(adapted_enc_features.shape, encoder_features.shape)
        self.assertEqual(adapted_dec_features.shape, decoder_features.shape)

    def test_no_adaptation(self):
        encoder_dim = 256
        batch_size, seq_len = 4, 100

        cfg = ASRModelAdapter.default_config().set(
            encoder_dim=encoder_dim,
            encoder_bottleneck_dim=64,
            adapt_encoder=False,
            adapt_decoder=False,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        encoder_features = jax.random.normal(input_key, (batch_size, seq_len, encoder_dim))

        adapted_features = layer.adapt_encoder_features(
            encoder_features,
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        assert_allclose(adapted_features, encoder_features)

    def test_direct_call_fallback(self):
        encoder_dim = 256
        batch_size, seq_len = 4, 100

        cfg = ASRModelAdapter.default_config().set(
            encoder_dim=encoder_dim,
            encoder_bottleneck_dim=64,
            adapt_encoder=True,
            adapt_decoder=False,
            dtype=jnp.float32,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        # Initialize params (required for layer setup, but not used in this direct call test)
        _ = layer.initialize_parameters_recursively(jax.random.PRNGKey(123))
        encoder_features = jax.random.normal(
            jax.random.PRNGKey(456), (batch_size, seq_len, encoder_dim)
        )

        adapted_features = layer.adapt_encoder_features(encoder_features, is_training=True)

        self.assertEqual(adapted_features.shape, encoder_features.shape)
