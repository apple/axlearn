# Copyright © 2023 Apple Inc.

"""Tests speech encoder layers."""

import jax.random
import pytest
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.audio.asr_encoder import ASREncoder, SpeechContextNetwork, SpeechFeatureLayer
from axlearn.audio.test_utils import fake_audio
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor, shapes


def _fake_audio_pairs(*, prng_key: Tensor, batch_size: int, seq_len: int):
    # Produce inputs s.t. inputs[:batch_size//2] and inputs[batch_size//2:] have the same values
    # only when padding mask is applied.
    inputs, paddings = fake_audio(prng_key=prng_key, batch_size=batch_size // 2, seq_len=seq_len)
    inputs = jnp.tile(inputs, [2, 1])
    paddings = jnp.tile(paddings, [2, 1])
    padding_data = jax.random.normal(jax.random.PRNGKey(135), paddings.shape)
    inputs = jnp.where(paddings, padding_data, inputs)
    return inputs, paddings


class SpeechFeatureLayerTest(TestCase):
    """Tests SpeechFeatureLayer."""

    @parameterized.parameters([True, False])
    @pytest.mark.fp64
    def test_speech_feature_layer(self, is_training: bool):
        num_filters, sample_rate, frame_size_ms, hop_size_ms = 80, 16000, 25, 10
        hidden_dim, output_dim = 32, 16

        cfg: SpeechFeatureLayer.Config = SpeechFeatureLayer.default_config().set(
            output_dim=output_dim
        )
        cfg.frontend.set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        cfg.augmenter.freq_mask_sampler.set(max_num_masks=2, max_mask_length=27)
        cfg.augmenter.time_mask_sampler.set(max_num_masks_ratio=0.05, max_mask_length=10)
        cfg.subsampler.set(hidden_dim=hidden_dim)

        layer: SpeechFeatureLayer = cfg.set(name="test").instantiate(parent=None)
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        self.assertEqual(
            {
                "frontend": {},
                "augmenter": {"freq_mask_sampler": {}, "time_mask_sampler": {}},
                "subsampler": {
                    "conv1": dict(weight=(3, 3, 1, hidden_dim), bias=(hidden_dim,)),
                    "conv2": dict(weight=(3, 3, hidden_dim, output_dim), bias=(output_dim,)),
                },
            },
            shapes(layer_params),
        )
        max_seconds, sampling_rate = 8, 16_000
        batch_size, seq_len = 4, max_seconds * sampling_rate
        inputs, paddings = _fake_audio_pairs(
            prng_key=input_key, batch_size=batch_size, seq_len=seq_len
        )

        # Slightly higher diff without fp64 from conv subsampler on jax 0.4.21.
        inputs = inputs.astype(jnp.float64)
        layer_params = jax.tree_map(lambda x: x.astype(jnp.float64), layer_params)

        output_batch, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=is_training,
            prng_key=prng_key,
            state=layer_params,
        )
        outputs, output_paddings = output_batch["outputs"], output_batch["paddings"]
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertSequenceEqual(outputs.shape, output_shape)
        self.assertSequenceEqual(output_paddings.shape, output_shape[:2])
        self.assertTrue(jnp.all(output_paddings[:2] == output_paddings[2:]))

        # If is_training, outputs should always be different due to augmentation.
        # Otherwise, outputs should be the same despite differences in padding.
        self.assertEqual(not is_training, jnp.allclose(outputs[:2], outputs[2:]))

        self.assertTrue(
            {
                "activations/subsampler_inputs_mean",
                "activations/subsampler_inputs_norm",
                "activations/subsampler_outputs_mean",
                "activations/subsampler_outputs_norm",
            }.issubset(set(output_collections.summaries["subsampler"]))
        )


class SpeechContextNetworkTest(TestCase):
    """Tests SpeechContextNetwork."""

    @parameterized.parameters([True, False])
    def test_speech_context_network(self, is_training: bool):
        input_dim, output_dim, dropout_rate, num_layers = 32, 16, 0.2, 2

        cfg = SpeechContextNetwork.default_config().set(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        cfg.dropout.rate = dropout_rate
        cfg.context.num_layers = num_layers
        cfg.context.layer.self_attention.attention.num_heads = 4
        cfg.context.layer.lconv.dropout.rate = dropout_rate

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key, length_key = jax.random.split(prng_key, num=4)
        layer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate inputs.
        batch_size, seq_len = 4, 10
        inputs = jnp.tile(
            jax.random.normal(input_key, [batch_size // 2, seq_len, input_dim]), [2, 1, 1]
        )
        lengths = jnp.tile(
            jax.random.randint(length_key, shape=[batch_size // 2, 1], minval=0, maxval=seq_len),
            [2, 1],
        )
        paddings = jnp.arange(seq_len)[None, :] >= lengths
        padding_data = jax.random.normal(jax.random.PRNGKey(135), inputs.shape)
        inputs = jnp.where(paddings[..., None], padding_data, inputs)

        # Compute outputs.
        output_batch, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=is_training,
            prng_key=prng_key,
            state=layer_params,
        )
        outputs, output_paddings = output_batch["outputs"], output_batch["paddings"]
        self.assertSequenceEqual(outputs.shape, (batch_size, seq_len, output_dim))
        self.assertTrue(jnp.all(output_paddings == paddings))

        # If is_training, outputs should always be different due to augmentation.
        # Otherwise, outputs should be the same despite differences in padding.
        self.assertEqual(not is_training, bool(jnp.allclose(outputs[:2], outputs[2:])))

        outputs = outputs * (1 - output_paddings[:, :, None])
        weights = jnp.sum(1 - output_paddings)
        output_norms = jnp.sqrt(jnp.sum(outputs**2, axis=2)) / jnp.sqrt(output_dim)
        expected_outputs_mean = jnp.sum(outputs) / weights / output_dim
        expected_outputs_norm = jnp.sum(output_norms) / weights

        self.assertNestedAllClose(
            output_collections.summaries["activations/speech_context_mean"].mean,
            expected_outputs_mean,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/speech_context_norm"].mean,
            expected_outputs_norm,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/speech_context_mean"].weight, weights
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/speech_context_norm"].weight, weights
        )


class ASREncoderTest(TestCase):
    """Tests ASREncoder."""

    @parameterized.product(is_training=[True, False], use_augmenter=[True, False])
    @pytest.mark.fp64
    def test_asr_encoder(self, is_training: bool, use_augmenter: bool):
        conv_dim, output_dim = 12, 36
        num_filters, sample_rate, frame_size_ms, hop_size_ms = 80, 16000, 25, 10
        num_layers, num_heads, dropout_rate = 2, 4, 0.0

        cfg = ASREncoder.default_config().set(
            dim=output_dim,
            feature=SpeechFeatureLayer.default_config(),
            context=SpeechContextNetwork.default_config(),
        )
        # Feature layers.
        cfg.feature.frontend.set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        if use_augmenter:
            cfg.feature.augmenter.freq_mask_sampler.set(max_num_masks=2, max_mask_length=27)
            cfg.feature.augmenter.time_mask_sampler.set(
                max_num_masks_ratio=0.05, max_mask_length=10
            )
        else:
            cfg.feature.augmenter = None
        cfg.feature.subsampler.hidden_dim = conv_dim
        cfg.feature.output_dim = conv_dim

        # Context layers.
        cfg.context.dropout.rate = dropout_rate
        cfg.context.context.num_layers = num_layers
        cfg.context.context.layer.self_attention.attention.num_heads = num_heads
        cfg.context.context.layer.lconv.dropout.rate = dropout_rate

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer: ASREncoder = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate inputs.
        batch_size, seq_len = 4, sample_rate * 8
        inputs, paddings = _fake_audio_pairs(
            prng_key=input_key, batch_size=batch_size, seq_len=seq_len
        )

        # Slightly higher diff without fp64 from conv subsampler on jax 0.4.21.
        inputs = inputs.astype(jnp.float64)
        layer_params = jax.tree_map(lambda x: x.astype(jnp.float64), layer_params)

        output_batch, _ = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=is_training,
            prng_key=prng_key,
            state=layer_params,
        )
        outputs, output_paddings = output_batch["outputs"], output_batch["paddings"]
        output_shape, _ = F(
            layer.feature,
            inputs=dict(input_shape=[None, seq_len]),
            is_training=False,
            state=layer_params["feature"],
            prng_key=prng_key,
            method="output_shape",
        )
        self.assertEqual(outputs.shape, (batch_size, output_shape[1], output_dim))
        self.assertEqual(output_paddings.shape, (batch_size, output_shape[1]))
        self.assertTrue(jnp.all(output_paddings[:2] == output_paddings[2:]))

        # If is_training and use_augmenter, outputs should always be different due to augmentation.
        # Otherwise, outputs should be the same despite differences in padding.
        self.assertEqual(
            not (is_training and use_augmenter), jnp.allclose(outputs[:2], outputs[2:])
        )
