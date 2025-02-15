# Copyright © 2024 Apple Inc.

"""Tests for ASR model layers."""

import jax.numpy as jnp
import jax.random
from absl.testing import parameterized

from axlearn.audio.decoder_asr import (
    BaseASRDecoderModel,
    CTCDecoderModel,
    LASDecoderModel,
    TransducerDecoderModel,
)
from axlearn.audio.encoder_asr import ASREncoder, SpeechContextNetwork, SpeechFeatureLayer
from axlearn.audio.model_asr import ASRModel
from axlearn.common.attention import StackedTransformerLayer, TransformerAttentionLayer
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.layers import set_dropout_rate_recursively
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor


def _model_config(decoder_cfg: BaseASRDecoderModel.Config) -> ASRModel.Config:
    subsampled_dim, encoder_dim, vocab_size = 12, 36, 16
    num_filters, sampling_rate, window_size_ms, window_step_ms = 80, 16000, 25, 10
    num_layers, num_heads, dropout_rate = 2, 4, 0.1

    cfg: ASRModel.Config = ASRModel.default_config().set(
        encoder=ASREncoder.default_config(),
        name="test-model",
    )
    cfg.encoder.dim = encoder_dim
    # Feature layer.
    cfg.encoder.feature = SpeechFeatureLayer.default_config()
    cfg.encoder.feature.output_dim = subsampled_dim
    cfg.encoder.feature.frontend.set(
        num_filters=num_filters,
        sample_rate=sampling_rate,
        frame_size_ms=window_size_ms,
        hop_size_ms=window_step_ms,
        mel_floor=1.0,
    )
    cfg.encoder.feature.augmenter.freq_mask_sampler.set(max_num_masks=2, max_mask_length=27)
    cfg.encoder.feature.augmenter.time_mask_sampler.set(
        max_num_masks_ratio=0.05, max_mask_length=10
    )
    # Context network.
    cfg.encoder.context = SpeechContextNetwork.default_config()
    cfg.encoder.context.dropout.rate = dropout_rate
    cfg.encoder.context.context.num_layers = num_layers
    cfg.encoder.context.context.layer.self_attention.attention.num_heads = num_heads
    set_dropout_rate_recursively(cfg.encoder.context.context.layer, dropout_rate=dropout_rate)
    cfg.decoder = decoder_cfg.set(vocab_size=vocab_size)
    return cfg


def _fake_input_batch(prng_key: Tensor, *, pad_id: int, eos_id: int):
    batch_size, max_src_len = 4, 4000
    src_inputs = jax.random.uniform(
        prng_key, minval=-(2**15), maxval=2**15, shape=[batch_size, max_src_len]
    )
    src_length = jnp.array([0, 3000, 4000, 4500])
    src_paddings = (jnp.arange(max_src_len)[None, :] >= src_length[:, None]).astype(jnp.int32)
    input_ids = jnp.array(
        [
            [1, eos_id, pad_id, pad_id, pad_id],
            [1, 5, 10, 3, eos_id],
            [1, 12, 13, 14, 15],
            [1, 11, 12, 13, 14],
        ]
    )
    target_labels = jnp.array(
        [
            [eos_id, pad_id, pad_id, pad_id, pad_id],
            [5, 10, 3, eos_id, pad_id],
            [12, 13, 14, 15, 11],
            [11, 12, 13, 14, eos_id],
        ]
    )
    return dict(
        source=dict(inputs=src_inputs, paddings=src_paddings),
        target=dict(input_ids=input_ids),
        target_labels=target_labels,
    )


def _ctc_decoder_config() -> CTCDecoderModel.Config:
    return CTCDecoderModel.default_config()


def _rnnt_decoder_config() -> TransducerDecoderModel.Config:
    cfg: TransducerDecoderModel.Config = TransducerDecoderModel.default_config().set(
        lm_dim=6,
        joint_dim=6,
    )
    cfg.prediction_network.rnn_cell.hidden_dim = None
    cfg.prediction_network.emb_dim = 8
    return cfg


def _las_decoder_config(vocab_size: int) -> LASDecoderModel.Config:
    num_heads = 2
    cfg = LASDecoderModel.default_config().set(
        vocab_size=vocab_size,
        decoder=gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=1,
            hidden_dim=10,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            max_position_embeddings=10,
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        ),
    )
    cfg.decoder.pad_token_id = -1
    transformer_cfg = cfg.decoder.transformer.layer
    # Add cross attention.
    transformer_cfg.cross_attention = TransformerAttentionLayer.default_config()
    transformer_cfg.cross_attention.attention.num_heads = num_heads
    return cfg


class ASRModelTest(TestCase):
    """Tests ASRModel."""

    @parameterized.parameters(
        (True, "forward", "ctc"),
        (False, "forward", "ctc"),
        (False, "beam_search_decode", "ctc"),
        (False, "predict", "ctc"),
        (True, "forward", "rnnt"),
        (False, "forward", "rnnt"),
        (False, "beam_search_decode", "rnnt"),
        (True, "forward", "las"),
        (False, "forward", "las"),
        (False, "beam_search_decode", "las"),
    )
    def test_asr_model(self, is_training: bool, method: str, decoder: str):
        batch_size, vocab_size, max_src_len = 4, 16, 4000
        if decoder == "ctc":
            pad_id = eos_id = -1
            decoder_cfg = _ctc_decoder_config()
        elif decoder == "rnnt":
            pad_id, eos_id = -1, 2
            decoder_cfg = _rnnt_decoder_config()
        elif decoder == "las":
            pad_id, eos_id = -1, 1
            decoder_cfg = _las_decoder_config(vocab_size)
        else:
            raise NotImplementedError(decoder)

        prng_key, init_key, data_key = jax.random.split(jax.random.PRNGKey(123), num=3)

        cfg = _model_config(decoder_cfg)
        layer: ASRModel = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(init_key)
        input_batch = _fake_input_batch(data_key, pad_id=pad_id, eos_id=eos_id)
        common_kwargs = dict(
            is_training=is_training, prng_key=prng_key, state=layer_params, method=method
        )

        if method == "forward":
            inputs = dict(input_batch=input_batch, return_aux=True)
            (loss, per_example), _ = F(layer, inputs=inputs, **common_kwargs)
            self.assertEqual((batch_size,), per_example["per_example_loss"].shape)
            self.assertGreater(loss, 0.0)
        elif method == "beam_search_decode":
            inputs = dict()
            if decoder == "las":
                max_decode_len = 300
                input_batch["prefix"] = jnp.ones([batch_size, 1], dtype=jnp.int32) * eos_id
                inputs.update(max_decode_len=max_decode_len)
            elif decoder == "rnnt":
                max_decode_len = 1000
                inputs.update(max_decode_len=max_decode_len)
            elif decoder == "ctc":
                encoder_output_shape = layer.encoder.feature.output_shape(
                    input_shape=input_batch["source"]["inputs"].shape
                )
                max_decode_len = encoder_output_shape[1]

            num_decodes = 4
            inputs.update(num_decodes=num_decodes, input_batch=input_batch)
            beam_search_outputs, _ = F(layer, inputs=inputs, **common_kwargs)
            self.assertEqual(
                (batch_size, num_decodes, max_decode_len), beam_search_outputs.sequences.shape
            )
        elif method == "predict":
            prediction_outputs, _ = F(layer, inputs=dict(input_batch=input_batch), **common_kwargs)
            encoder_feature_shape = layer.encoder.feature.output_shape(
                input_shape=[batch_size, max_src_len]
            )
            self.assertSequenceEqual(
                encoder_feature_shape[:2] + [vocab_size], prediction_outputs["logits"].shape
            )
        else:
            raise NotImplementedError(method)
