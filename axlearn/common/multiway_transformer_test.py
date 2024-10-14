# Copyright © 2023 Apple Inc.

"""Tests Multiway transformer layers."""
# pylint: disable=no-member,no-self-use,duplicate-code
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.attention import (
    NEG_INF,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
    build_remat_spec,
    make_causal_biases,
)
from axlearn.common.module import functional as F
from axlearn.common.multiway_transformer import (
    IMAGE_MODALITY,
    TEXT_IMAGE_MODALITY,
    TEXT_MODALITY,
    MultiModalEncoder,
    MultiwayTransformerLayer,
    TransformerAttentionLayer,
    _set_model_config,
)
from axlearn.common.test_utils import assert_allclose
from axlearn.common.utils import VDict, as_tensor, count_model_params
from axlearn.vision import mask_generator


class ModelTest(parameterized.TestCase):
    @parameterized.product(
        num_ffn=(1, 3),
        checkpoint_feed_forward=(False, True),
        checkpoint_self_attention=(False, True),
    )
    def test_stacked_with_multiway_transformer_layer(
        self, num_ffn, checkpoint_feed_forward, checkpoint_self_attention
    ):
        batch_size, tgt_len = 10, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4
        model_dim = 16
        num_heads = 4
        cfg = StackedTransformerLayer.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
            layer=MultiwayTransformerLayer.default_config().set(num_ffn=num_ffn),
        )
        layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        layer_cfg.remat_spec = build_remat_spec(
            cfg, self_attention=checkpoint_self_attention, feed_forward=checkpoint_feed_forward
        )

        # Test forward pass for all experts.
        data = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        for feed_forward_index in range(num_ffn):
            F(
                layer,
                inputs=dict(data=data, feed_forward_index=feed_forward_index),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
        # Test parameter count against the basic TransformerLayer.
        ref_cfg = StackedTransformerLayer.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
            layer=TransformerLayer.default_config().set(),
        )
        ref_layer_cfg = ref_cfg.layer
        ref_layer_cfg.self_attention.attention.set(num_heads=num_heads)
        ref_layer_cfg.feed_forward.hidden_dim = model_dim * 4
        ref_layer = ref_cfg.instantiate(parent=None)
        ref_layer_params = ref_layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        ffn_cfg = TransformerFeedForwardLayer.default_config().set(
            name="test", input_dim=model_dim, hidden_dim=model_dim * 4
        )
        ffn_layer = ffn_cfg.instantiate(parent=None)
        ffn_params = ffn_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # multiway_transformer_params = base_transformer_params + additional_ffn_params
        self.assertEqual(
            count_model_params(layer_params),
            count_model_params(ref_layer_params)
            + (num_ffn - 1) * num_dec_layers * count_model_params(ffn_params),
        )

    def test_transformer_extend_step(self):
        batch_size, src_len, tgt_len = 10, 4, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4

        model_dim = 16
        num_heads = 4
        cfg = StackedTransformerLayer.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
            layer=MultiwayTransformerLayer.default_config(),
        )
        cross_atten_cfg = TransformerAttentionLayer.default_config().set(
            source_dim=model_dim * 2,
            structure="postnorm",
        )
        cross_atten_cfg.attention.set(num_heads=num_heads)
        layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.cross_attention = cross_atten_cfg
        layer_cfg.feed_forward.hidden_dim = model_dim * 4

        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        source = jax.random.normal(jax.random.PRNGKey(456), [batch_size, src_len, model_dim * 2])

        self_attention_logit_biases = make_causal_biases(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )
        return_aux = {"self_attention_probs", "cross_attention_probs"}

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
                return_aux=return_aux,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        initial_state = layer.init_states(target_batch_size=batch_size, target_max_len=tgt_len)
        inputs = dict(
            cached_states=initial_state, cross_attention_data=source, return_aux=return_aux
        )
        decoder_output = jnp.zeros(shape=[tgt_len, batch_size, model_dim])
        decoder_self_attention_probs = jnp.zeros(
            shape=[tgt_len, num_dec_layers, batch_size, num_heads, tgt_len]
        )
        decoder_cross_attention_probs = jnp.zeros(
            shape=[tgt_len, num_dec_layers, batch_size, num_heads, src_len]
        )
        for t in range(tgt_len):
            inputs["data"] = jnp.expand_dims(target[:, t, :], axis=1)
            inputs["self_attention_logit_biases"] = self_attention_logit_biases[
                jnp.newaxis, jnp.newaxis, t, :
            ]
            inputs["cross_attention_logit_biases"] = cross_attention_logit_biases[
                jnp.newaxis, jnp.newaxis, t, :
            ]
            (updated_states, layer_outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = updated_states
            decoder_output = decoder_output.at[t].set(jnp.squeeze(layer_outputs.data, axis=1))
            decoder_self_attention_probs = decoder_self_attention_probs.at[t].set(
                jnp.squeeze(layer_outputs.self_attention_probs, axis=3)
            )
            decoder_cross_attention_probs = decoder_cross_attention_probs.at[t].set(
                jnp.squeeze(layer_outputs.cross_attention_probs, axis=3)
            )
        decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
        decoder_self_attention_probs_transposed = jnp.transpose(
            decoder_self_attention_probs, [1, 2, 3, 0, 4]
        )
        decoder_cross_attention_probs_transposed = jnp.transpose(
            decoder_cross_attention_probs, [1, 2, 3, 0, 4]
        )

        assert_allclose(decoder_out_transposed, forward_outputs.data, atol=1e-6)
        assert_allclose(
            decoder_self_attention_probs_transposed, forward_outputs.self_attention_probs, atol=1e-6
        )
        assert_allclose(
            decoder_cross_attention_probs_transposed,
            forward_outputs.cross_attention_probs,
            atol=1e-6,
        )

    @parameterized.parameters(StackedTransformerLayer, RepeatedTransformerLayer)
    # pylint: disable-next=too-many-statements
    def test_prefill_states(self, transformer_type):
        batch_size, src_len, tgt_len = 10, 4, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4

        model_dim = 16
        num_heads = 4
        cfg = transformer_type.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
            layer=MultiwayTransformerLayer.default_config(),
        )
        cross_atten_cfg = TransformerAttentionLayer.default_config().set(
            source_dim=model_dim * 2,
            structure="postnorm",
        )
        cross_atten_cfg.attention.set(num_heads=num_heads)
        layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.cross_attention = cross_atten_cfg
        layer_cfg.feed_forward.hidden_dim = model_dim * 4

        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        source = jax.random.normal(jax.random.PRNGKey(456), [batch_size, src_len, model_dim * 2])

        self_attention_logit_biases = make_causal_biases(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )
        return_aux = {"self_attention_probs", "cross_attention_probs"}

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
                return_aux=return_aux,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        # Initialize state.
        time_step = jnp.arange(batch_size)
        (initial_states, initial_output), _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                time_step=time_step,
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
                return_aux=return_aux,
            ),
            method="prefill_states",
        )

        # Zero-out outputs starting from initial time_step, and test that we can recover the full
        # outputs by calling extend_step starting from time_step.
        time_step_mask = jnp.arange(tgt_len) < time_step[:, None]
        # [batch, tgt_len, model_dim].
        decoder_output = initial_output.data * time_step_mask[..., None]
        # [num_layers, batch, num_heads, tgt_len, tgt_len].
        decoder_self_attention_probs = (
            initial_output.self_attention_probs * time_step_mask[None, :, None, :, None]
        )
        # [num_layers, batch, num_heads, tgt_len, src_len].
        decoder_cross_attention_probs = (
            initial_output.cross_attention_probs * time_step_mask[None, :, None, :, None]
        )

        # Transpose for simpler updates during extend_step.
        # [batch, tgt_len, model_dim] --> [batch, model_dim, tgt_len].
        decoder_output = jnp.moveaxis(decoder_output, -2, -1)
        # [..., tgt_len, src_len] --> [..., src_len, tgt_len].
        decoder_self_attention_probs = jnp.moveaxis(decoder_self_attention_probs, -2, -1)
        decoder_cross_attention_probs = jnp.moveaxis(decoder_cross_attention_probs, -2, -1)

        # Call extend_step from time_step, ensuring that outputs match.
        inputs = dict(
            cached_states=initial_states, cross_attention_data=source, return_aux=return_aux
        )
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1, model_dim].
            inputs["data"] = jnp.take_along_axis(
                target, time_step[:, None, None], axis=1, mode="clip"
            )
            # [batch=1, tgt_len=1, tgt_len].
            inputs["self_attention_logit_biases"] = jnp.take_along_axis(
                self_attention_logit_biases[None, :, :],
                time_step[:, None, None],
                axis=1,
                mode="clip",
            )
            # [batch=1, tgt_len=1, src_len].
            inputs["cross_attention_logit_biases"] = jnp.take_along_axis(
                cross_attention_logit_biases[None, :, :],
                time_step[:, None, None],
                axis=1,
                mode="clip",
            )
            (updated_states, layer_outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            # Check that updated_states are VDicts for the Repeated layer.
            if transformer_type is RepeatedTransformerLayer:
                self.assertIsInstance(updated_states, VDict)
            inputs["cached_states"] = updated_states

            # [batch, model_dim, tgt_len=1]
            curr_outputs = jnp.moveaxis(layer_outputs.data, -2, -1)
            # [..., tgt_len, tgt_len=1]
            curr_self_attention_probs = jnp.moveaxis(layer_outputs.self_attention_probs, -2, -1)
            # [..., src_len, tgt_len=1]
            curr_cross_attention_probs = jnp.moveaxis(layer_outputs.cross_attention_probs, -2, -1)

            # [batch, 1, tgt_len].
            oh_indices = jax.nn.one_hot(time_step, tgt_len)[:, None, :]
            decoder_output = decoder_output + curr_outputs * oh_indices
            # [num_layers=1, batch, num_heads=1, tgt_len=1, tgt_len].
            oh_indices = oh_indices[None, :, None, :, :]
            decoder_self_attention_probs = (
                decoder_self_attention_probs + curr_self_attention_probs * oh_indices
            )
            decoder_cross_attention_probs = (
                decoder_cross_attention_probs + curr_cross_attention_probs * oh_indices
            )
            time_step = time_step + 1

        # [batch, model_dim, tgt_len] --> [batch, tgt_len, model_dim].
        decoder_output = jnp.moveaxis(decoder_output, -1, -2)
        # [..., src_len, tgt_len] --> [..., tgt_len, src_len].
        decoder_self_attention_probs = jnp.moveaxis(decoder_self_attention_probs, -1, -2)
        decoder_cross_attention_probs = jnp.moveaxis(decoder_cross_attention_probs, -1, -2)

        assert_allclose(decoder_output, forward_outputs.data)
        assert_allclose(decoder_self_attention_probs, forward_outputs.self_attention_probs)
        assert_allclose(decoder_cross_attention_probs, forward_outputs.cross_attention_probs)

    @parameterized.product(
        is_training=(False, True),
        num_modalities=(2, 3),
    )
    def test_multiway_transformer(self, is_training, num_modalities):
        batch_size = 2
        output_dim = 32
        text_vocab_size = 24
        max_text_len = 11

        cfg = MultiModalEncoder.default_config().set(output_dim=output_dim, num_cls_tokens=1)
        _set_model_config(
            cfg,
            num_modalities=num_modalities,
            num_layers=2,
            model_dim=output_dim,
            num_heads=8,
            text_vocab_size=text_vocab_size,
            max_text_len=max_text_len,
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = {
            TEXT_MODALITY: np.random.randint(1, text_vocab_size, size=(batch_size, max_text_len)),
            IMAGE_MODALITY: np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            TEXT_IMAGE_MODALITY: {
                TEXT_MODALITY: np.random.randint(
                    1, text_vocab_size, size=(batch_size, max_text_len)
                ),
                IMAGE_MODALITY: np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(
                    np.float32
                ),
            },
        }
        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=as_tensor(inputs)),
        )
        for modality, embeddings in outputs.items():
            self.assertIn(modality, outputs)
            self.assertEqual(embeddings.shape, (batch_size, output_dim))

    @parameterized.parameters(0, 118)
    def test_model_forward_with_masked_pos(self, num_masking_patches):
        batch_size = 2
        output_dim = 32
        text_vocab_size = 24
        max_text_len = 11
        num_modalities = 3

        cfg = MultiModalEncoder.default_config().set(output_dim=output_dim, num_cls_tokens=1)
        _set_model_config(
            cfg,
            num_modalities=num_modalities,
            num_layers=2,
            model_dim=output_dim,
            num_heads=8,
            text_vocab_size=text_vocab_size,
            max_text_len=max_text_len,
        )
        cfg.use_mask_tokens = True
        layer = cfg.set(name="test").instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = {
            TEXT_MODALITY: np.random.randint(1, text_vocab_size, size=(batch_size, max_text_len)),
            IMAGE_MODALITY: np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            TEXT_IMAGE_MODALITY: {
                TEXT_MODALITY: np.random.randint(
                    1, text_vocab_size, size=(batch_size, max_text_len)
                ),
                IMAGE_MODALITY: np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(
                    np.float32
                ),
            },
        }

        mask_model = mask_generator.MaskingGenerator(
            input_size=(14, 14),
            num_masking_patches=num_masking_patches,
        )
        is_masked = mask_model()
        is_masked = jnp.expand_dims(is_masked, axis=0)
        is_masked = jnp.tile(is_masked, [batch_size, 1, 1])
        is_masked = jnp.reshape(is_masked, (batch_size, 196))

        outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=as_tensor(inputs), is_masked=is_masked),
        )
        # A reference model in which no mask is used.
        ref, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=as_tensor(inputs), is_masked=None),
        )
        if num_masking_patches == 0:
            np.testing.assert_array_equal(outputs[IMAGE_MODALITY], ref[IMAGE_MODALITY])
            np.testing.assert_array_equal(
                outputs[TEXT_IMAGE_MODALITY][IMAGE_MODALITY],
                ref[TEXT_IMAGE_MODALITY][IMAGE_MODALITY],
            )
        else:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                outputs[IMAGE_MODALITY],
                ref[IMAGE_MODALITY],
            )
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                outputs[TEXT_IMAGE_MODALITY][IMAGE_MODALITY],
                ref[TEXT_IMAGE_MODALITY][IMAGE_MODALITY],
            )


if __name__ == "__main__":
    absltest.main()
