# Copyright © 2023 Apple Inc.

"""Tests Conformer layers."""
import jax
import numpy as np
import torch
from absl import logging
from absl.testing import absltest, parameterized
from fairseq.modules import conformer_layer as fairseq_conformer
from jax import numpy as jnp

from axlearn.common import utils
from axlearn.common.attention import build_remat_spec, sinusoidal_positional_embeddings
from axlearn.common.conformer import (
    ConformerLayer,
    RepeatedConformerLayer,
    compute_attention_logit_biases,
)
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.t5 import T5RelativePositionalEmbedding
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer


class ConformerLayerTest(TestCase):
    """Tests Conformer layers."""

    def test_against_fairseq(self):
        """Compares our implementation against the conformer implementation from fairseq/ESPNET.

        The fairseq implementation does not respect padding when computing convolution, so for
        comparison we use a small convolution window (3) and only compare the prefixes that are not
        affected by padding.
        """
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        dim, num_heads = 6, 2
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.lconv.vlog = 5
        cfg.lconv.linear1.bias = cfg.lconv.linear2.bias = False  # follow ESPNET setup
        cfg.lconv.conv.window = 3  # use a small window to handle the padding difference.
        cfg.self_attention.attention.num_heads = num_heads
        layer = cfg.instantiate(parent=None)  # type: ConformerLayer
        ref_layer = fairseq_conformer.ConformerEncoderLayer(
            embed_dim=dim,
            ffn_embed_dim=dim * 4,
            dropout=0,
            attention_heads=num_heads,
            use_fp16=False,
            depthwise_conv_kernel_size=3,
            activation_fn="swish",
            attn_type="espnet",
            pos_enc_type="rel_pos",
        )
        batch_size, seq_len, min_num_tokens = 2, 10, 5
        inputs = jax.random.normal(jax.random.PRNGKey(123), [batch_size, seq_len, dim])
        num_tokens = jax.random.randint(
            jax.random.PRNGKey(101),
            minval=min_num_tokens,
            maxval=seq_len + 1,
            shape=[batch_size],
        )
        # [batch_size, seq_len].
        paddings = jnp.arange(seq_len)[None, :] >= num_tokens[:, None]
        # [1, 2 * seq_len - 1, model_dim].
        pos_emb = jnp.expand_dims(
            sinusoidal_positional_embeddings(jnp.arange(seq_len - 1, -seq_len, -1), dim=dim), 0
        )
        test_outputs, (ref_outputs, _) = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(inputs=inputs, paddings=paddings),
            ref_inputs=as_torch_tensor(
                dict(
                    # fairseq_conformer requires time-major inputs
                    # (except for encoder_padding_mask)
                    x=inputs.transpose([1, 0, 2]),
                    encoder_padding_mask=paddings,  # mask.transpose([1, 0]),
                    position_emb=pos_emb.transpose([1, 0, 2]),
                )
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
            # moving_{mean,variance} of the BatchNorm layer are not included
            require_same_num_params=False,
        )
        # [batch_size, seq_len, model_dim].
        ref_outputs = utils.as_tensor(ref_outputs).transpose([1, 0, 2])
        # Only look at [:min_num_tokens - 1] because fairseq does not fully respect padding.
        test_outputs = test_outputs[:, : min_num_tokens - 1]
        ref_outputs = ref_outputs[:, : min_num_tokens - 1]
        logging.info("test_outputs=%s", test_outputs[0])
        logging.info("ref_outputs=%s", ref_outputs[0])
        assert_allclose(test_outputs, ref_outputs)

    @parameterized.parameters(False, True)
    def test_respect_paddings(self, is_training):
        """Tests that ConformerLayer respects paddings.

        Generates two input sequences with identical data at the non-pad positions but different
        data at the pad positions. Checks that the outputs at the non-pad positions are the same.
        """
        dim, num_heads = 6, 2
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        layer = cfg.instantiate(parent=None)  # type: ConformerLayer
        batch_size, seq_len, num_tokens = 2, 10, 7
        # [batch_size, seq_len, dim] with the same data across sequences.
        inputs = jnp.tile(
            jax.random.normal(jax.random.PRNGKey(123), [1, seq_len, dim]), [batch_size, 1, 1]
        )
        # [batch_size, seq_len].
        paddings = jnp.tile(jnp.arange(seq_len)[None, :] >= num_tokens, [batch_size, 1])
        # Generate different padding data.
        padding_data = jax.random.normal(jax.random.PRNGKey(124), [batch_size, seq_len, dim])
        # Generate input sequences with the same data at non-pad positions.
        inputs_with_different_paddings = jnp.where(paddings[:, :, None], padding_data, inputs)
        self.assertAlmostEqual(
            inputs_with_different_paddings[0, :num_tokens].sum(),
            inputs_with_different_paddings[1, :num_tokens].sum(),
        )
        self.assertNotAlmostEqual(
            inputs_with_different_paddings[0, num_tokens:].sum(),
            inputs_with_different_paddings[1, num_tokens:].sum(),
        )
        state = layer.initialize_parameters_recursively(jax.random.PRNGKey(100))
        outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs_with_different_paddings, paddings=paddings),
            is_training=is_training,
            prng_key=jax.random.PRNGKey(200),
            state=state,
        )
        # Check that the outputs are the same despite differences in padding.
        assert_allclose(outputs[0, :num_tokens], outputs[1, :num_tokens])

    def test_repeated_conformer_config(self):
        """Tests RepeatedConformerLayer config.

        It tests the ConformerLayer default config is correctly set in RepeatedConformerLayer.
        """
        dim, num_heads = 6, 2
        cfg = RepeatedConformerLayer.default_config().set(
            name="repeat_conformer", input_dim=dim, num_layers=3
        )
        cfg.layer.self_attention.attention.num_heads = num_heads
        for ff_cfg in (cfg.layer.ff_start, cfg.layer.ff_end):
            self.assertEqual(ff_cfg.hidden_dim.scale, 4)
            self.assertEqual(ff_cfg.residual_weight, 0.5)
            self.assertEqual(ff_cfg.activation, "nn.silu")
        self.assertEqual(cfg.layer.self_attention.attention.input_linear.layer.bias, True)

    @parameterized.parameters((True, True), (False, True), (True, False), (False, False))
    def test_repeated_conformer_forward(self, checkpoint_self_attention, checkpoint_feed_forward):
        """Tests RepeatedConformerLayer."""
        dim, num_heads = 6, 2
        # Create a conformer layer.
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        layer = cfg.instantiate(parent=None)  # type: ConformerLayer

        # Create a Repeat Conformer layer.
        num_layers = 5
        repeat_cfg = RepeatedConformerLayer.default_config().set(
            name="repeat_conformer", input_dim=dim, num_layers=num_layers
        )
        repeat_cfg.layer.self_attention.attention.num_heads = num_heads
        repeat_cfg.layer.remat_spec = build_remat_spec(
            repeat_cfg,
            self_attention=checkpoint_self_attention,
            feed_forward=checkpoint_feed_forward,
        )
        repeat_layer = repeat_cfg.instantiate(parent=None)  # type: RepeatedConformerLayer
        repeat_state = repeat_layer.initialize_parameters_recursively(jax.random.PRNGKey(100))
        # Generate synthetic inputs.
        batch_size, seq_len, min_num_tokens = 4, 10, 5
        inputs = jax.random.normal(jax.random.PRNGKey(123), [batch_size, seq_len, dim]) * 10e6
        num_tokens = jax.random.randint(
            jax.random.PRNGKey(101),
            minval=min_num_tokens,
            maxval=seq_len + 1,
            shape=[batch_size],
        )
        # [batch_size, seq_len].
        paddings = jnp.arange(seq_len)[None, :] >= num_tokens[:, None]

        # disable dropout.
        is_training = False
        outputs = inputs
        for ll in range(num_layers):
            # Run a stack of layers by loop
            state_i = jax.tree_util.tree_map(lambda param, i=ll: param[i], repeat_state)["repeat"][
                "layer"
            ]
            outputs, _ = F(
                layer,
                inputs=dict(inputs=outputs, paddings=paddings),
                is_training=is_training,
                prng_key=jax.random.PRNGKey(200),
                state=state_i,
            )
        with utils.numeric_checks(False):
            repeat_outs, _ = F(
                repeat_layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                is_training=is_training,
                prng_key=jax.random.PRNGKey(200),
                state=repeat_state,
            )

        self.assertNestedAllClose(outputs, repeat_outs)

    def test_rel_pos_emb(self):
        dim, num_heads = 6, 2
        # Create a conformer layer.
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        cfg.rel_pos_emb = T5RelativePositionalEmbedding.default_config().set(
            bidirectional=True, num_buckets=128, max_distance=256
        )
        with self.assertRaisesRegex(
            ValueError, "rel_pos_emb should only be set in MultiheadAttention"
        ):
            _ = cfg.instantiate(parent=None)  # type: ConformerLayer

    @parameterized.parameters(
        (4, 10, None, None), (2, 12, 1, None), (6, 20, None, 3), (7, 30, 4, 6), (2, 10, -1, -2)
    )
    # pylint: disable-next=no-self-use
    def test_attention_logit_biases(self, batch_size, seq_len, left_context, right_context):
        lengths = jax.random.randint(
            jax.random.PRNGKey(3), shape=(batch_size,), minval=0, maxval=seq_len + 1
        )
        paddings = jnp.arange(seq_len)[None, :] >= lengths[:, None]
        if (left_context is not None and left_context < 0) or (
            right_context is not None and right_context < 0
        ):
            with self.assertRaises(ValueError):
                compute_attention_logit_biases(
                    paddings=paddings, left_context=left_context, right_context=right_context
                )
        else:
            logit_bias = compute_attention_logit_biases(
                paddings=paddings, left_context=left_context, right_context=right_context
            )
            expected = np.zeros((batch_size, 1, seq_len, seq_len))
            if left_context is None:
                left_context = seq_len
            if right_context is None:
                right_context = seq_len
            for i in range(batch_size):
                for query in range(lengths[i]):
                    for key in range(lengths[i]):
                        if query - left_context <= key <= query + right_context:
                            expected[i, 0, query, key] = 1
            assert_allclose(jnp.exp(logit_bias), expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
