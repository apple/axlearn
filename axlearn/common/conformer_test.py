# Copyright © 2023 Apple Inc.

"""Tests Conformer layers."""

import os
from unittest.mock import patch

import jax
import numpy as np
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.attention import build_remat_spec
from axlearn.common.attention_bias import (
    CausalAttentionBias,
    LeftRightWindowAttentionBias,
    SlidingWindowAttentionBias,
)
from axlearn.common.conformer import (
    ConformerLayer,
    LConvLayer,
    RepeatedConformerLayer,
    set_double_shard_weights_config,
)
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import safe_not

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")


class LConvLayerTest(TestCase):
    """Tests Lconv layer."""

    def test_conv_norm_padding(self):
        dim = 2
        cfg = LConvLayer.default_config().set(name="lconv", input_dim=dim)
        layer = cfg.instantiate(parent=None)

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
        segment_ids = (jnp.arange(seq_len)[None, :] < num_tokens[:, None]).astype(jnp.int32)

        # Forward
        state = layer.initialize_parameters_recursively(jax.random.PRNGKey(100))
        with patch.object(layer.conv_norm, "forward", wraps=layer.conv_norm.forward) as mock:
            _ = F(
                layer,
                inputs=dict(inputs=inputs, segment_ids=segment_ids),
                is_training=True,
                prng_key=jax.random.PRNGKey(100),
                state=state,
            )
            self.assertIn("segment_ids", mock.call_args.kwargs)


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
        layer: ConformerLayer = cfg.instantiate(parent=None)
        min_num_tokens = 5

        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_against_fairseq.npy"),
            allow_pickle=True,
        ).item()
        segment_ids = safe_not(testcase["paddings"]).astype(jnp.int32)

        test_outputs, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=testcase["params"],
            inputs=dict(inputs=testcase["inputs"], segment_ids=segment_ids),
        )
        ref_outputs = testcase["outputs"]

        # Only look at [:min_num_tokens - 1] because fairseq does not fully respect padding.
        test_outputs = test_outputs[:, : min_num_tokens - 1]
        ref_outputs = ref_outputs[:, : min_num_tokens - 1]
        logging.info("test_outputs=%s", test_outputs[0])
        logging.info("ref_outputs=%s", ref_outputs[0])
        assert_allclose(test_outputs, ref_outputs)

    @parameterized.parameters(False, True)
    def test_respect_segment_ids(self, is_training):
        """Tests that ConformerLayer respects segment_ids.

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
        segment_ids = jnp.tile(
            (jnp.arange(seq_len)[None, :] < num_tokens).astype(jnp.int32), [batch_size, 1]
        )
        # Generate different padding data.
        padding_data = jax.random.normal(jax.random.PRNGKey(124), [batch_size, seq_len, dim])
        # Generate input sequences with the same data at non-pad positions.
        inputs_with_different_segment_ids = jnp.where(
            segment_ids[:, :, None] == 0, padding_data, inputs
        )
        self.assertAlmostEqual(
            inputs_with_different_segment_ids[0, :num_tokens].sum(),
            inputs_with_different_segment_ids[1, :num_tokens].sum(),
        )
        self.assertNotAlmostEqual(
            inputs_with_different_segment_ids[0, num_tokens:].sum(),
            inputs_with_different_segment_ids[1, num_tokens:].sum(),
        )
        state = layer.initialize_parameters_recursively(jax.random.PRNGKey(100))
        outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs_with_different_segment_ids, segment_ids=segment_ids),
            is_training=is_training,
            prng_key=jax.random.PRNGKey(200),
            state=state,
        )
        # Check that the outputs are the same despite differences in padding.
        assert_allclose(outputs[0, :num_tokens], outputs[1, :num_tokens])

    @parameterized.parameters(None, "lconv_before_ff", "lconv_before_mhsa", "mhsa_before_lconv")
    def test_repeated_conformer_config(self, layer_order):
        """Tests RepeatedConformerLayer config.

        It tests the ConformerLayer default config is correctly set in RepeatedConformerLayer.
        """
        dim, num_heads = 6, 2
        cfg = RepeatedConformerLayer.default_config().set(
            name="repeat_conformer",
            input_dim=dim,
            num_layers=3,
        )
        cfg.layer.self_attention.attention.num_heads = num_heads
        cfg.layer.layer_order = layer_order
        for ff_cfg in (cfg.layer.ff_start, cfg.layer.ff_end):
            self.assertEqual(ff_cfg.hidden_dim.scale, 4)
            self.assertEqual(ff_cfg.residual_weight, 0.5)
            self.assertEqual(ff_cfg.activation, "nn.silu")
        self.assertEqual(cfg.layer.self_attention.attention.input_linear.layer.bias, True)

    @parameterized.product(
        test_remat=(True, False),
        layer_order=(None, "lconv_before_ff", "lconv_before_mhsa", "mhsa_before_lconv"),
    )
    def test_repeated_conformer_forward(self, test_remat, layer_order):
        """Tests RepeatedConformerLayer."""
        dim, num_heads = 6, 2
        # Create a conformer layer.
        cfg = ConformerLayer.default_config().set(
            name="conformer", input_dim=dim, layer_order=layer_order
        )
        cfg.self_attention.attention.num_heads = num_heads
        layer = cfg.instantiate(parent=None)  # type: ConformerLayer

        # Create a Repeat Conformer layer.
        num_layers = 5
        repeat_cfg = RepeatedConformerLayer.default_config().set(
            name="repeat_conformer",
            input_dim=dim,
            num_layers=num_layers,
        )
        repeat_cfg.layer.layer_order = layer_order
        repeat_cfg.layer.self_attention.attention.num_heads = num_heads
        if test_remat:
            repeat_cfg.layer.remat_spec = build_remat_spec(repeat_cfg)
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
        segment_ids = (jnp.arange(seq_len)[None, :] < num_tokens[:, None]).astype(jnp.int32)

        # disable dropout.
        is_training = False
        outputs = inputs
        for ll in range(num_layers):
            # Run a stack of layers by loop
            state_i = jax.tree.map(lambda param, i=ll: param[i], repeat_state)["repeat"]["layer"]
            outputs, _ = F(
                layer,
                inputs=dict(inputs=outputs, segment_ids=segment_ids),
                is_training=is_training,
                prng_key=jax.random.PRNGKey(200),
                state=state_i,
            )
        repeat_outs, _ = F(
            repeat_layer,
            inputs=dict(inputs=inputs, segment_ids=segment_ids),
            is_training=is_training,
            prng_key=jax.random.PRNGKey(200),
            state=repeat_state,
        )
        self.assertNestedAllClose(outputs, repeat_outs)

    @parameterized.parameters(
        (4, 10, None, None), (2, 12, 1, None), (6, 20, None, 3), (7, 30, 4, 6)
    )
    # pylint: disable-next=no-self-use
    def test_attention_sliding_window(self, batch_size, seq_len, left_context, right_context):
        lengths = jax.random.randint(
            jax.random.PRNGKey(3), shape=(batch_size,), minval=0, maxval=seq_len + 1
        )
        segment_ids = jnp.arange(seq_len)[None, :] < lengths[:, None]
        segment_ids = segment_ids.astype(jnp.int32)

        if right_context is not None:
            left_context = left_context or 0
            mask = LeftRightWindowAttentionBias.default_config(
                left_context=left_context, right_context=right_context
            )
        elif left_context is not None:
            mask = SlidingWindowAttentionBias.default_config(sliding_window_size=left_context)
        else:
            mask = CausalAttentionBias.default_config()

        dim, num_heads = 6, 2
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        cfg.self_attention.attention.mask = mask
        layer: ConformerLayer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=(batch_size, seq_len, dim))
        outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs, segment_ids=segment_ids),
            is_training=False,
            prng_key=jax.random.PRNGKey(2),
            state=state,
        )

        # Check outputs are zeroed for padding positions
        self.assertTrue(jnp.allclose(outputs * (segment_ids[:, :, None] == 0), 0))

        # Check attention bias pattern (independent of segment_ids/batch)
        target_positions = jnp.arange(seq_len)[None, :]
        attention_bias = mask.instantiate(
            target_positions=target_positions, source_positions=target_positions
        )
        logit_bias = attention_bias.value()

        # Expected pattern: single batch dimension since mask doesn't vary by batch
        expected = np.zeros((1, 1, seq_len, seq_len))
        if left_context is None:
            left_context = seq_len
        if right_context is None:
            right_context = 0
        for query in range(seq_len):
            for key in range(seq_len):
                if -left_context <= (key - query) <= right_context:
                    expected[0, 0, query, key] = 1
        assert_allclose(jnp.exp(logit_bias), expected, atol=1e-6, rtol=1e-6)

    def test_segment_ids(self):
        """Test that segment_ids properly mask different segments."""
        batch_size, seq_len, dim = 2, 10, 6
        num_heads = 2
        segment_ids = jnp.array([[1, 1, 1, 0, 2, 2, 2, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        layer: ConformerLayer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=(batch_size, seq_len, dim))
        outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs, segment_ids=segment_ids),
            is_training=False,
            prng_key=jax.random.PRNGKey(2),
            state=state,
        )

        # Outputs should be zero for positions where segment_id == 0
        mask = (segment_ids == 0)[:, :, None]
        self.assertTrue(jnp.allclose(outputs * mask, 0))

    @parameterized.product(
        batch_axis_names=("data", ("replica", "data", "fsdp")),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
        seq_axis_names=("seq",),
    )
    def test_set_double_shard_weights_config(
        self,
        batch_axis_names,
        fsdp_axis_names,
        tp_axis_names,
        seq_axis_names,
    ):
        dim, num_heads = 6, 2
        cfg = ConformerLayer.default_config().set(name="conformer", input_dim=dim)
        cfg.self_attention.attention.num_heads = num_heads
        set_double_shard_weights_config(
            cfg,
            batch_axis_names=batch_axis_names,
            fsdp_axis_names=fsdp_axis_names,
            tp_axis_names=tp_axis_names,
            seq_axis_names=seq_axis_names,
        )

        def check_ff(ff_layer):
            self.assertSequenceEqual(
                ff_layer.linear1.param_partition_spec, (fsdp_axis_names, tp_axis_names)
            )
            self.assertSequenceEqual(
                ff_layer.linear2.param_partition_spec, (tp_axis_names, fsdp_axis_names)
            )
            self.assertSequenceEqual(
                ff_layer.linear1.output_partition_spec,
                (batch_axis_names, seq_axis_names, tp_axis_names),
            )
            self.assertSequenceEqual(
                ff_layer.linear2.output_partition_spec,
                (batch_axis_names, seq_axis_names, tp_axis_names),
            )

        check_ff(cfg.ff_start)
        check_ff(cfg.ff_end)

        self_atten = cfg.self_attention.attention
        input_linear = self_atten.input_linear
        # Shard weights.
        self.assertSequenceEqual(
            input_linear.layer.param_partition_spec,
            (fsdp_axis_names, tp_axis_names, None),
        )
        self.assertSequenceEqual(
            self_atten.output_linear.param_partition_spec, (fsdp_axis_names, tp_axis_names, None)
        )


if __name__ == "__main__":
    absltest.main()
