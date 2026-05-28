# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/DiT:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC-BY-NC.

"""Tests DiT layers."""

# pylint: disable=no-self-use
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from axlearn.common.attention import CausalAttentionBias
from axlearn.common.attention_bias import NEG_INF
from axlearn.common.dit import (
    AdaptiveLayerNormModulation,
    DiTAttentionLayer,
    DiTBlock,
    DiTFeedForwardLayer,
    DiTFinalLayer,
    LabelEmbedding,
    TimeStepEmbedding,
)
from axlearn.common.golden import load_golden
from axlearn.common.kv_cache.sliding_window_kv_cache import enable_sliding_window_attention
from axlearn.common.layers import LayerNormStateless
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.vision_transformer import ConvertToSequence

MODULE_NAME = "axlearn.common.dit_test"


class TestTimeStepEmbedding(parameterized.TestCase):
    """Tests TimeStepEmbedding."""

    @parameterized.product(
        pos_embed_dim=[256, 257],
        norm_cls=[None, LayerNormStateless],
    )
    def test_time_step_embed(self, pos_embed_dim, norm_cls):
        output_dim = 512
        timestep = jnp.linspace(0, 1, 5)

        output_norm_cfg = norm_cls.default_config() if norm_cls else None
        axlearn_model_cfg = TimeStepEmbedding.default_config().set(
            name="test",
            pos_embed_dim=pos_embed_dim,
            output_dim=output_dim,
            output_norm=output_norm_cfg,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        if norm_cls is None:
            golden = load_golden(MODULE_NAME, f"test_time_step_embed_dim{pos_embed_dim}")
            layer_params = golden["params"]
            layer_output, _ = F(
                axlearn_model,
                inputs=dict(positions=timestep),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            assert_allclose(layer_output, golden["outputs"]["ref"])

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_positional_embedding_always_float32(self, dtype):
        """Test that positional embeddings are always computed in float32."""
        pos_embed_dim = 256
        output_dim = 512
        timestep = jnp.linspace(0, 1, 5, dtype=dtype)

        model_cfg = TimeStepEmbedding.default_config().set(
            name=f"test_{dtype}",
            pos_embed_dim=pos_embed_dim,
            output_dim=output_dim,
            dtype=dtype,
        )
        model = model_cfg.instantiate(parent=None)

        # Verify positional embeddings are always computed in float32,
        # regardless of model dtype or input dtype
        pos_emb = model.dit_sinusoidal_positional_embeddings(timestep)
        self.assertEqual(
            pos_emb.dtype,
            jnp.float32,
            f"Expected pos_emb to be float32 when dtype={dtype}, got {pos_emb.dtype}",
        )

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_embed_proj_input_dtype_matches_param_dtype(self, dtype):
        """Test that pos_emb is cast to match parameter dtype before projection."""
        pos_embed_dim = 256
        output_dim = 512
        timestep = jnp.linspace(0, 1, 5, dtype=dtype)

        model_cfg = TimeStepEmbedding.default_config().set(
            name=f"test_{dtype}",
            pos_embed_dim=pos_embed_dim,
            output_dim=output_dim,
            dtype=dtype,
        )
        model = model_cfg.instantiate(parent=None)
        params = model.initialize_parameters_recursively(jax.random.PRNGKey(123))

        # Run forward pass
        output, _ = F(
            model,
            inputs=dict(positions=timestep),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(output.dtype, dtype)


class TestLabelEmbedding(parameterized.TestCase):
    """Tests LabelEmbedding."""

    @parameterized.parameters(True, False)
    def test_label_embed(self, is_training):
        batch_size = 5
        class_dropout_prob = 0.5
        num_classes = 1000
        hidden_size = 1152

        golden = load_golden(MODULE_NAME, "test_label_embed")

        axlearn_model_cfg = LabelEmbedding.default_config().set(
            name="test",
            num_classes=num_classes,
            output_dim=hidden_size,
            dropout_rate=class_dropout_prob,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        layer_params = golden["params"]
        labels = jnp.array(golden["inputs"]["labels"])

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(labels=labels),
            state=layer_params,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(0),
        )
        if is_training:
            self.assertEqual(layer_output.shape, (batch_size, hidden_size))
        else:
            assert_allclose(layer_output, golden["outputs"]["ref"])


class TestAdaptiveLayerNormModulation(parameterized.TestCase):
    """Tests AdaptiveLayerNormModulation."""

    @parameterized.parameters(0, 3)
    def test_adaln(self, seq_len):
        dim = 32
        num_outputs = 6

        golden = load_golden(MODULE_NAME, f"test_adaln_seq{seq_len}")
        inputs = jnp.array(golden["inputs"]["input"])

        axlearn_model_cfg = AdaptiveLayerNormModulation.default_config().set(
            name="test",
            dim=dim,
            num_outputs=num_outputs,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        layer_params = golden["params"]

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(input=inputs),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        ref_outputs = tuple(jnp.array(golden["outputs"][f"out_{i}"]) for i in range(num_outputs))
        assert_allclose(layer_output, ref_outputs)

    @parameterized.parameters([(2, 8), (2, 1, 8)], [(2, 3, 8), (2, 3, 8)])
    def test_shape(self, shape, expected_shape):
        dim = shape[-1]
        num_outputs = 6

        prng_key = jax.random.PRNGKey(123)
        prng_key, data_key = jax.random.split(prng_key)
        inputs = jax.random.normal(data_key, shape=shape)

        cfg = AdaptiveLayerNormModulation.default_config().set(
            name="test",
            dim=dim,
            num_outputs=num_outputs,
        )
        layer = cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        layer_output, _ = F(
            layer,
            inputs=dict(input=inputs),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(len(layer_output), num_outputs)
        for i in range(num_outputs):
            self.assertEqual(layer_output[i].shape, expected_shape)


class TestDiTFFN(parameterized.TestCase):
    """Tests DiTFFN."""

    @parameterized.parameters(["prenorm", "postnorm", "hybridnorm"])
    def test_dit_ffn(self, structure: str):
        batch_size = 2
        seq_len = 12
        dim = 32

        axlearn_model_cfg = DiTFeedForwardLayer.default_config().set(
            name="test",
            input_dim=dim,
            structure=structure,
        )
        axlearn_model_cfg.norm.eps = 1e-6
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        if structure == "prenorm":
            golden = load_golden(MODULE_NAME, "test_dit_ffn_prenorm")
            layer_params = golden["params"]
            inputs = jnp.array(golden["inputs"]["input"])
            shift = jnp.array(golden["inputs"]["shift"])
            scale = jnp.array(golden["inputs"]["scale"])
            gate = jnp.array(golden["inputs"]["gate"])

            layer_output, _ = F(
                axlearn_model,
                inputs=dict(input=inputs, shift=shift, scale=scale, gate=gate),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            assert_allclose(layer_output, golden["outputs"]["ref"])
        else:
            # For non-prenorm structures, just test that the layer runs without error.
            inputs = jnp.array(np.random.random(size=(batch_size, seq_len, dim)))
            shift = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            scale = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            gate = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            layer_params = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
            layer_output, _ = F(
                axlearn_model,
                inputs=dict(input=inputs, shift=shift, scale=scale, gate=gate),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            self.assertEqual(layer_output.shape, inputs.shape)


class TestDiTAttn(parameterized.TestCase):
    """Tests DiTAttn."""

    @parameterized.parameters(["prenorm", "postnorm", "hybridnorm"])
    def test_dit_attn(self, structure: str):
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2

        axlearn_model_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
            structure=structure,
        )
        axlearn_model_cfg.attention.num_heads = num_heads
        axlearn_model_cfg.norm.eps = 1e-6
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        if structure == "prenorm":
            golden = load_golden(MODULE_NAME, "test_dit_attn_prenorm")
            layer_params = golden["params"]
            inputs = jnp.array(golden["inputs"]["input"])
            shift = jnp.array(golden["inputs"]["shift"])
            scale = jnp.array(golden["inputs"]["scale"])
            gate = jnp.array(golden["inputs"]["gate"])

            layer_output, _ = F(
                axlearn_model,
                inputs=dict(input=inputs, shift=shift, scale=scale, gate=gate),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            assert_allclose(layer_output, golden["outputs"]["ref"])
        else:
            # For non-prenorm structures, just test that the layer runs without error.
            inputs = jnp.array(np.random.random(size=(batch_size, seq_len, dim)))
            shift = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            scale = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            gate = jnp.array(np.random.random(size=(batch_size, 1, dim)))
            layer_params = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
            layer_output, _ = F(
                axlearn_model,
                inputs=dict(input=inputs, shift=shift, scale=scale, gate=gate),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            self.assertEqual(layer_output.shape, inputs.shape)

    def test_dit_attn_logit_biases(self):
        seq_len = 3
        batch_size = 2
        dim = 32
        num_heads = 2

        prng_key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(prng_key, shape=(batch_size, seq_len, dim))
        shift = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        scale = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        gate = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        valid_seq = jnp.asarray(
            [
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=jnp.bool,
        )
        valid_mask = valid_seq[:, :, None]
        logit_biases = NEG_INF * (1 - jnp.einsum("bi,bj->bij", valid_seq, valid_seq))

        layer_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
        )
        layer_cfg.attention.num_heads = num_heads
        layer_cfg.norm.eps = 1e-6
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=prng_key)

        layer_output, _ = F(
            layer,
            inputs=dict(
                input=inputs,
                shift=shift,
                scale=scale,
                gate=gate,
                attention_logit_biases=logit_biases,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )

        # Change inputs on non-valid seq.
        inputs2 = jax.random.normal(jax.random.PRNGKey(1), shape=(batch_size, seq_len, dim))
        modified_inputs = inputs * valid_mask + inputs2 * (1 - valid_mask)
        layer_output2, _ = F(
            layer,
            inputs=dict(
                input=modified_inputs,
                shift=shift,
                scale=scale,
                gate=gate,
                attention_logit_biases=logit_biases,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )
        # Expect the output be the same for valid items because of logit_biases.
        assert_allclose(layer_output * valid_mask, layer_output2 * valid_mask)

    def test_dit_attn_segment_ids(self):
        batch_size = 2
        seq_len = 3
        dim = 32
        num_heads = 2

        prng_key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(prng_key, shape=(batch_size, seq_len, dim))
        shift = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        scale = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        gate = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        segment_ids = jnp.ones((batch_size, seq_len))

        layer_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
        )
        layer_cfg.attention.num_heads = num_heads
        layer_cfg.norm.eps = 1e-6
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=prng_key)

        layer_output, _ = F(
            layer,
            inputs=dict(
                input=inputs,
                shift=shift,
                scale=scale,
                gate=gate,
                segment_ids=segment_ids,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )
        assert_allclose(layer_output.shape, inputs.shape)

    @parameterized.parameters([True, False])
    def test_dit_attn_optional_input(self, use_ssg):
        batch_size = 2
        seq_len = 3
        dim = 32
        num_heads = 2

        prng_key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(prng_key, shape=(batch_size, seq_len, dim))

        if use_ssg:
            shift = scale = gate = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        else:
            shift = scale = gate = None

        layer_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
        )
        layer_cfg.attention.num_heads = num_heads
        layer_cfg.norm.eps = 1e-6
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=prng_key)

        layer_output, _ = F(
            layer,
            inputs=dict(
                input=inputs,
                shift=shift,
                scale=scale,
                gate=gate,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )
        assert_allclose(layer_output.shape, inputs.shape)

    def test_dit_attn_optional_input_value_error(self):
        batch_size = 2
        seq_len = 3
        dim = 32
        num_heads = 2

        prng_key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(prng_key, shape=(batch_size, seq_len, dim))
        shift = jax.random.normal(prng_key, shape=(batch_size, 1, dim))
        scale = gate = None

        layer_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
        )
        layer_cfg.attention.num_heads = num_heads
        layer_cfg.norm.eps = 1e-6
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=prng_key)

        with pytest.raises(
            ValueError, match=re.escape("shift and scale must be both provided or both None.")
        ):
            layer_output, _ = F(
                layer,
                inputs=dict(
                    input=inputs,
                    shift=shift,
                    scale=scale,
                    gate=gate,
                ),
                state=state,
                is_training=False,
                prng_key=prng_key,
            )
            assert_allclose(layer_output.shape, inputs.shape)

    @parameterized.parameters("causal", "sliding_window")
    def test_dit_attn_extend_step(self, causal_type):
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2
        prng_key = jax.random.PRNGKey(123)
        prng_key, data_key = jax.random.split(prng_key)
        inputs = jax.random.normal(data_key, shape=(batch_size, seq_len, dim))
        shift = jax.random.normal(data_key, shape=(batch_size, 1, dim))
        scale = jax.random.normal(data_key, shape=(batch_size, 1, dim))
        gate = jax.random.normal(data_key, shape=(batch_size, 1, dim))

        layer_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
        )
        layer_cfg.attention.num_heads = num_heads
        if causal_type == "causal":
            layer_cfg.attention.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            layer_cfg.attention = enable_sliding_window_attention(
                layer_cfg.attention, sliding_window_size=10
            )

        layer = layer_cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        fwd_output, _ = F(
            layer,
            inputs=dict(
                input=inputs,
                shift=shift,
                scale=scale,
                gate=gate,
            ),
            state=layer_params,
            is_training=False,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(
            batch_size=batch_size, max_len=seq_len, dtype=inputs.dtype
        )
        step_sizes = (1, 2, 3)
        step_outputs = []
        i = 0
        while i < seq_len:
            step_size = step_sizes[i % len(step_sizes)]
            step_inputs = dict(
                cached_states=cached_states,
                target=inputs[:, i : i + step_size],
                shift=shift,
                scale=scale,
                gate=gate,
            )
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                inputs=step_inputs,
                state=layer_params,
                is_training=False,
                prng_key=prng_key,
                method="extend_step",
            )
            step_outputs.append(step_output)
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)


class TestDiTBlock(parameterized.TestCase):
    """Tests DiTBlock."""

    @parameterized.parameters(False, True)
    def test_dit_block(self, seq_cond):
        dim = 32
        num_heads = 2

        golden = load_golden(MODULE_NAME, f"test_dit_block_seq_cond_{seq_cond}")

        axlearn_model_cfg = DiTBlock.default_config().set(name="test", input_dim=dim)
        axlearn_model_cfg.attention.attention.num_heads = num_heads
        axlearn_model_cfg.attention.norm.eps = 1e-6
        axlearn_model_cfg.feed_forward.norm.eps = 1e-6
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = golden["params"]
        inputs = jnp.array(golden["inputs"]["input"])
        condition = jnp.array(golden["inputs"]["condition"])

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(input=inputs, condition=condition),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output, golden["outputs"]["ref"])

    def test_dit_block_segment_ids(self):
        """Verifies `target_segment_ids` makes self-attention segment-aware.

        Two segments packed in one slot (`[1,1,1,2,2,2]`) must produce the same
        outputs on segment 2's positions as running segment 2 alone — proving the
        attention mask blocks cross-segment attention.
        """
        batch_size = 1
        seg_len = 3
        dim = 32
        num_heads = 2
        prng_key = jax.random.PRNGKey(0)

        layer_cfg = DiTBlock.default_config().set(name="test", input_dim=dim)
        layer_cfg.attention.attention.num_heads = num_heads
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=prng_key)

        prng_key, key1, key2 = jax.random.split(prng_key, 3)
        seg1 = jax.random.normal(key1, shape=(batch_size, seg_len, dim))
        seg2 = jax.random.normal(key2, shape=(batch_size, seg_len, dim))
        condition = jax.random.normal(prng_key, shape=(batch_size, dim))

        # Packed: two segments concatenated, segment_ids marks the boundary.
        packed_input = jnp.concatenate([seg1, seg2], axis=1)
        packed_segment_ids = jnp.array(
            [[1] * seg_len + [2] * seg_len] * batch_size, dtype=jnp.int32
        )
        packed_out, _ = F(
            layer,
            inputs=dict(
                input=packed_input,
                condition=condition,
                segment_ids=packed_segment_ids,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )

        # Reference: run seg2 alone — segment-aware attention should yield identical
        # outputs on positions 3-5 since seg2 cannot see seg1.
        seg2_segment_ids = jnp.ones((batch_size, seg_len), dtype=jnp.int32)
        seg2_out, _ = F(
            layer,
            inputs=dict(
                input=seg2,
                condition=condition,
                segment_ids=seg2_segment_ids,
            ),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )

        assert_allclose(packed_out[:, seg_len:], seg2_out)

        # Sanity: without segment_ids, packed seg2 positions DO see seg1 → output differs.
        packed_unsegmented_out, _ = F(
            layer,
            inputs=dict(input=packed_input, condition=condition),
            state=state,
            is_training=False,
            prng_key=prng_key,
        )
        self.assertFalse(jnp.allclose(packed_unsegmented_out[:, seg_len:], seg2_out))

    @parameterized.product(
        causal_type=["causal", "sliding_window"],
        seq_cond=[False, True],
    )
    def test_dit_block_extend_step(self, causal_type, seq_cond):
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2
        prng_key = jax.random.PRNGKey(123)
        prng_key, data_key = jax.random.split(prng_key)
        inputs = jax.random.normal(data_key, shape=(batch_size, seq_len, dim))
        if seq_cond:
            condition = jax.random.normal(data_key, shape=(batch_size, seq_len, dim))
        else:
            condition = jax.random.normal(data_key, shape=(batch_size, dim))

        layer_cfg = DiTBlock.default_config().set(name="test", input_dim=dim)
        layer_cfg.attention.attention.num_heads = num_heads
        if causal_type == "causal":
            layer_cfg.attention.attention.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            layer_cfg.attention.attention = enable_sliding_window_attention(
                layer_cfg.attention.attention, sliding_window_size=10
            )

        layer = layer_cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        fwd_output, _ = F(
            layer,
            inputs=dict(input=inputs, condition=condition),
            state=layer_params,
            is_training=False,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(
            batch_size=batch_size, max_len=seq_len, dtype=inputs.dtype
        )
        step_sizes = (1, 2, 3)
        step_outputs = []
        i = 0
        while i < seq_len:
            step_size = step_sizes[i % len(step_sizes)]
            step_inputs = dict(
                cached_states=cached_states,
                target=inputs[:, i : i + step_size],
                condition=condition[:, i : i + step_size] if seq_cond else condition,
            )
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                inputs=step_inputs,
                state=layer_params,
                is_training=False,
                prng_key=prng_key,
                method="extend_step",
            )
            step_outputs.append(step_output)
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)


class TestDiTFinalLayer(parameterized.TestCase):
    """Tests DiTFinalLayer."""

    @parameterized.parameters(False, True)
    def test_dit_final_layer(self, seq_cond):
        dim = 32
        patch_size = 4
        out_channels = 3

        golden = load_golden(MODULE_NAME, f"test_dit_final_layer_seq_cond_{seq_cond}")

        axlearn_model_cfg = DiTFinalLayer.default_config().set(
            name="test", input_dim=dim, output_dim=patch_size * patch_size * out_channels
        )
        axlearn_model_cfg.norm.eps = 1e-6
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = golden["params"]
        inputs = jnp.array(golden["inputs"]["input"])
        condition = jnp.array(golden["inputs"]["condition"])

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(input=inputs, condition=condition),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output, golden["outputs"]["ref"])


class TestDiTPatchEmbed(parameterized.TestCase):
    """Tests DiTPatchEmbed."""

    def test_dit_patch_embed(self):
        patch_size = 4
        img_channels = 3
        dim = 32

        golden = load_golden(MODULE_NAME, "test_dit_patch_embed")

        axlearn_model_cfg = ConvertToSequence.default_config().set(
            name="test",
            patch_size=(patch_size, patch_size),
            input_dim=img_channels,
            output_dim=dim,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = golden["params"]
        inputs = jnp.array(golden["inputs"]["x"])

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(x=inputs),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output, golden["outputs"]["ref"])


if __name__ == "__main__":
    absltest.main()
