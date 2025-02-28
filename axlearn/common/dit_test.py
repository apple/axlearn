# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/DiT:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC-BY-NC.

"""Tests DiT layers."""
# pylint: disable=no-self-use
import math
import re

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from absl.testing import absltest, parameterized
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import nn

from axlearn.common.attention_bias import NEG_INF, CausalAttentionBias, SlidingWindowAttentionBias
from axlearn.common.dit import (
    AdaptiveLayerNormModulation,
    DiTAttentionLayer,
    DiTBlock,
    DiTFeedForwardLayer,
    DiTFinalLayer,
    LabelEmbedding,
    TimeStepEmbedding,
)
from axlearn.common.layers import LayerNormStateless
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import TensorSpec, as_tensor
from axlearn.common.vision_transformer import ConvertToSequence


class RefTimestepEmbedder(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L27-L64
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class RefLabelEmbedder(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L67-L94
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class RefAdaLNModulate(nn.Module):
    """https://github.com/facebookresearch/DiT/blob/master/models.py#L113-L115"""

    def __init__(self, hidden_size):
        super().__init__()
        # pylint: disable-next=invalid-name
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, c):
        # pylint: disable-next=invalid-name
        output = self.adaLN_modulation(c)
        if output.ndim == 2:
            output = einops.rearrange(output, "b d -> b 1 d")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = output.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def Refmodulate(x, shift, scale):
    return x * (1 + scale) + shift


class RefDiTMLP(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L110-L112
    MLP layer of DiT block.
    """

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, shift, scale, gate):
        x = x + gate * self.mlp(Refmodulate(self.norm2(x), shift, scale))
        return x


class RefDiTAttn(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L107:L108
    Attention layer of DiT block.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

    def forward(self, x, shift, scale, gate):
        ln_x = Refmodulate(self.norm1(x), shift, scale)
        x = x + gate * self.attn(ln_x)
        return x


class RefDiTBlock(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L101-L122
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        # pylint: disable-next=invalid-name
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # pylint: disable-next=invalid-name
        output = self.adaLN_modulation(c)
        if output.ndim == 2:
            output = einops.rearrange(output, "b d -> b 1 d")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = output.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(Refmodulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(Refmodulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class RefFinalLayer(nn.Module):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L125-L142
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # pylint: disable-next=invalid-name
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # pylint: disable-next=invalid-name
        output = self.adaLN_modulation(c)
        if output.ndim == 2:
            output = einops.rearrange(output, "b d -> b 1 d")
        shift, scale = output.chunk(2, dim=-1)
        x = Refmodulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TestTimeStepEmbedding(parameterized.TestCase):
    """Tests TimeStepEmbedding."""

    @parameterized.product(
        pos_embed_dim=[256, 257],
        norm_cls=[None, LayerNormStateless],
    )
    def test_time_step_embed(self, pos_embed_dim, norm_cls):
        batch_size = 5
        output_dim = 512

        timestep = np.random.randint(0, 10, size=batch_size)

        output_norm_cfg = norm_cls.default_config() if norm_cls else None
        axlearn_model_cfg = TimeStepEmbedding.default_config().set(
            name="test",
            pos_embed_dim=pos_embed_dim,
            output_dim=output_dim,
            output_norm=output_norm_cfg,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        ref_model = RefTimestepEmbedder(
            hidden_size=output_dim, frequency_embedding_size=pos_embed_dim
        )
        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(positions=jnp.asarray(timestep)),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        if norm_cls is None:
            ref_output = ref_model.forward(torch.as_tensor(timestep))
            assert_allclose(layer_output, as_tensor(ref_output))


class TestLabelEmbedding(parameterized.TestCase):
    """Tests LabelEmbedding."""

    @parameterized.parameters(True, False)
    def test_label_embed(self, is_training):
        # The hyperparam is set to be the same as default setting.
        # Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L151-L160
        batch_size = 5
        class_dropout_prob = 0.5
        num_classes = 1000
        hidden_size = 1152

        label = np.random.randint(0, num_classes, size=batch_size)
        ref_model = RefLabelEmbedder(
            num_classes=num_classes, hidden_size=hidden_size, dropout_prob=class_dropout_prob
        )
        ref_output = ref_model.forward(torch.as_tensor(label), train=is_training)

        axlearn_model_cfg = LabelEmbedding.default_config().set(
            name="test",
            num_classes=num_classes,
            output_dim=hidden_size,
            dropout_rate=class_dropout_prob,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(labels=jnp.asarray(label)),
            state=layer_params,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(0),
        )
        if is_training:
            assert_allclose(layer_output.shape, as_tensor(ref_output).shape)
        else:
            assert_allclose(layer_output, as_tensor(ref_output))


class TestAdaptiveLayerNormModulation(parameterized.TestCase):
    """Tests AdaptiveLayerNormModulation."""

    @parameterized.parameters(0, 3)
    def test_adaln(self, seq_len):
        batch_size = 2
        dim = 32
        num_outputs = 6
        if seq_len:
            inputs = np.random.random(size=(batch_size, seq_len, dim))
        else:
            inputs = np.random.random(size=(batch_size, dim))
        ref_model = RefAdaLNModulate(hidden_size=dim)
        ref_output = ref_model.forward(torch.as_tensor(inputs).float())

        axlearn_model_cfg = AdaptiveLayerNormModulation.default_config().set(
            name="test",
            dim=dim,
            num_outputs=num_outputs,
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(input=jnp.asarray(inputs)),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_output, as_tensor(ref_output))

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
        inputs = np.random.random(size=(batch_size, seq_len, dim))
        shift = np.random.random(size=(batch_size, 1, dim))
        scale = np.random.random(size=(batch_size, 1, dim))
        gate = np.random.random(size=(batch_size, 1, dim))

        axlearn_model_cfg = DiTFeedForwardLayer.default_config().set(
            name="test",
            input_dim=dim,
            structure=structure,
        )
        axlearn_model_cfg.norm.eps = 1e-6
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        ref_model = RefDiTMLP(hidden_size=dim)
        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(
                input=jnp.asarray(inputs),
                shift=jnp.asarray(shift),
                scale=jnp.asarray(scale),
                gate=jnp.asarray(gate),
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        if structure == "prenorm":
            ref_output = ref_model.forward(
                torch.as_tensor(inputs).float(),
                torch.as_tensor(shift).float(),
                torch.as_tensor(scale).float(),
                torch.as_tensor(gate).float(),
            )
            assert_allclose(layer_output, as_tensor(ref_output))


class TestDiTAttn(parameterized.TestCase):
    """Tests DiTAttn."""

    @parameterized.parameters(["prenorm", "postnorm", "hybridnorm"])
    def test_dit_attn(self, structure: str):
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2
        inputs = np.random.random(size=(batch_size, seq_len, dim))
        shift = np.random.random(size=(batch_size, 1, dim))
        scale = np.random.random(size=(batch_size, 1, dim))
        gate = np.random.random(size=(batch_size, 1, dim))

        axlearn_model_cfg = DiTAttentionLayer.default_config().set(
            name="test",
            source_dim=dim,
            target_dim=dim,
            structure=structure,
        )
        axlearn_model_cfg.attention.num_heads = num_heads
        axlearn_model_cfg.norm.eps = 1e-6

        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        ref_model = RefDiTAttn(hidden_size=dim, num_heads=num_heads)
        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(
                input=jnp.asarray(inputs),
                shift=jnp.asarray(shift),
                scale=jnp.asarray(scale),
                gate=jnp.asarray(gate),
                attention_logit_biases=None,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        if structure == "prenorm":
            ref_output = ref_model.forward(
                torch.as_tensor(inputs).float(),
                torch.as_tensor(shift).float(),
                torch.as_tensor(scale).float(),
                torch.as_tensor(gate).float(),
            )
            assert_allclose(layer_output, as_tensor(ref_output))

    def test_dit_attn_logit_biases(self):
        batch_size = 2
        seq_len = 3
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
            layer_cfg.attention.mask = SlidingWindowAttentionBias.default_config(
                sliding_window_size=10
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
                attention_logit_biases=None,
            ),
            state=layer_params,
            is_training=False,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(input_spec=TensorSpec(inputs.shape, inputs.dtype))
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
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2
        inputs = np.random.random(size=(batch_size, seq_len, dim))
        if seq_cond:
            condition = np.random.random(size=(batch_size, seq_len, dim))
        else:
            condition = np.random.random(size=(batch_size, dim))
        ref_model = RefDiTBlock(hidden_size=dim, num_heads=num_heads)
        ref_output = ref_model.forward(
            torch.as_tensor(inputs).float(),
            torch.as_tensor(condition).float(),
        )

        axlearn_model_cfg = DiTBlock.default_config().set(name="test", input_dim=dim)
        axlearn_model_cfg.attention.attention.num_heads = num_heads
        axlearn_model_cfg.attention.norm.eps = 1e-6
        axlearn_model_cfg.feed_forward.norm.eps = 1e-6

        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(
                input=jnp.asarray(inputs),
                condition=jnp.asarray(condition),
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_output, as_tensor(ref_output))

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
            layer_cfg.attention.attention.mask = SlidingWindowAttentionBias.default_config(
                sliding_window_size=10
            )

        layer = layer_cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        fwd_output, _ = F(
            layer,
            inputs=dict(
                input=inputs,
                condition=condition,
            ),
            state=layer_params,
            is_training=False,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(input_spec=TensorSpec(inputs.shape, inputs.dtype))
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
    def test_dit_attn(self, seq_cond):
        batch_size = 2
        seq_len = 12
        dim = 32
        patch_size = 4
        out_channels = 3
        inputs = np.random.random(size=(batch_size, seq_len, dim))
        if seq_cond:
            input_condition = np.random.random(size=(batch_size, seq_len, dim))
        else:
            input_condition = np.random.random(size=(batch_size, dim))
        ref_model = RefFinalLayer(hidden_size=dim, patch_size=patch_size, out_channels=out_channels)
        ref_output = ref_model.forward(
            torch.as_tensor(inputs).float(),
            torch.as_tensor(input_condition).float(),
        )

        axlearn_model_cfg = DiTFinalLayer.default_config().set(
            name="test", input_dim=dim, output_dim=patch_size * patch_size * out_channels
        )
        axlearn_model_cfg.norm.eps = 1e-6

        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(
                input=jnp.asarray(inputs),
                condition=jnp.asarray(input_condition),
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_output, as_tensor(ref_output))


class TestDiTPatchEmbed(parameterized.TestCase):
    """Tests DiTPatchEmbed."""

    def test_dit_patch_embed(self):
        batch_size = 2
        img_size = 16
        patch_size = 4
        img_channels = 3
        dim = 32
        inputs = np.random.random(size=(batch_size, img_size, img_size, img_channels))
        ref_model = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim)
        ref_output = ref_model.forward(
            torch.as_tensor(np.transpose(inputs, [0, 3, 1, 2])).float(),
        )

        axlearn_model_cfg = ConvertToSequence.default_config().set(
            name="test",
            patch_size=(patch_size, patch_size),
            input_dim=img_channels,
            output_dim=dim,
        )

        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)

        layer_output, _ = F(
            axlearn_model,
            inputs=dict(
                x=jnp.asarray(inputs),
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_output, as_tensor(ref_output))


if __name__ == "__main__":
    absltest.main()
