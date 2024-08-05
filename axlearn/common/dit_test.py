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

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import parameterized
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import nn

from axlearn.common.attention import NEG_INF
from axlearn.common.dit import (
    AdaptiveLayerNormModulation,
    DiTAttentionLayer,
    DiTBlock,
    DiTFeedForwardLayer,
    DiTFinalLayer,
    LabelEmbedding,
    TimeStepEmbedding,
)
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def Refmodulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        x = x + gate.unsqueeze(1) * self.mlp(Refmodulate(self.norm2(x), shift, scale))
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
        x = x + gate.unsqueeze(1) * self.attn(ln_x)
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(Refmodulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(Refmodulate(self.norm2(x), shift_mlp, scale_mlp))
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
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = Refmodulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TestTimeStepEmbedding(parameterized.TestCase):
    """Tests TimeStepEmbedding."""

    @parameterized.parameters(256, 257)
    def test_time_step_embed(self, pos_embed_dim):
        batch_size = 5
        output_dim = 512

        timestep = np.random.randint(0, 10, size=batch_size)
        ref_model = RefTimestepEmbedder(
            hidden_size=output_dim, frequency_embedding_size=pos_embed_dim
        )
        ref_output = ref_model.forward(torch.as_tensor(timestep))

        axlearn_model_cfg = TimeStepEmbedding.default_config().set(
            name="test", pos_embed_dim=pos_embed_dim, output_dim=output_dim
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(ref_model, dst_layer=axlearn_model)
        layer_output, _ = F(
            axlearn_model,
            inputs=dict(positions=jnp.asarray(timestep)),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

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

    def test_adaln(self):
        batch_size = 2
        dim = 32
        num_outputs = 6
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


class TestDiTFFN(parameterized.TestCase):
    """Tests DiTFFN."""

    @parameterized.parameters(["prenorm", "postnorm", "hybridnorm"])
    def test_dit_ffn(self, structure: str):
        batch_size = 2
        seq_len = 12
        dim = 32
        inputs = np.random.random(size=(batch_size, seq_len, dim))
        shift = np.random.random(size=(batch_size, dim))
        scale = np.random.random(size=(batch_size, dim))
        gate = np.random.random(size=(batch_size, dim))

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
        shift = np.random.random(size=(batch_size, dim))
        scale = np.random.random(size=(batch_size, dim))
        gate = np.random.random(size=(batch_size, dim))

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
        shift = jax.random.normal(prng_key, shape=(batch_size, dim))
        scale = jax.random.normal(prng_key, shape=(batch_size, dim))
        gate = jax.random.normal(prng_key, shape=(batch_size, dim))
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


class TestDiTBlock(parameterized.TestCase):
    """Tests DiTBlock."""

    def test_dit_block(self):
        batch_size = 2
        seq_len = 12
        dim = 32
        num_heads = 2
        inputs = np.random.random(size=(batch_size, seq_len, dim))
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


class TestDiTFinalLayer(parameterized.TestCase):
    """Tests DiTFinalLayer."""

    def test_dit_attn(self):
        batch_size = 2
        seq_len = 12
        dim = 32
        patch_size = 4
        out_channels = 3
        inputs = np.random.random(size=(batch_size, seq_len, dim))
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
