# Copyright © 2023 Apple Inc.

"""Tests PyTorch adapter layers."""

# pylint: disable=too-many-lines
import itertools
from collections import OrderedDict
from typing import Optional, Union, cast

import jax
import numpy as np
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import attention_bias
from axlearn.common.adapter_torch import (
    NEG_INF,
    AdapterCausalLmModelBuilder,
    ALiBiAttentionLogitBiasLayer,
    AttentionPooling,
    AveragePooling,
    BottleNeckAdapterTransformerLayer,
    CausalAttentionLogitBiasLayer,
    CausalLmModelBuilder,
    CoCaImageStreamEncoderBuilder,
    Decoder,
    Embedding,
    FirstNTokenPooling,
    FusedQKVLinear,
    Linear,
    MultiheadAttention,
    QKVLinear,
    RMSNorm,
    SampleDecodingSession,
    ScaleBy,
    StackedTransformerLayer,
    TopK,
    TopP,
    TransformerAttentionLayer,
    TransformerEmbeddings,
    TransformerFeedForwardLayer,
    TransformerLayer,
    ViTModelBuilder,
    _segment_ids_from_causal_input_ids,
    _torch_activation_fn,
    alibi_get_slopes,
)
from axlearn.common.attention import (
    ALiBiAttentionLogitBiasLayer as AxlearnALiBiAttentionLogitBiasLayer,
)
from axlearn.common.attention import (
    BottleNeckAdapterTransformerLayer as AxlearnBottleNeckAdapterTransformerLayer,
)
from axlearn.common.attention import (
    CausalAttentionLogitBiasLayer as AxlearnCausalAttentionLogitBiasLayer,
)
from axlearn.common.attention import FusedQKVLinear as AxlearnFusedQKVLinear
from axlearn.common.attention import LearnedPositionalEmbedding as AxlearnLearnedPositionalEmbedding
from axlearn.common.attention import QKVLinear as AxlearnQKVLinear
from axlearn.common.attention import StackedTransformerLayer as AxlearnStackedTransformerLayer
from axlearn.common.attention import TransformerAttentionLayer as AxlearnTransformerAttentionLayer
from axlearn.common.attention import (
    TransformerFeedForwardLayer as AxlearnTransformerFeedForwardLayer,
)
from axlearn.common.attention import TransformerLayer as AxlearnTransformerLayer
from axlearn.common.attention import alibi_get_slopes as axlearn_alibi_get_slopes
from axlearn.common.attention import scaled_hidden_dim as axlearn_scaled_hidden_dim
from axlearn.common.causal_lm import Model as AxlearnCausalLmModel
from axlearn.common.config import InstantiableConfig
from axlearn.common.decoder import Decoder as AxlearnDecoder
from axlearn.common.embedding import TransformerTextEmbeddings as AxlearnTransformerEmbeddings
from axlearn.common.layers import Embedding as AxlearnEmbedding
from axlearn.common.layers import LayerNorm as AxlearnLayerNorm
from axlearn.common.layers import Linear as AxlearnLinear
from axlearn.common.layers import RMSNorm as AxlearnRMSNorm
from axlearn.common.layers import set_bias_recursively, set_norm_recursively
from axlearn.common.logit_modifiers import top_k_logits as axlearn_top_k_logits
from axlearn.common.module import NestedTensor
from axlearn.common.module import functional as F
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, WeightInitializer
from axlearn.common.poolings import AttentionPooling as AxlearnAttentionPooling
from axlearn.common.poolings import AveragePooling as AxlearnAveragePooling
from axlearn.common.poolings import FirstNTokenPooling as AxlearnFirstNTokenPooling
from axlearn.common.test_utils import TestCase, set_threefry_partitionable
from axlearn.common.utils import flatten_items
from axlearn.common.vision_transformer import named_model_configs as axlearn_vit_configs
from axlearn.vision.coca import set_coca_vision_encoder_config
from axlearn.vision.image_classification import ImageClassificationModel


def create_axlearn_state_dict(axlearn_state: NestedTensor) -> OrderedDict[str, torch.Tensor]:
    # Flattens Axlearn state, with '.' separators and weights as torch Tensors.
    # Note: Python3 dicts preserve insertion order by default.
    state = {
        el[0]: torch.as_tensor(np.array(el[1]))
        for el in flatten_items(axlearn_state, separator=".")  # pytype: disable=not-callable
    }
    return cast(OrderedDict, state)


class PreventParamBroadcastTest(TestCase):
    def test_broadcast_blocked(self):
        def load_broadcastable_params():
            linear_torch_layer = Linear(in_features=6, out_features=10)
            linear_axlearn_layer = (
                AxlearnLinear.default_config()
                .set(input_dim=1, output_dim=1, name="axlearn")
                .instantiate(parent=None)
            )
            linear_axlearn_layer_state = linear_axlearn_layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(0)
            )

            linear_torch_layer.load_axlearn_state_dict(
                create_axlearn_state_dict(linear_axlearn_layer_state)
            )

        self.assertRaises(ValueError, load_broadcastable_params)


class TorchDefaultModulesTest(TestCase):
    def test_embedding_module(self):
        dim = 3
        num_embeddings = 7

        torch_layer = Embedding(num_embeddings=num_embeddings, embedding_dim=dim)
        axlearn_layer = (
            AxlearnLearnedPositionalEmbedding.default_config()
            .set(dim=dim, shape=(num_embeddings,), name="test")
            .instantiate(parent=None)
        )
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        axlearn_outputs = np.array(
            F(
                axlearn_layer,
                jax.random.PRNGKey(0),
                state=axlearn_layer_state,
                inputs=(),
                is_training=False,
                method="embeddings",
            )[0]
        )
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        torch_outputs = torch_layer.weight.detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs)


class NormalizationModulesTest(TestCase):
    @parameterized.parameters(1, 5, 128)
    def test_rmsnorm_layer(self, input_dim: int):
        torch_layer = RMSNorm(input_dim)
        axlearn_layer = (
            AxlearnRMSNorm.default_config()
            .set(name="test", input_dim=input_dim)
            .instantiate(parent=None)
        )
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        rng = np.random.RandomState(123)
        inputs = rng.randn(2, 3 * input_dim, input_dim).astype(np.float32)
        torch_inputs = torch.as_tensor(inputs)
        axlearn_inputs = jnp.array(inputs)
        axlearn_outputs = np.array(
            F(
                axlearn_layer,
                jax.random.PRNGKey(0),
                state=axlearn_layer_state,
                inputs=(axlearn_inputs,),
                is_training=False,
                method="forward",
            )[0]
        )
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        torch_outputs = torch_layer.eval()(torch_inputs).detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs)


class AttentionModulesTest(TestCase):
    STRUCTURES = ("postnorm", "prenorm")
    NORMS = ("layernorm", "rmsnorm")

    @parameterized.parameters(
        itertools.product(
            STRUCTURES,
            ("nn.gelu", "nn.silu", ("nn.gelu", "nn.silu"), ("nn.silu", "linear")),
            (True, False),
            NORMS,
        )
    )
    def test_transformer_feed_forward_layer(
        self,
        structure: str,
        activation: Union[str, tuple[str, str]],
        linear_biases: bool,
        norm: str,
    ):
        input_dim = 4
        hidden_dim = 16

        if isinstance(activation, tuple):
            torch_activation = tuple(_torch_activation_fn(el) for el in activation)
        else:
            torch_activation = _torch_activation_fn(activation)

        torch_layer = TransformerFeedForwardLayer(
            input_dim,
            hidden_dim=hidden_dim,
            activation=torch_activation,
            structure=structure,
            linear_biases=linear_biases,
            norm=norm,
        ).eval()
        axlearn_layer_cfg = AxlearnTransformerFeedForwardLayer.default_config().set(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            structure=structure,
            activation=activation,
            norm=_axlearn_norm_from_name(norm),
            name="test",
        )
        set_bias_recursively(axlearn_layer_cfg, linear_biases)
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        rng = np.random.RandomState(123)
        inputs = rng.randn(2, 7, input_dim).astype(np.float32)
        torch_inputs = torch.as_tensor(inputs)
        axlearn_inputs = jnp.array(inputs)
        axlearn_outputs = np.array(
            F(
                axlearn_layer,
                jax.random.PRNGKey(0),
                state=axlearn_layer_state,
                inputs=(axlearn_inputs,),
                is_training=False,
                method="forward",
            )[0]
        )
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        torch_outputs = torch_layer.eval()(torch_inputs).detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs)

    @parameterized.parameters(itertools.product((True, False), (True, False)))
    def test_qkv_projection_layer(self, use_fused_qkv_impl: bool, use_linear_biases: bool):
        query_dim = key_dim = value_dim = 8
        num_heads = 2
        per_head_dim = query_dim // num_heads
        if use_fused_qkv_impl:
            torch_layer_cls = FusedQKVLinear
            axlearn_layer_cls = AxlearnFusedQKVLinear
        else:
            torch_layer_cls = QKVLinear
            axlearn_layer_cls = AxlearnQKVLinear
        torch_layer = torch_layer_cls(
            query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            linear_biases=use_linear_biases,
        ).eval()
        axlearn_layer_cfg = axlearn_layer_cls.default_config().set(
            query_dim=query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            name="test",
        )
        set_bias_recursively(axlearn_layer_cfg, use_linear_biases)
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        rng = np.random.RandomState(123)
        inputs = rng.randn(3, 11, query_dim).astype(np.float32)
        torch_inputs = torch.as_tensor(inputs)
        torch_outputs = torch_layer(query=torch_inputs, key=torch_inputs, value=torch_inputs)
        axlearn_inputs = jnp.asarray(inputs)
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=dict(query=axlearn_inputs, key=axlearn_inputs, value=axlearn_inputs),
            is_training=False,
            method="forward",
        )[0]
        self.assertNestedAllClose(torch_outputs.query.detach().numpy(), axlearn_outputs.query)
        self.assertNestedAllClose(torch_outputs.key.detach().numpy(), axlearn_outputs.key)
        self.assertNestedAllClose(torch_outputs.value.detach().numpy(), axlearn_outputs.value)

    @parameterized.parameters(1, 2, 4, 12)
    def test_multihead_attention_layer_extend(self, extend_chunk_size: int):
        query_dim = 8
        num_heads = 4

        multihead_attention = MultiheadAttention(
            query_dim=query_dim,
            key_dim=query_dim,
            value_dim=query_dim,
            num_heads=num_heads,
        ).eval()

        batch_size = 3
        max_seq_len = 12
        rng = np.random.RandomState(123)
        query = torch.as_tensor(rng.randn(batch_size, max_seq_len, query_dim).astype(np.float32))
        attention_logit_biases = self._causal_self_attention_logit_biases(query.shape[1])
        forward_data = multihead_attention(
            query,
            key=query,
            value=query,
            attention_logit_biases=attention_logit_biases,
        ).data
        cache = multihead_attention.init_state(
            target_batch_size=batch_size, target_max_len=max_seq_len
        )
        extend_data: torch.Tensor = None
        for i in range(0, max_seq_len, extend_chunk_size):
            end_chunk_ix = i + extend_chunk_size
            cache, output = multihead_attention.extend(
                cache,
                query[:, i:end_chunk_ix, :],
                attention_logit_biases=attention_logit_biases[..., i:end_chunk_ix, :end_chunk_ix],
            )
            if extend_data is None:
                extend_data = output.data
            else:
                extend_data = torch.concat((extend_data, output.data), axis=1)
        # Check that sequential decoding yielded the same output as forward decoding.
        self.assertNestedAllClose(extend_data.detach().numpy(), forward_data.detach().numpy())

    @parameterized.parameters(itertools.product(STRUCTURES, NORMS))
    def test_transformer_attention_layer_forward(self, structure: str, norm: str):
        target_dim = 12
        source_dim = 12
        num_heads = 2

        torch_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        ).eval()

        axlearn_layer_cfg = AxlearnTransformerAttentionLayer.default_config().set(
            target_dim=target_dim,
            source_dim=source_dim,
            structure=structure,
            norm=_axlearn_norm_from_name(norm),
            name="test",
        )
        axlearn_layer_cfg.attention.set(num_heads=num_heads)
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))

        rng = np.random.RandomState(123)
        target = rng.randn(2, 7, target_dim).astype(np.float32)
        attention_logit_biases = np.zeros(target.shape[:-1]).astype(float)[:, :, None]
        attention_logit_biases[:, -2:] = attention_bias.NEG_INF
        torch_inputs = {
            "target": torch.as_tensor(target),
            "attention_logit_biases": torch.as_tensor(attention_logit_biases),
        }
        axlearn_inputs = {
            "target": jnp.array(target),
            "attention_logit_biases": jnp.array(attention_logit_biases),
        }

        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=axlearn_inputs,
            is_training=False,
            method="forward",
        )[0]
        torch_outputs = torch_layer(**torch_inputs).data.detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs.data)

    @parameterized.parameters(itertools.product(STRUCTURES, NORMS))
    def test_transformer_attention_layer_extend(self, structure: str, norm: str):
        target_dim = 12
        source_dim = 12
        num_heads = 2

        layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        ).eval()

        batch_size = 3
        max_seq_len = 11
        rng = np.random.RandomState(123)
        target = torch.as_tensor(rng.randn(batch_size, max_seq_len, target_dim).astype(np.float32))
        attention_logit_biases = self._causal_self_attention_logit_biases(target.shape[1])
        cache = layer.init_state(target_batch_size=batch_size, target_max_len=max_seq_len)
        extend_data: torch.Tensor = None
        for i in range(max_seq_len):
            end_chunk_ix = i + 1
            cache, output = layer.extend(
                cache,
                target[:, i:end_chunk_ix, :],
                attention_logit_biases=attention_logit_biases[..., i:end_chunk_ix, :end_chunk_ix],
            )
            if extend_data is None:
                extend_data = output.data
            else:
                extend_data = torch.concat((extend_data, output.data), axis=1)
        forward_output = layer(target=target, attention_logit_biases=attention_logit_biases)
        # Forward matches iterative decoding.
        self.assertNestedAllClose(
            extend_data.detach().numpy(), forward_output.data.detach().numpy()
        )

    @parameterized.parameters(itertools.product(STRUCTURES, NORMS))
    def test_transformer_layer(self, structure: str, norm: str):
        target_dim = 12
        source_dim = 12
        num_heads = 2

        # Torch layer.
        torch_self_attention_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        )
        torch_cross_attention_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        )
        torch_feed_forward_layer = TransformerFeedForwardLayer(
            input_dim=target_dim, hidden_dim=4 * target_dim, structure=structure, norm=norm
        )
        torch_layer = TransformerLayer(
            self_attention=torch_self_attention_layer,
            feed_forward=torch_feed_forward_layer,
            cross_attention=torch_cross_attention_layer,
        ).eval()

        # AXLearn layer.
        axlearn_layer_cfg = AxlearnTransformerLayer.default_config().set(
            name="test",
            input_dim=target_dim,
        )
        axlearn_layer_cfg.self_attention.set(
            structure=structure,
        )
        axlearn_norm = _axlearn_norm_from_name(norm)
        axlearn_layer_cfg.self_attention.norm = axlearn_norm
        axlearn_layer_cfg.self_attention.attention.num_heads = num_heads
        axlearn_layer_cfg.feed_forward.hidden_dim = 4 * target_dim
        axlearn_layer_cfg.feed_forward.activation = "nn.gelu"
        axlearn_layer_cfg.feed_forward.structure = structure
        axlearn_layer_cfg.feed_forward.norm = axlearn_norm
        axlearn_layer_cfg.cross_attention = axlearn_layer_cfg.self_attention.clone()
        axlearn_layer_cfg.cross_attention.source_dim = source_dim
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))

        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))

        rng = np.random.RandomState(123)
        seq_len = 7
        batch_size = 2
        data = rng.randn(batch_size, seq_len, target_dim).astype(np.float32)
        self_attention_logit_biases = self._causal_self_attention_logit_biases(seq_len)
        cross_attention_logit_biases = np.zeros_like(self_attention_logit_biases)
        cross_attention_logit_biases[:, :2] = True
        cross_attention_logit_biases = cross_attention_logit_biases * NEG_INF
        torch_inputs = {
            "data": torch.as_tensor(data),
            "self_attention_logit_biases": torch.as_tensor(self_attention_logit_biases),
            "cross_attention_logit_biases": torch.as_tensor(cross_attention_logit_biases),
        }
        axlearn_inputs = {
            "data": jnp.array(data),
            "self_attention_logit_biases": jnp.array(self_attention_logit_biases),
            "cross_attention_logit_biases": jnp.array(cross_attention_logit_biases),
        }
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=axlearn_inputs,
            is_training=False,
            method="forward",
        )[0]
        torch_outputs = torch_layer(**torch_inputs).data.detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs.data)

        # Test iterative decoding.
        cache = torch_layer.init_state(target_batch_size=batch_size, target_max_len=seq_len)
        extend_data = None
        for i in range(seq_len):
            end_chunk_ix = i + 1
            cache, output = torch_layer.extend(
                cache,
                data=torch_inputs["data"][:, i:end_chunk_ix, :],
                self_attention_logit_biases=torch_inputs["self_attention_logit_biases"][
                    ..., i:end_chunk_ix, :end_chunk_ix
                ],
                cross_attention_logit_biases=torch_inputs["cross_attention_logit_biases"],
            )
            if extend_data is None:
                extend_data = output.data
            else:
                extend_data = torch.concat((extend_data, output.data), axis=1)
        self.assertNestedAllClose(extend_data, torch_outputs)

    @parameterized.parameters(itertools.product(STRUCTURES, NORMS))
    def test_bottleneck_adapter_transformer_layer(self, structure: str, norm: str):
        target_dim, source_dim, num_heads = 8, 8, 2

        # Torch layer.
        torch_self_attention_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        )
        torch_cross_attention_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
            norm=norm,
        )
        torch_feed_forward_layer = TransformerFeedForwardLayer(
            input_dim=target_dim, hidden_dim=4 * target_dim, structure=structure, norm=norm
        )
        torch_transformer_layer = TransformerLayer(
            self_attention=torch_self_attention_layer,
            feed_forward=torch_feed_forward_layer,
            cross_attention=torch_cross_attention_layer,
        )
        torch_adapter = TransformerFeedForwardLayer(
            input_dim=target_dim,
            hidden_dim=int(target_dim * 0.5),
            structure="postnorm",
            activation=_torch_activation_fn("nn.relu"),
        )
        torch_layer = BottleNeckAdapterTransformerLayer(
            layer=torch_transformer_layer, adapter=torch_adapter
        ).eval()

        # AXLearn layer.
        axlearn_transformer_layer_cfg = AxlearnTransformerLayer.default_config().set(
            input_dim=target_dim,
        )
        axlearn_transformer_layer_cfg.feed_forward.set(input_dim=target_dim)
        axlearn_transformer_layer_cfg.self_attention.set(
            structure=structure,
        )
        axlearn_transformer_layer_cfg.self_attention.attention.num_heads = num_heads

        axlearn_norm = _axlearn_norm_from_name(norm)
        axlearn_transformer_layer_cfg.self_attention.norm = axlearn_norm
        axlearn_transformer_layer_cfg.self_attention.attention.num_heads = num_heads
        axlearn_transformer_layer_cfg.feed_forward.hidden_dim = 4 * target_dim
        axlearn_transformer_layer_cfg.feed_forward.activation = "nn.gelu"
        axlearn_transformer_layer_cfg.feed_forward.structure = structure
        axlearn_transformer_layer_cfg.feed_forward.norm = axlearn_norm
        axlearn_transformer_layer_cfg.cross_attention = (
            axlearn_transformer_layer_cfg.self_attention.clone()
        )
        axlearn_transformer_layer_cfg.cross_attention.source_dim = source_dim

        axlearn_adapter_cfg = AxlearnTransformerFeedForwardLayer.default_config()
        axlearn_adapter_cfg.set(
            input_dim=target_dim,
            hidden_dim=target_dim // 2,
            structure="postnorm",
            activation="nn.relu",
        )
        axlearn_layer_cfg = AxlearnBottleNeckAdapterTransformerLayer.default_config()
        axlearn_layer_cfg.set(
            layer=axlearn_transformer_layer_cfg,
            adapter=axlearn_adapter_cfg,
            name="test",
            input_dim=target_dim,
        )
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)

        # Init axlearn layer state
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))

        # Load axlearn layer state into torch layer
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))

        # Define inputs
        batch_size, seq_len = 3, 8
        rng = np.random.RandomState(1)
        data = rng.randn(batch_size, seq_len, target_dim).astype(np.float32)
        self_attention_logit_biases = self._causal_self_attention_logit_biases(seq_len)
        cross_attention_logit_biases = np.zeros_like(self_attention_logit_biases)
        cross_attention_logit_biases[:, :2] = True
        cross_attention_logit_biases = cross_attention_logit_biases * NEG_INF
        torch_inputs = {
            "data": torch.as_tensor(data),
            "self_attention_logit_biases": torch.as_tensor(self_attention_logit_biases),
            "cross_attention_logit_biases": torch.as_tensor(cross_attention_logit_biases),
        }
        axlearn_inputs = {
            "data": jnp.array(data),
            "self_attention_logit_biases": jnp.array(self_attention_logit_biases),
            "cross_attention_logit_biases": jnp.array(cross_attention_logit_biases),
        }
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=axlearn_inputs,
            is_training=False,
            method="forward",
        )[0]
        torch_outputs = torch_layer(**torch_inputs).data.detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs.data)

        # Test iterative decoding.
        cache = torch_layer.init_state(target_batch_size=batch_size, target_max_len=seq_len)
        extend_data = None
        for i in range(seq_len):
            end_chunk_ix = i + 1
            cache, output = torch_layer.extend(
                cache,
                data=torch_inputs["data"][:, i:end_chunk_ix, :],
                self_attention_logit_biases=torch_inputs["self_attention_logit_biases"][
                    ..., i:end_chunk_ix, :end_chunk_ix
                ],
                cross_attention_logit_biases=torch_inputs["cross_attention_logit_biases"],
            )
            if extend_data is None:
                extend_data = output.data
            else:
                extend_data = torch.concat((extend_data, output.data), axis=1)
        self.assertNestedAllClose(extend_data, torch_outputs)

    @parameterized.parameters(*STRUCTURES)
    def test_stacked_transformer_layer(self, structure: str):
        target_dim = 12
        source_dim = 12
        num_heads = 2
        num_layers = 4

        # Torch layer.
        torch_self_attention_layer = TransformerAttentionLayer(
            target_dim=target_dim,
            source_dim=source_dim,
            num_heads=num_heads,
            structure=structure,
        )
        torch_feed_forward_layer = TransformerFeedForwardLayer(
            input_dim=target_dim,
            hidden_dim=4 * target_dim,
            structure=structure,
            activation=_torch_activation_fn("nn.relu"),
        )
        torch_transformer_layer = TransformerLayer(
            self_attention=torch_self_attention_layer,
            feed_forward=torch_feed_forward_layer,
            cross_attention=None,
        )
        torch_layer = StackedTransformerLayer(
            num_layers=num_layers,
            layer=torch_transformer_layer,
        ).eval()

        # AXLearn layer.
        axlearn_transformer_layer_cfg = AxlearnTransformerLayer.default_config()
        axlearn_transformer_layer_cfg.self_attention.set(
            structure=structure,
        )
        axlearn_transformer_layer_cfg.self_attention.attention.num_heads = num_heads
        axlearn_transformer_layer_cfg.feed_forward.hidden_dim = 4 * target_dim
        axlearn_transformer_layer_cfg.feed_forward.activation = "nn.relu"
        axlearn_transformer_layer_cfg.feed_forward.structure = structure
        axlearn_transformer_layer_cfg.cross_attention = None
        axlearn_layer_cfg = AxlearnStackedTransformerLayer.default_config().set(
            input_dim=target_dim,
            num_layers=num_layers,
            layer=axlearn_transformer_layer_cfg,
            name="test",
        )
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))

        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))

        batch_size = 3
        seq_len = 7
        rng = np.random.RandomState(1)
        data = rng.randn(batch_size, seq_len, target_dim).astype(np.float32)
        self_attention_logit_biases = self._causal_self_attention_logit_biases(seq_len)
        cross_attention_logit_biases = np.zeros_like(self_attention_logit_biases)
        cross_attention_logit_biases[:, :2] = True
        cross_attention_logit_biases = cross_attention_logit_biases * NEG_INF
        torch_inputs = {
            "data": torch.as_tensor(data),
            "self_attention_logit_biases": torch.as_tensor(self_attention_logit_biases),
            "cross_attention_logit_biases": torch.as_tensor(cross_attention_logit_biases),
        }
        axlearn_inputs = {
            "data": jnp.array(data),
            "self_attention_logit_biases": jnp.array(self_attention_logit_biases),
            "cross_attention_logit_biases": jnp.array(cross_attention_logit_biases),
        }
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=axlearn_inputs,
            is_training=False,
            method="forward",
        )[0]
        torch_outputs = torch_layer(**torch_inputs).data.detach().numpy()
        self.assertNestedAllClose(torch_outputs, axlearn_outputs.data)

        # Test iterative decoding.
        cache = torch_layer.init_state(target_batch_size=batch_size, target_max_len=seq_len)
        extend_data = None
        for i in range(seq_len):
            end_chunk_ix = i + 1
            cache, output = torch_layer.extend(
                cache,
                data=torch_inputs["data"][:, i:end_chunk_ix, :],
                self_attention_logit_biases=torch_inputs["self_attention_logit_biases"][
                    ..., i:end_chunk_ix, :end_chunk_ix
                ],
                cross_attention_logit_biases=torch_inputs["cross_attention_logit_biases"],
            )
            if extend_data is None:
                extend_data = output.data
            else:
                extend_data = torch.concat((extend_data, output.data), axis=1)
        self.assertNestedAllClose(extend_data, torch_outputs)

    def test_causal_attention_mask_layer(self):
        batch_size = 2
        seq_len = 4
        rng = np.random.RandomState(1)
        input_ids = rng.randint(0, 32, (batch_size, seq_len)).astype(np.int32)
        segment_ids = (input_ids != 0).astype(np.int32)
        positions = np.tile(np.arange(3, seq_len + 3)[None, :], (batch_size, 1))
        torch_outputs = (
            CausalAttentionLogitBiasLayer()(
                segment_ids=torch.as_tensor(segment_ids),
                positions=torch.as_tensor(positions),
            )
            .detach()
            .numpy()
        )
        axlearn_outputs = F(
            AxlearnCausalAttentionLogitBiasLayer.default_config()
            .set(name="test")
            .instantiate(parent=None),
            jax.random.PRNGKey(1),
            state={},
            inputs=dict(
                segment_ids=jnp.array(segment_ids),
                positions=jnp.array(positions),
            ),
            is_training=False,
            method="forward",
        )[0]
        self.assertNestedAllClose(np.exp(torch_outputs), np.exp(axlearn_outputs))

    @parameterized.parameters(2, 4, 16, 64)
    def test_alibi_slopes(self, num_heads: int):
        self.assertNestedAllClose(alibi_get_slopes(num_heads), axlearn_alibi_get_slopes(num_heads))

    @parameterized.parameters(2, 4, 16)
    def test_alibi_attention_mask_layer(self, num_heads: int):
        batch_size = 3
        seq_len = 13
        rng = np.random.RandomState(1)
        input_ids = rng.randint(0, 22, (batch_size, seq_len)).astype(np.int32)
        segment_ids = (input_ids != 0).astype(np.int32)
        positions = np.tile(np.arange(5, seq_len + 5, dtype=np.int32)[None, :], (batch_size, 1))
        torch_outputs = (
            ALiBiAttentionLogitBiasLayer(num_heads)(
                segment_ids=torch.as_tensor(segment_ids),
                positions=torch.as_tensor(positions),
            )
            .detach()
            .numpy()
        )
        axlearn_outputs = F(
            AxlearnALiBiAttentionLogitBiasLayer.default_config()
            .set(name="test", num_heads=num_heads)
            .instantiate(parent=None),
            jax.random.PRNGKey(2),
            state={},
            inputs=dict(
                segment_ids=jnp.array(segment_ids),
                positions=jnp.array(positions),
            ),
            is_training=False,
            method="forward",
        )[0]
        self.assertNestedAllClose(np.exp(torch_outputs), np.exp(axlearn_outputs))

    @staticmethod
    def _causal_self_attention_logit_biases(query_len: int) -> torch.Tensor:
        positions = torch.arange(query_len)[None, :]
        attention_logit_biases = (
            positions[:, None, :, None] < positions[:, None, None, :]
        ) * NEG_INF
        return attention_logit_biases


def _axlearn_norm_from_name(norm: str) -> InstantiableConfig:
    if norm == "layernorm":
        return AxlearnLayerNorm.default_config()
    elif norm == "rmsnorm":
        return AxlearnRMSNorm.default_config()
    else:
        raise NotImplementedError(f"No mapping for {norm}")


class TransformerEmbeddingsTest(TestCase):
    @parameterized.parameters(
        itertools.product((True, False), (True, False), (None, "layernorm", "rmsnorm"))
    )
    def test_transformer_embeddings_forward(
        self,
        use_type_emb: bool,
        use_pos_emb: bool,
        norm: Optional[str],
    ):
        batch_size = 5
        seq_len = 11
        vocab_size = 13
        emb_dim = 7
        token_emb = Embedding(vocab_size, embedding_dim=emb_dim)
        rng = np.random.RandomState(1)
        input_ids = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        if use_type_emb:
            num_type_emb = 3
            type_emb = Embedding(num_type_emb, embedding_dim=emb_dim)
            axlearn_type_emb = AxlearnEmbedding.default_config().set(num_embeddings=num_type_emb)
            token_type_ids = rng.randint(0, num_type_emb, (batch_size, seq_len)).astype(np.int32)
            axlearn_token_type_ids = jnp.asarray(token_type_ids)
            torch_token_type_ids = torch.as_tensor(token_type_ids)
        else:
            type_emb = None
            axlearn_type_emb = None
            axlearn_token_type_ids = None
            torch_token_type_ids = None
        if use_pos_emb:
            pos_emb = Embedding(seq_len, embedding_dim=emb_dim)
            axlearn_pos_emb = AxlearnEmbedding.default_config().set(num_embeddings=seq_len)
            positions = np.tile(np.arange(seq_len, dtype=np.int32)[None, :], (batch_size, 1))
            axlearn_positions = jnp.array(positions)
            torch_positions = torch.as_tensor(positions)
        else:
            pos_emb = None
            axlearn_pos_emb = None
            axlearn_positions = None
            torch_positions = None
        torch_layer = TransformerEmbeddings(
            token_emb, type_emb=type_emb, pos_emb=pos_emb, norm=norm
        )
        axlearn_layer = (
            AxlearnTransformerEmbeddings.default_config()
            .set(
                dim=emb_dim,
                vocab_size=vocab_size,
                type_emb=axlearn_type_emb,
                pos_emb=axlearn_pos_emb,
                norm=None if norm is None else _axlearn_norm_from_name(norm),
                name="test",
            )
            .instantiate(parent=None)
        )
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        # Load weights into torch model.
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        # Model outputs.
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=dict(
                input_batch=dict(
                    inputs=jnp.asarray(input_ids),
                    token_type_ids=axlearn_token_type_ids,
                    positions=axlearn_positions,
                ),
            ),
            is_training=False,
            method="forward",
        )[0]
        torch_outputs = (
            torch_layer(
                torch.as_tensor(input_ids),
                token_type_ids=torch_token_type_ids,
                positions=torch_positions,
            )
            .detach()
            .numpy()
        )
        self.assertNestedAllClose(torch_outputs, axlearn_outputs)


class DecoderTest(TestCase):
    @set_threefry_partitionable(False)  # TODO(markblee): update for threefry_partitionable True
    def test_decoder_inference(self):
        vocab_size = 13
        emb_dim = 8
        num_layers = 4
        num_heads = 2
        ff_dim = 4 * emb_dim
        torch_layer = Decoder(
            attention_mask=CausalAttentionLogitBiasLayer(),
            emb=TransformerEmbeddings(token_emb=Embedding(vocab_size, embedding_dim=emb_dim)),
            transformer=StackedTransformerLayer(
                num_layers,
                layer=TransformerLayer(
                    self_attention=TransformerAttentionLayer(
                        target_dim=emb_dim,
                        source_dim=emb_dim,
                        num_heads=num_heads,
                        structure="prenorm",
                        norm="layernorm",
                    ),
                    feed_forward=TransformerFeedForwardLayer(
                        input_dim=emb_dim,
                        hidden_dim=ff_dim,
                        activation=_torch_activation_fn("nn.relu"),
                        structure="prenorm",
                        norm="layernorm",
                    ),
                ),
            ),
            output_norm="layernorm",
        ).eval()
        axlearn_transformer_layer_cfg = AxlearnTransformerLayer.default_config()
        axlearn_transformer_layer_cfg.feed_forward.activation = "nn.relu"
        axlearn_transformer_layer_cfg.feed_forward.hidden_dim = ff_dim
        axlearn_transformer_layer_cfg.self_attention.attention.num_heads = num_heads
        axlearn_layer_cfg = AxlearnDecoder.default_config().set(
            vocab_size=vocab_size,
            dim=emb_dim,
            transformer=AxlearnStackedTransformerLayer.default_config().set(
                num_layers=num_layers, layer=axlearn_transformer_layer_cfg
            ),
            name="test",
        )
        # Default emb initialization is small enough that 1/500 elements will differ by 1e-5
        # across torch/axlearn impl.
        axlearn_layer_cfg.emb.token_emb.param_init.init_by_param_name = {
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, distribution="normal"
            )
        }
        axlearn_layer = axlearn_layer_cfg.instantiate(parent=None)
        axlearn_layer_state = axlearn_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        torch_layer.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_layer_state))
        batch_size = 5
        seq_len = 12
        rng = np.random.RandomState(1)
        input_ids = rng.randint(0, vocab_size, (batch_size, seq_len))
        torch_input_ids = torch.as_tensor(input_ids)
        torch_outputs = torch_layer(torch_input_ids)
        axlearn_outputs = F(
            axlearn_layer,
            jax.random.PRNGKey(0),
            state=axlearn_layer_state,
            inputs=dict(input_batch=dict(input_ids=jnp.asarray(input_ids))),
            is_training=False,
            method="forward",
        )[0]
        self.assertNestedAllClose(
            torch_outputs["hidden_states"].detach().numpy(), axlearn_outputs["hidden_states"]
        )
        self.assertNestedAllClose(
            torch_outputs["logits"].detach().numpy(), axlearn_outputs["logits"]
        )
        # Also test extend.
        for chunk_size in [1, 6]:
            extend_logits: torch.Tensor = None
            extend_hidden_states: torch.Tensor = None
            cache = torch_layer.init_state(batch_size=batch_size, max_sequence_length=seq_len)
            for i in range(0, seq_len, chunk_size):
                end_chunk_ix = i + chunk_size
                cache, output = torch_layer.extend(
                    cached_state=cache,
                    input_ids=torch_input_ids[:, i:end_chunk_ix],
                )
                if extend_hidden_states is None:
                    extend_hidden_states = output["hidden_states"]
                    extend_logits = output["logits"]
                else:
                    extend_hidden_states = torch.concat(
                        (extend_hidden_states, output["hidden_states"]), axis=1
                    )
                    extend_logits = torch.concat((extend_logits, output["logits"]), axis=1)
            # Check that sequential decoding yielded the same output as forward decoding.
            self.assertNestedAllClose(
                extend_hidden_states.detach().numpy(),
                torch_outputs["hidden_states"].detach().numpy(),
            )
            self.assertNestedAllClose(
                extend_logits.detach().numpy(), torch_outputs["logits"].detach().numpy()
            )

    @parameterized.parameters(
        dict(
            input_ids=torch.tensor(
                [
                    [1, 0, 2, 3, 0, 0],
                    [1, 2, 3, 4, 0, 5],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            pad_token_id=0,
            expected_segment_ids=torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        dict(
            input_ids=torch.tensor(
                [
                    [1, -1, 2, 3, -1, -1],
                    [1, 2, 3, 4, -1, 5],
                    [0, 0, 0, -1, -1, -1],
                ]
            ),
            pad_token_id=-1,
            expected_segment_ids=torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                ]
            ),
        ),
    )
    def test_segment_ids_from_causal_input_ids(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
        expected_segment_ids: torch.Tensor,
    ):
        self.assertTrue(
            torch.all(
                _segment_ids_from_causal_input_ids(input_ids=input_ids, pad_token_id=pad_token_id)
                == expected_segment_ids
            )
        )


class VisionTransformerModulesTest(TestCase):
    @parameterized.parameters("cls_token", "gap")
    def test_vit_model(self, global_feature_extraction: str):
        cfg_name = "Test16"
        image_size = (224, 224)
        # Model init.
        torch_model = ViTModelBuilder.from_name(
            cfg_name,
            image_size=image_size,
            global_feature_extraction=global_feature_extraction,
        ).eval()
        axlearn_vit_cfg = axlearn_vit_configs()[cfg_name]
        axlearn_model_cfg = ImageClassificationModel.default_config().set(
            name="test", backbone=axlearn_vit_cfg, num_classes=1000
        )
        feed_forward_dim = torch_model.backbone.encoder_1d.pos_emb.weight.shape[1]
        if global_feature_extraction == "cls_token":
            axlearn_model_cfg.backbone.num_cls_tokens = 1  # pytype: disable=attribute-error
            axlearn_model_cfg.backbone.pooler = (
                AxlearnFirstNTokenPooling.default_config().set(  # pytype: disable=attribute-error
                    input_dim=feed_forward_dim, output_dim=feed_forward_dim, num_outputs=1
                )
            )
        elif global_feature_extraction == "gap":
            axlearn_model_cfg.backbone.num_cls_tokens = 0  # pytype: disable=attribute-error
            axlearn_model_cfg.backbone.pooler = (
                AxlearnAveragePooling.default_config().set(  # pytype: disable=attribute-error
                    input_dim=feed_forward_dim,
                    output_dim=feed_forward_dim,
                    num_outputs=1,
                )
            )
        axlearn_model_cfg.backbone.encoder_1d.pos_emb.shape = (
            torch_model.backbone.encoder_1d.pos_emb.weight.shape[
                0
            ],  # pytype: disable=attribute-error
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        axlearn_model_state = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
        # Load weights into torch model.
        torch_model.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_model_state))
        # Fake inputs.
        rng = np.random.RandomState(1)
        image = rng.randn(2, 3, *image_size).astype(np.float32)
        torch_inputs = dict(image=torch.as_tensor(image))
        axlearn_inputs = dict(image=jnp.array(image).transpose(0, 2, 3, 1))
        # Model outputs.
        axlearn_outputs = F(
            axlearn_model,
            jax.random.PRNGKey(0),
            state=axlearn_model_state,
            inputs=dict(input_batch=axlearn_inputs),
            is_training=False,
            method="predict",
        )[0]
        torch_logits = torch_model.predict(**torch_inputs).detach().numpy()
        self.assertNestedAllClose(torch_logits, axlearn_outputs["logits"])


def _build_axlearn_causal_lm_cfg(
    vocab_size: int, *, num_layers: int, num_heads: int, hidden_dim: int
) -> AxlearnCausalLmModel.Config:
    # Build AXLearn Causal LM.
    axlearn_layer_cfg = AxlearnTransformerLayer.default_config()
    axlearn_layer_cfg.feed_forward.activation = ("nn.silu", "linear")
    axlearn_layer_cfg.feed_forward.hidden_dim = axlearn_scaled_hidden_dim(21.0 / 8.0)
    axlearn_layer_cfg.self_attention.attention.num_heads = num_heads
    axlearn_layer_cfg.self_attention.attention.input_linear = AxlearnFusedQKVLinear.default_config()
    axlearn_stack_cfg = AxlearnStackedTransformerLayer.default_config().set(
        num_layers=num_layers,
        layer=axlearn_layer_cfg,
    )
    axlearn_decoder_cfg = AxlearnDecoder.default_config().set(
        transformer=axlearn_stack_cfg,
        attention_mask=AxlearnALiBiAttentionLogitBiasLayer.default_config().set(
            num_heads=num_heads
        ),
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=AxlearnTransformerEmbeddings.default_config().set(pos_emb=None),
        dropout_rate=0.0,
    )
    axlearn_model_cfg = AxlearnCausalLmModel.default_config().set(
        decoder=axlearn_decoder_cfg,
        dtype=jnp.float32,
        name="test",
    )
    set_bias_recursively(axlearn_model_cfg, False)
    set_norm_recursively(
        axlearn_model_cfg, AxlearnRMSNorm.default_config().set(eps=1e-5, forward_dtype=None)
    )
    return axlearn_model_cfg


def _build_axlearn_adapter_causal_lm_cfg(
    vocab_size: int, *, num_layers: int, num_heads: int, hidden_dim: int
) -> AxlearnCausalLmModel.Config:
    axlearn_layer_cfg = AxlearnTransformerLayer.default_config()
    axlearn_layer_cfg.feed_forward.activation = ("nn.silu", "linear")
    axlearn_layer_cfg.feed_forward.hidden_dim = axlearn_scaled_hidden_dim(21.0 / 8.0)
    axlearn_layer_cfg.self_attention.attention.num_heads = num_heads
    axlearn_layer_cfg.self_attention.attention.input_linear = AxlearnFusedQKVLinear.default_config()
    axlearn_layer_cfg.input_dim = hidden_dim
    axlearn_linear_cfg = AxlearnTransformerFeedForwardLayer.default_config()
    axlearn_linear_cfg.set(
        input_dim=hidden_dim,
        structure="postnorm",
        activation="nn.relu",
    )
    axlearn_adapter_cfg = AxlearnBottleNeckAdapterTransformerLayer.default_config()
    axlearn_adapter_cfg.set(
        layer=axlearn_layer_cfg,
        adapter=axlearn_linear_cfg,
    )
    axlearn_stack_cfg = AxlearnStackedTransformerLayer.default_config().set(
        num_layers=num_layers,
        layer=axlearn_adapter_cfg,
    )
    axlearn_decoder_cfg = AxlearnDecoder.default_config().set(
        transformer=axlearn_stack_cfg,
        attention_mask=AxlearnALiBiAttentionLogitBiasLayer.default_config().set(
            num_heads=num_heads
        ),
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=AxlearnTransformerEmbeddings.default_config().set(pos_emb=None),
        dropout_rate=0.0,
    )
    axlearn_model_cfg = AxlearnCausalLmModel.default_config().set(
        decoder=axlearn_decoder_cfg,
        dtype=jnp.float32,
        name="test",
    )
    set_bias_recursively(axlearn_model_cfg, False)
    set_norm_recursively(
        axlearn_model_cfg, AxlearnRMSNorm.default_config().set(eps=1e-5, forward_dtype=None)
    )
    return axlearn_model_cfg


class CausalLmModelModulesTest(TestCase):
    def test_causal_lm_v1_from_args(self):
        vocab_size = 11
        torch_cfg = CausalLmModelBuilder.configs["v1-test"]
        # Build AXLearn model.
        axlearn_model_cfg = _build_axlearn_causal_lm_cfg(
            vocab_size,
            num_layers=torch_cfg["num_layers"],
            num_heads=torch_cfg["num_heads"],
            hidden_dim=torch_cfg["hidden_dim"],
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        axlearn_model_state = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
        # Build torch model.
        torch_model = CausalLmModelBuilder.v1_from_args(vocab_size=vocab_size, **torch_cfg).eval()
        torch_model.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_model_state))
        # Fake inputs.
        rng = np.random.RandomState(1)
        batch_size = 3
        seq_len = 7
        input_ids = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        torch_input_ids = torch.as_tensor(input_ids)
        target_labels = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        # Model outputs.
        _, axlearn_predictions = F(
            axlearn_model,
            jax.random.PRNGKey(0),
            state=axlearn_model_state,
            inputs=dict(
                input_batch=dict(
                    input_ids=jnp.asarray(input_ids), target_labels=jnp.asarray(target_labels)
                ),
                return_aux=True,
            ),
            is_training=False,
            method="forward",
        )[0]
        _, torch_predictions = torch_model(
            torch_input_ids, target_labels=torch.as_tensor(target_labels)
        )
        self.assertNestedAllClose(
            torch_predictions["logits"].detach().numpy(), axlearn_predictions["logits"]
        )
        self.assertNestedAllClose(
            torch_predictions["hidden_states"].detach().numpy(),
            axlearn_predictions["hidden_states"],
        )
        # Also test iterative decoding.
        extend_logits: torch.Tensor = None
        extend_hidden_states: torch.Tensor = None
        cache = torch_model.init_state(batch_size=batch_size, max_sequence_length=seq_len)
        for i in range(seq_len):
            end_chunk_ix = i + 1
            cache, output = torch_model.extend(
                cached_state=cache,
                input_ids=torch_input_ids[:, i:end_chunk_ix],
            )
            if extend_hidden_states is None:
                extend_hidden_states = output["hidden_states"]
                extend_logits = output["logits"]
            else:
                extend_hidden_states = torch.concat(
                    (extend_hidden_states, output["hidden_states"]), axis=1
                )
                extend_logits = torch.concat((extend_logits, output["logits"]), axis=1)
        # Check that sequential decoding yielded the same output as forward decoding.
        self.assertNestedAllClose(
            extend_hidden_states.detach().numpy(),
            torch_predictions["hidden_states"].detach().numpy(),
        )
        self.assertNestedAllClose(
            extend_logits.detach().numpy(), torch_predictions["logits"].detach().numpy()
        )


class AdapterCausalLmModelModulesTest(TestCase):
    def test_adapter_causal_lm_v1_from_args(self):
        vocab_size = 11
        torch_cfg = AdapterCausalLmModelBuilder.configs["v1-test"]
        # Build AXLearn model.
        axlearn_model_cfg = _build_axlearn_adapter_causal_lm_cfg(
            vocab_size,
            num_layers=torch_cfg["num_layers"],
            num_heads=torch_cfg["num_heads"],
            hidden_dim=torch_cfg["hidden_dim"],
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        axlearn_model_state = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
        # Build torch model.
        torch_model = AdapterCausalLmModelBuilder.v1_from_args(
            vocab_size=vocab_size, **torch_cfg
        ).eval()
        torch_model.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_model_state))
        # Fake inputs.
        rng = np.random.RandomState(1)
        batch_size = 3
        seq_len = 7
        input_ids = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        torch_input_ids = torch.as_tensor(input_ids)
        target_labels = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        # Model outputs.
        _, axlearn_predictions = F(
            axlearn_model,
            jax.random.PRNGKey(0),
            state=axlearn_model_state,
            inputs=dict(
                input_batch=dict(
                    input_ids=jnp.asarray(input_ids), target_labels=jnp.asarray(target_labels)
                ),
                return_aux=True,
            ),
            is_training=False,
            method="forward",
        )[0]
        _, torch_predictions = torch_model(
            torch_input_ids, target_labels=torch.as_tensor(target_labels)
        )
        self.assertNestedAllClose(
            torch_predictions["logits"].detach().numpy(), axlearn_predictions["logits"]
        )
        self.assertNestedAllClose(
            torch_predictions["hidden_states"].detach().numpy(),
            axlearn_predictions["hidden_states"],
        )
        # Also test iterative decoding.
        extend_logits: torch.Tensor = None
        extend_hidden_states: torch.Tensor = None
        cache = torch_model.init_state(batch_size=batch_size, max_sequence_length=seq_len)
        for i in range(seq_len):
            end_chunk_ix = i + 1
            cache, output = torch_model.extend(
                cached_state=cache,
                input_ids=torch_input_ids[:, i:end_chunk_ix],
            )
            if extend_hidden_states is None:
                extend_hidden_states = output["hidden_states"]
                extend_logits = output["logits"]
            else:
                extend_hidden_states = torch.concat(
                    (extend_hidden_states, output["hidden_states"]), axis=1
                )
                extend_logits = torch.concat((extend_logits, output["logits"]), axis=1)
        # Check that sequential decoding yielded the same output as forward decoding.
        self.assertNestedAllClose(
            extend_hidden_states.detach().numpy(),
            torch_predictions["hidden_states"].detach().numpy(),
        )
        self.assertNestedAllClose(
            extend_logits.detach().numpy(), torch_predictions["logits"].detach().numpy()
        )


class SampleDecodingTest(TestCase):
    @parameterized.parameters(1, 3, 5)
    def test_top_k_logits_modifier_without_ties(self, top_k: int):
        shape = (3, 4, 7)
        logits = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape)
        top_k_logits = TopK(top_k)(logits)
        num_k = (top_k_logits > NEG_INF).sum(-1)
        self.assertNestedEqual(num_k, torch.ones_like(num_k) * top_k)

    @parameterized.parameters(1, 3, 4)
    def test_top_k_logits_modifier_with_ties(self, top_k: int):
        shape = (2, 18)
        logits = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape).tile(1, 2)
        top_k_logits = TopK(top_k)(logits)
        num_k = (top_k_logits > NEG_INF).sum(-1)
        # Two of each value so top-k is += top_k % 2.
        self.assertNestedEqual(num_k, (torch.ones_like(num_k) * top_k) + (top_k % 2))

    @parameterized.parameters(1e-4, 0.1, 0.5, 0.99, 1.0)
    def test_top_p_logits_modifier_without_ties(self, top_p: float):
        shape = (2, 11)
        logits = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape)
        top_p_logits = TopP(top_p)(logits)
        for eg, modified_eg in zip(logits, top_p_logits):
            probs = torch.nn.functional.softmax(eg, dim=-1)
            sorted_p, sorted_ix = torch.topk(probs, k=len(probs))
            cumulative_prob = 0
            for prob, ix in zip(sorted_p, sorted_ix):
                if cumulative_prob >= top_p:
                    # Outside of top-p, should be very small.
                    self.assertAlmostEqual(modified_eg[ix], NEG_INF)
                else:
                    cumulative_prob += prob
                    # Include.
                    self.assertAlmostEqual(modified_eg[ix], eg[ix], delta=1e-6)

    def test_scale_by(self):
        temperature = 0.5
        shape = (2, 11)
        logits = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape)
        scaled_logits = ScaleBy(temperature)(logits)
        self.assertNestedAllClose(
            scaled_logits.detach().numpy(), logits.detach().numpy() / temperature
        )

    def test_sample_decoding_session(self):
        # Test interactive model decoding.
        vocab_size = 11
        batch_size = 2
        max_sequence_len = 128
        session = SampleDecodingSession(
            CausalLmModelBuilder.from_name("v1-test", vocab_size=vocab_size).eval(),
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
        )
        # Run forward for several steps conditioned on initial user prompt.
        initial_user_prompt = torch.randint(0, vocab_size, (batch_size, 5))
        model_continuation = session.decode(7, prompt_ids=initial_user_prompt)
        self.assertEqual(torch.Size((batch_size, 7)), model_continuation.shape)
        # Continue for a few more steps.
        model_continuation = session.decode(3)
        self.assertEqual(torch.Size((batch_size, 3)), model_continuation.shape)
        # Respond to model continuation and generate a response.
        user_response = torch.randint(0, vocab_size, (batch_size, 3))
        model_continuation = session.decode(10, prompt_ids=user_response)
        self.assertEqual(torch.Size((batch_size, 10)), model_continuation.shape)
        # Continue for one more step.
        model_continuation = session.decode(1)
        self.assertEqual(torch.Size((batch_size, 1)), model_continuation.shape)

    def test_sample_decoding_session_exceed_length(self):
        # Test interactive model decoding.
        vocab_size = 11
        batch_size = 1
        max_sequence_len = 128

        # Fix PyTorch RNG.
        torch.manual_seed(10)

        session = SampleDecodingSession(
            CausalLmModelBuilder.from_name("v1-test", vocab_size=vocab_size).eval(),
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
        )

        # Run forward for several steps conditioned on initial user prompt.
        rng = np.random.RandomState(1)
        initial_user_prompt = torch.as_tensor(
            rng.randint(2, vocab_size, (batch_size, 5)).astype(np.int32)
        )
        model_continuation = session.decode(7, prompt_ids=initial_user_prompt)
        self.assertEqual(7, torch.count_nonzero(model_continuation))
        # Continue for a few more steps.
        model_continuation = session.decode(128)
        # Call extend 7 times, but the first call uses 5 positions.
        self.assertEqual(128 - (5 + 7 - 1), torch.count_nonzero(model_continuation))

    @parameterized.parameters(
        dict(
            builder=CausalLmModelBuilder,
            cfg_fn=_build_axlearn_causal_lm_cfg,
        ),
        dict(
            builder=AdapterCausalLmModelBuilder,
            cfg_fn=_build_axlearn_adapter_causal_lm_cfg,
        ),
    )
    def test_greedy_decoding(self, builder, cfg_fn):
        # Test that we get the same behavior as AXLearn for top-k decoding.
        vocab_size = 256
        torch_cfg = builder.configs["v1-test"]
        # Build AXLearn model.
        axlearn_model_cfg = cfg_fn(
            vocab_size,
            num_layers=torch_cfg["num_layers"],
            num_heads=torch_cfg["num_heads"],
            hidden_dim=torch_cfg["hidden_dim"],
        )
        axlearn_model = axlearn_model_cfg.instantiate(parent=None)
        axlearn_model_state = axlearn_model.initialize_parameters_recursively(jax.random.PRNGKey(0))
        # Build torch model.
        torch_model = builder.v1_from_args(vocab_size=vocab_size, **torch_cfg).eval()
        torch_model.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_model_state))
        # Prompt IDs.
        rng = np.random.RandomState(1)
        batch_size = 3
        prompt_len = 7
        max_sequence_len = 32
        prompt_ids = rng.randint(1, vocab_size, (batch_size, prompt_len)).astype(np.int32)
        # Model outputs.
        axlearn_predictions = F(
            axlearn_model,
            jax.random.PRNGKey(0),
            state=axlearn_model_state,
            inputs=dict(
                input_batch=dict(
                    prefix=jnp.pad(
                        jnp.asarray(prompt_ids), ((0, 0), (0, max_sequence_len - prompt_len))
                    ),
                ),
                num_decodes=1,
                logits_modifier=axlearn_top_k_logits(1),
            ),
            is_training=False,
            method="sample_decode",
        )[0].sequences[:, 0, :]
        session = SampleDecodingSession(
            torch_model,
            batch_size=batch_size,
            max_sequence_len=max_sequence_len,
            logits_modifier=TopK(1),
        )
        torch_predictions = session.decode(
            num_tokens=max_sequence_len - prompt_len, prompt_ids=torch.as_tensor(prompt_ids)
        )
        # Check that the greedy continuations match in AXLearn vs Torch.
        self.assertNestedEqual(
            torch_predictions.detach().numpy(), axlearn_predictions[:, prompt_len:]
        )


class AttentionPoolingTest(TestCase):
    @parameterized.parameters(itertools.product([1, 3], [13, 14], [12], [1, 3], [0, 0.1, 0.9]))
    def test_attention_pooling(
        self, num_outputs: int, input_dim: int, output_dim: int, num_heads: int, dropout_rate: float
    ):
        batch_size = 2
        seq_len = 10

        # Random inputs.
        rng = np.random.RandomState(123)
        inputs = rng.randn(batch_size, seq_len, input_dim).astype(np.float32)
        torch_inputs = torch.as_tensor(inputs)
        axlearn_inputs = dict(tokens=jnp.array(inputs))

        # Generate AXLearn output.
        cfg = AxlearnAttentionPooling.default_config().set(
            name="attention_pooling",
            num_outputs=num_outputs,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        cfg.cross_attention.attention.num_heads = num_heads
        cfg.cross_attention.attention.dropout.rate = dropout_rate
        cfg.feed_forward.dropout.rate = dropout_rate
        axlearn_pooler = cfg.instantiate(parent=None)  # type: AxlearnAttentionPooling

        # Initialize pooler parameters.
        axlearn_pooler_state = axlearn_pooler.initialize_parameters_recursively(
            jax.random.PRNGKey(0)
        )

        axlearn_output, _ = F(
            axlearn_pooler,
            inputs=axlearn_inputs,
            is_training=False,
            state=axlearn_pooler_state,
            prng_key=jax.random.PRNGKey(0),
        )
        axlearn_output = np.array(axlearn_output)

        # Generate PyTorch output.
        torch_pooler = AttentionPooling(
            num_outputs=num_outputs,
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
        ).eval()

        torch_pooler.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_pooler_state))
        torch_output = torch_pooler(torch_inputs).detach().numpy()
        self.assertNestedAllClose(torch_output, axlearn_output)


class CoCaImageStreamEncoderTest(TestCase):
    @staticmethod
    def _get_axlearn_pooler_cls(pooler_name: str) -> type:
        if pooler_name == "attention":
            return AxlearnAttentionPooling
        elif pooler_name == "average":
            return AxlearnAveragePooling
        elif pooler_name == "first_n_token":
            return AxlearnFirstNTokenPooling
        else:
            raise ValueError(f"unknown {pooler_name=}")

    @staticmethod
    def _get_pytorch_pooler_cls(pooler_name: str) -> type:
        if pooler_name == "attention":
            return AttentionPooling
        elif pooler_name == "average":
            return AveragePooling
        elif pooler_name == "first_n_token":
            return FirstNTokenPooling
        else:
            raise ValueError(f"unknown {pooler_name=}")

    @parameterized.parameters(
        itertools.product(
            ("attention", "average", "first_n_token"),
            ("attention", "average", "first_n_token"),
            ("bottleneck", "cascade", "parallel"),
        )
    )
    def test_coca_image_stream_encoder(
        self, constrastive_pooler_cfg: str, caption_pooler_cfg: str, pooler_mode: str
    ):
        batch_size = 2
        num_images = 1
        image_size = 224
        patch_size = 16
        caption_pooler_num_outputs = 1 if caption_pooler_cfg == "average" else 4

        rng = np.random.RandomState(123)
        image = rng.uniform(-1, 1, [batch_size, num_images, 3, image_size, image_size]).astype(
            np.float32
        )
        torch_inputs = torch.as_tensor(image)
        axlearn_inputs = dict(image=jnp.array(image).transpose(0, 1, 3, 4, 2))

        vision_encoder_dict = {
            "num_layers": ViTModelBuilder.configs["Test16"]["num_layers"],
            "model_dim": ViTModelBuilder.configs["Test16"]["model_dim"],
            "num_heads": ViTModelBuilder.configs["Test16"]["num_heads"],
            "feed_forward_act": "nn.gelu",
            "image_size": (image_size, image_size),
            "patch_size": (patch_size, patch_size),
            "dropout_rate": 0,
        }

        axlearn_cfg = set_coca_vision_encoder_config(
            **vision_encoder_dict,
            contrastive_output_dim=1,
            caption_pooler_num_outputs=caption_pooler_num_outputs,
            contrastive_pooler_config=self._get_axlearn_pooler_cls(
                constrastive_pooler_cfg
            ).default_config(),
            caption_pooler_config=self._get_axlearn_pooler_cls(caption_pooler_cfg).default_config(),
            num_cls_tokens=1,
            layer_norm_eps=1e-6,  # layernorm in torch adapter uses 1e-6
            pooler_mode=pooler_mode,
        )

        axlearn_model = axlearn_cfg.set(name="test").instantiate(parent=None)
        axlearn_state = axlearn_model.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        axlearn_outputs, _ = F(
            axlearn_model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=axlearn_state,
            inputs=dict(input_batch=axlearn_inputs),
        )

        axlearn_constrastive_features = axlearn_outputs["output_features"]
        axlearn_caption_features = axlearn_outputs["caption_features"]

        torch_model = CoCaImageStreamEncoderBuilder.from_name(
            "Test16",
            caption_pooler_num_outputs=caption_pooler_num_outputs,
            pooler_mode=pooler_mode,
            contrastive_pooler_cls=self._get_pytorch_pooler_cls(constrastive_pooler_cfg),
            caption_pooler_cls=self._get_pytorch_pooler_cls(caption_pooler_cfg),
        ).eval()
        torch_model.load_axlearn_state_dict(create_axlearn_state_dict(axlearn_state))

        torch_outputs = torch_model(torch_inputs)
        torch_contrastive_features, torch_caption_features = torch_outputs
        torch_contrastive_features = torch_contrastive_features.detach().numpy()
        torch_caption_features = torch_caption_features.detach().numpy()

        self.assertNestedAllClose(torch_contrastive_features, axlearn_constrastive_features)
        self.assertNestedAllClose(torch_caption_features, axlearn_caption_features)


if __name__ == "__main__":
    absltest.main()
