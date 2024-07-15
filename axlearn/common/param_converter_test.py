# Copyright © 2023 Apple Inc.

"""Tests param converter utils."""
# pylint: disable=too-many-lines
import os
from typing import Any, Callable, Optional, Type

import jax
import pytest
import torch
from absl.testing import parameterized
from jax import numpy as jnp
from transformers import BertConfig, PreTrainedModel
from transformers.models.bert import modeling_bert as hf_bert
from transformers.models.deberta_v2 import modeling_deberta_v2 as hf_deberta_v2
from transformers.models.gpt2 import modeling_gpt2 as hf_gpt2
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.t5 import modeling_t5 as hf_t5
from transformers.utils import ModelOutput

from axlearn.common import bert, causal_lm, decoder, t5
from axlearn.common.attention import (
    BaseQKVLinear,
    BaseStackedTransformerLayer,
    FusedQKVLinear,
    LearnedPositionalEmbedding,
    MultiheadInputLinear,
    QKVLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.bert import BertSequenceClassificationHead
from axlearn.common.bert_test import bert_encoder_config_from_hf, dummy_inputs_for_mlm
from axlearn.common.deberta import DisentangledSelfAttention
from axlearn.common.deberta_test import (
    build_attention_cfg as deberta_build_disentangled_attention_cfg,
)
from axlearn.common.deberta_test import build_cfg as deberta_build_cfg
from axlearn.common.deberta_test import build_model_config as deberta_build_model_config
from axlearn.common.layers import (
    BaseClassificationHead,
    Embedding,
    L2Norm,
    LayerNorm,
    LayerNormStateless,
    Linear,
    set_dropout_rate_recursively,
)
from axlearn.common.module import functional as F
from axlearn.common.param_converter import (
    _parameters_from_t5x_attention,
    _parameters_from_t5x_decoder,
    _parameters_from_t5x_encoder,
    _parameters_from_t5x_ff,
    _parameters_from_t5x_layer_norm,
    _parameters_from_t5x_linear_like,
    _parameters_from_t5x_rel_pos_emb,
    _parameters_from_t5x_transformer_layer,
    as_torch_tensor,
    axlearn_to_torch,
    parameters_from_t5x_encoder_decoder,
    torch_to_axlearn,
)
from axlearn.common.t5_test import prepare_hf_t5_inputs, random_inputs_for_t5
from axlearn.common.test_utils import TestCase, dummy_padding_mask
from axlearn.common.text_dual_encoder import (
    POSITIVE_EMBEDDINGS,
    POSITIVE_INPUT_IDS,
    TextEmbeddingDualEncoder,
)
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor
from axlearn.common.utils_text_dual_encoder import bert_text_embedding_stream_encoder_config
from axlearn.huggingface import hf_text_encoder

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")
tokenizers_dir = os.path.join(testdata_dir, "tokenizers")


def torch_output_to_dict(ref_outputs):
    if isinstance(ref_outputs, ModelOutput):
        ref_outputs = dict(ref_outputs)
        for key in ref_outputs:
            ref_outputs[key] = torch_output_to_dict(ref_outputs[key])
    return ref_outputs


class BaseParamConverterTest(TestCase):
    """Base class for test methods. Do not add tests here."""

    # We test by copying params from test to ref, instead of in parent _compute_layer_outputs.
    # pylint: disable-next=arguments-differ
    def _compute_layer_outputs(
        self,
        *,
        test_layer: BaseLayer,
        ref_layer: torch.nn.Module,
        test_inputs: Any,
        ref_inputs: Any,
        parameters_to_ref_layer: Callable = axlearn_to_torch,
        method: str = "forward",
        test_torch_to_axlearn: bool = False,
    ):
        params = test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        parameters_to_ref_layer(test_layer, params, ref_layer)
        if test_torch_to_axlearn:
            params_from_ref_layer = torch_to_axlearn(ref_layer, dst_layer=test_layer)
            self.assertNestedAllClose(params_from_ref_layer, params)
        test_outputs, _ = F(
            test_layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=params,
            inputs=test_inputs,
            method=method,
        )
        ref_outputs = (
            ref_layer(**ref_inputs) if isinstance(ref_inputs, dict) else ref_layer(ref_inputs)
        )
        ref_outputs = torch_output_to_dict(ref_outputs)
        return test_outputs, jax.tree_util.tree_map(as_tensor, ref_outputs)


class ParameterTest(BaseParamConverterTest):
    def setUp(self):
        super().setUp()
        hf_cfg_kwargs = dict(
            vocab_size=24,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=10,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu_new",
            pad_token_id=0,
            bos_token_id=3,
            eos_token_id=1,
        )
        self.hf_cfg = hf_roberta.RobertaConfig(**hf_cfg_kwargs)
        self.hf_bert_cfg = hf_bert.BertConfig(**hf_cfg_kwargs)

    def _bert_model_config(
        self,
        type_vocab_size: Optional[int] = None,
        head_cfg: Optional[BaseClassificationHead] = None,
    ):
        # TODO(markblee): Consider adding a shared util to convert HF to AXLearn config, something
        # like a config_converter, to avoid the many copies/variants of this code.
        model_cfg = bert.bert_model_config(
            vocab_size=self.hf_cfg.vocab_size,
            hidden_dim=self.hf_cfg.hidden_size,
            dropout_rate=0.0,
            embedding_cfg=bert.bert_embedding_config(
                max_position_embeddings=self.hf_cfg.max_position_embeddings,
                type_vocab_size=type_vocab_size,
            ),
            stack_cfg=bert.bert_transformer_config(
                num_layers=self.hf_cfg.num_hidden_layers,
                num_heads=self.hf_cfg.num_attention_heads,
            ),
            head_cfg=head_cfg,
        )
        return model_cfg

    @parameterized.parameters(LayerNorm, LayerNormStateless)
    def test_layer_norm(self, norm_cls):
        batch, dim = 20, 10
        cfg = norm_cls.default_config().set(input_dim=dim)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        # pylint: disable-next=superfluous-parens
        hf_layer = torch.nn.LayerNorm(dim, elementwise_affine=(norm_cls == LayerNorm))

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch, dim), minval=-10, maxval=10
        )

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
            test_torch_to_axlearn=True,
        )
        self.assertNestedAllClose(out, hf_out)

    def test_linear(self):
        batch, in_dim, out_dim = 3, 10, 20
        cfg = Linear.default_config().set(input_dim=in_dim, output_dim=out_dim)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = torch.nn.Linear(in_dim, out_dim)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch, in_dim), minval=-10, maxval=10
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(out, hf_out)

    @parameterized.parameters(True, False)
    def test_multihead_input_linear(self, use_bias):
        batch, seq_len, model_dim, num_heads, per_head_dim = 5, 7, 4, 3, 6
        cfg = MultiheadInputLinear.default_config().set(
            model_dim=model_dim, num_heads=num_heads, per_head_dim=per_head_dim, bias=use_bias
        )
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = torch.nn.Linear(4, num_heads * per_head_dim, bias=use_bias)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch, seq_len, model_dim), minval=-10, maxval=10
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(out, hf_out.reshape(batch, seq_len, num_heads, per_head_dim))

    def test_embedding(self):
        batch, num_embeddings, dim = 3, 10, 20
        cfg = Embedding.default_config().set(num_embeddings=num_embeddings, dim=dim)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = torch.nn.Embedding(num_embeddings, dim)

        seq_len = 5
        inputs = jax.random.randint(
            jax.random.PRNGKey(1), shape=(batch, seq_len), minval=0, maxval=num_embeddings
        )
        # Exercise the boundary cases.
        inputs = inputs.at[:, 0].set(0)
        inputs = inputs.at[:, -1].set(num_embeddings - 1)
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(out, hf_out)

    def _hf_bert_embedding(self):
        emb_cfg = bert.bert_embedding_config(
            max_position_embeddings=self.hf_cfg.max_position_embeddings,
            layer_norm_epsilon=self.hf_cfg.layer_norm_eps,
            type_vocab_size=self.hf_cfg.type_vocab_size,
        )
        return emb_cfg.set(
            name="convert_test", dim=self.hf_cfg.hidden_size, vocab_size=self.hf_cfg.vocab_size
        ).instantiate(parent=None)

    def test_bert_embeddings(self):
        batch = 3
        layer = self._hf_bert_embedding()
        hf_layer = hf_bert.BertEmbeddings(self.hf_cfg)

        inputs = jax.random.randint(
            jax.random.PRNGKey(11),
            shape=(batch, self.hf_cfg.max_position_embeddings),
            minval=0,
            maxval=self.hf_cfg.vocab_size,
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(out, hf_out)

    def test_roberta_embeddings(self):
        # Note: Huggingface RoBERTa embeddings are slightly different from BERT:
        # Positions start at pad_token_id+1 and padding positions are explicitly masked out.
        batch = 3
        layer = self._hf_bert_embedding()
        hf_layer = hf_roberta.RobertaEmbeddings(self.hf_cfg)

        # Ensure that we are testing against LearnedPositionalEmbedding.
        self.assertIsInstance(layer.pos_emb, LearnedPositionalEmbedding)

        # Construct a "realistic" input where padding tokens only appear near the end.
        input_len = jax.random.randint(
            jax.random.PRNGKey(111),
            shape=(),
            minval=1,
            maxval=self.hf_cfg.max_position_embeddings + 1,
        )
        inputs = jax.random.randint(
            jax.random.PRNGKey(222),
            shape=(batch, input_len),
            minval=1,
            maxval=self.hf_cfg.vocab_size,
        )
        inputs = jnp.pad(inputs, [(0, 0), (0, self.hf_cfg.max_position_embeddings - input_len)])

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        # Compare only at non-padding positions.
        self.assertNestedAllClose(out[:, :input_len], hf_out[:, :input_len])

    def _param_converter_bert_encoder_config_from_hf(
        self, hf_cfg, remat
    ):  # pylint: disable=no-self-use
        if remat:
            cfg = bert_encoder_config_from_hf(
                hf_cfg, base_cfg=RepeatedTransformerLayer.default_config()
            )
        else:
            cfg = bert_encoder_config_from_hf(
                hf_cfg, base_cfg=StackedTransformerLayer.default_config()
            )
        return cfg

    @parameterized.product(
        hf_model=[hf_bert.BertAttention, hf_roberta.RobertaAttention], remat=[False, True]
    )
    def test_bert_attention(self, hf_model, remat):
        batch = 3
        cfg = self._param_converter_bert_encoder_config_from_hf(self.hf_cfg, remat)

        cfg = cfg.transformer.layer.self_attention.set(
            source_dim=self.hf_cfg.hidden_size, target_dim=self.hf_cfg.hidden_size
        )
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = hf_model(self.hf_cfg)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1),
            shape=(batch, self.hf_cfg.max_position_embeddings, self.hf_cfg.hidden_size),
            minval=-10,
            maxval=-10,
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(target=inputs),
            ref_inputs=as_torch_tensor(inputs),
        )
        # Compare attention outputs.
        self.assertNestedAllClose(out.data, hf_out[0])

    @parameterized.product(
        hf_model=[hf_bert.BertLayer, hf_roberta.RobertaLayer], remat=[False, True]
    )
    def test_bert_feed_forward(self, hf_model, remat):
        batch = 3
        cfg = self._param_converter_bert_encoder_config_from_hf(self.hf_cfg, remat)
        cfg = cfg.transformer.layer.set(input_dim=self.hf_cfg.hidden_size)
        set_dropout_rate_recursively(cfg, dropout_rate=0.0)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = hf_model(self.hf_cfg)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1),
            shape=(batch, self.hf_cfg.max_position_embeddings, self.hf_cfg.hidden_size),
            minval=-10,
            maxval=10,
        )
        params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        axlearn_to_torch(layer, params, hf_layer)
        out, _ = F(
            layer.feed_forward,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=params["feed_forward"],
            inputs=[inputs],
        )
        hf_out = hf_layer.feed_forward_chunk(as_torch_tensor(inputs))
        self.assertNestedAllClose(out, as_tensor(hf_out))

    @parameterized.product(
        hf_model=[hf_bert.BertLayer, hf_roberta.RobertaLayer], remat=[False, True]
    )
    def test_bert_layer(self, hf_model, remat):
        batch = 3
        cfg = self._param_converter_bert_encoder_config_from_hf(self.hf_cfg, remat)
        cfg = cfg.transformer.layer.set(input_dim=self.hf_cfg.hidden_size)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = hf_model(self.hf_cfg)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1),
            shape=(batch, self.hf_cfg.max_position_embeddings, self.hf_cfg.hidden_size),
            minval=-10,
            maxval=10,
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=dict(
                hidden_states=as_torch_tensor(inputs),
            ),
        )
        self.assertNestedAllClose(out.data, hf_out[0])

    @parameterized.product(
        hf_model=[hf_bert.BertEncoder, hf_roberta.RobertaEncoder],
        remat=[False, True],
        test_torch_to_axlearn=[False, True],
    )
    def test_bert_encoder(self, hf_model, remat, test_torch_to_axlearn):
        batch = 3
        cfg = self._param_converter_bert_encoder_config_from_hf(self.hf_cfg, remat)
        cfg = cfg.transformer.set(input_dim=self.hf_cfg.hidden_size)
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = hf_model(self.hf_cfg)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1),
            shape=(batch, self.hf_cfg.max_position_embeddings, self.hf_cfg.hidden_size),
            minval=-10,
            maxval=10,
        )
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=dict(
                hidden_states=as_torch_tensor(inputs),
                return_dict=False,
            ),
            test_torch_to_axlearn=test_torch_to_axlearn,
        )
        self.assertNestedAllClose(out.data, hf_out[0])

    @parameterized.parameters(
        hf_bert.BertModel,
        hf_roberta.RobertaModel,
        hf_bert.BertForSequenceClassification,
        hf_roberta.RobertaForSequenceClassification,
    )
    def test_bert_model(self, hf_model: PreTrainedModel):
        batch_size, num_classes = 3, 2
        has_seq_cls_head = hf_model in (
            hf_bert.BertForSequenceClassification,
            hf_roberta.RobertaForSequenceClassification,
        )
        head_cfg = (
            BertSequenceClassificationHead.default_config().set(num_classes=num_classes)
            if has_seq_cls_head
            else None
        )
        layer = (
            self._bert_model_config(head_cfg=head_cfg)
            .set(name="convert_test")
            .instantiate(parent=None)
        )
        hf_cfg = (
            self.hf_bert_cfg
            if hf_model in (hf_bert.BertModel, hf_bert.BertForSequenceClassification)
            else self.hf_cfg
        )
        if has_seq_cls_head:
            hf_cfg.num_label = num_classes
            hf_layer = hf_model(hf_cfg)
        else:
            hf_layer = hf_model(hf_cfg, add_pooling_layer=False)

        padding_input_id = 0
        test_inputs, hf_inputs = dummy_inputs_for_mlm(
            batch_size=batch_size,
            max_seq_len=hf_cfg.max_position_embeddings,
            vocab_size=hf_cfg.vocab_size,
            type_vocab_size=hf_cfg.type_vocab_size,
            mask_input_id=5,
            padding_input_id=padding_input_id,
            ignored_target_id=0,
        )
        if has_seq_cls_head:
            target_labels = jax.random.randint(
                jax.random.PRNGKey(432),
                shape=(batch_size,),
                minval=0,
                maxval=num_classes,
            )
            test_inputs["target_labels"] = target_labels
            hf_inputs["labels"] = as_torch_tensor(target_labels).to(torch.long)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(
                input_batch=test_inputs,
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
                token_type_ids=hf_inputs["token_type_ids"],
                return_dict=False,
            ),
        )

        if has_seq_cls_head:
            self.assertNestedAllClose(out[1]["logits"], hf_out[0])
        else:
            # Construct padding mask of shape [batch_size, max_seq_len, 1].
            non_padding = (test_inputs["input_ids"] != padding_input_id).astype(jnp.float32)
            non_padding = non_padding[..., None]
            # Compare only at non-padding positions.
            self.assertNestedAllClose(
                out[1]["sequence_output"] * non_padding,
                hf_out[0] * non_padding,
            )

    @parameterized.parameters(hf_bert.BertModel, hf_roberta.RobertaModel)
    def test_roundtrip(self, hf_model):
        """Test Hugging Face to AXLearn and back."""
        # TODO(markblee): Add back pooler after we add classification head.
        hf_layer = hf_model(self.hf_cfg, add_pooling_layer=False)
        hf_layer_copy = hf_model(self.hf_cfg, add_pooling_layer=False)
        layer = (
            self._bert_model_config(type_vocab_size=self.hf_cfg.type_vocab_size)
            .set(dim=self.hf_cfg.hidden_size, name="convert_test")
            .instantiate(parent=None)
        )

        params = parameters_from_torch_layer(hf_layer)
        axlearn_to_torch(layer, params, hf_layer_copy)

        _, hf_inputs = dummy_inputs_for_mlm(
            batch_size=3,
            max_seq_len=self.hf_cfg.max_position_embeddings,
            vocab_size=self.hf_cfg.vocab_size,
            type_vocab_size=self.hf_cfg.type_vocab_size,
            mask_input_id=5,
            padding_input_id=0,
            ignored_target_id=0,
        )
        hf_inputs = dict(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
            token_type_ids=hf_inputs["token_type_ids"],
            return_dict=False,
        )
        expected, actual = jax.tree_util.tree_map(
            as_tensor, (hf_layer(**hf_inputs), hf_layer_copy(**hf_inputs))
        )
        self.assertNestedAllClose(expected, actual)


class T5ModelConverterTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(2)

    hf_cfg = None

    def set_hf_cfg(self, arch):
        model_dim = 12
        if arch == "t5-v1":
            d_ff = 4 * model_dim
            feed_forward_proj = "relu"
        else:
            assert arch == "t5-v1-1", ValueError(f"unsupported t5 arch {arch}")
            d_ff = model_dim * 8 // 3
            feed_forward_proj = "gated-gelu"
        self.hf_cfg = hf_t5.T5Config(
            vocab_size=128,
            d_model=12,
            d_kv=4,
            d_ff=d_ff,
            num_layers=2,
            num_heads=3,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            feed_forward_proj=feed_forward_proj,
            dropout_rate=0.0,
            use_cache=False,
            tie_word_embeddings=False,
        )

    def _get_axlearn_t5_cfg(
        self, arch: str, repeat: bool, fuse_qkv: bool
    ) -> t5.T5EncoderDecoderModel.config:
        # Set base stack cfg.
        if repeat:
            base_cfg = RepeatedTransformerLayer.default_config()
        else:
            base_cfg = StackedTransformerLayer.default_config()
        # Set fused QKV if specified.
        if fuse_qkv:
            proj_cfg = FusedQKVLinear.default_config()
            base_cfg.layer.self_attention.attention.input_linear = proj_cfg
        # Create t5 encoder decoder config.
        t5_cfg = t5.t5_encoder_decoder_config(
            vocab_size=self.hf_cfg.vocab_size,
            dim=self.hf_cfg.d_model,
            num_attention_heads=self.hf_cfg.num_heads,
            num_encoder_layers=self.hf_cfg.num_layers,
            num_decoder_layers=self.hf_cfg.num_decoder_layers,
            dropout_rate=self.hf_cfg.dropout_rate,
            z_loss_scale=0,
            stack_cfg=t5.t5_transformer_stack_config(arch=arch, base_cfg=base_cfg),
        )
        return t5_cfg

    @parameterized.product(arch=["t5-v1-1", "t5-v1"], repeat=[False, True], fuse_qkv=[False, True])
    def test_roundtrip(self, arch, repeat, fuse_qkv):
        """Test Hugging Face T5 model to AXLearn and back."""
        self.set_hf_cfg(arch)
        hf_layer = hf_t5.T5ForConditionalGeneration(self.hf_cfg)
        hf_layer_copy = hf_t5.T5ForConditionalGeneration(self.hf_cfg)
        axlearn_t5_cfg = self._get_axlearn_t5_cfg(arch, repeat, fuse_qkv)
        test_layer = axlearn_t5_cfg.set(name="convert_test").instantiate(parent=None)

        params = parameters_from_torch_layer(hf_layer, dst_layer=test_layer)
        axlearn_to_torch(test_layer, params, hf_layer_copy)

        t5_inputs = random_inputs_for_t5(
            source_length=64,
            target_length=64,
            source_vocab_size=self.hf_cfg.vocab_size,
            target_vocab_size=self.hf_cfg.vocab_size,
        )
        hf_inputs = prepare_hf_t5_inputs(
            source_ids=t5_inputs["source"]["input_ids"],
            target_ids=t5_inputs["target"]["input_ids"],
            target_labels=t5_inputs["target_labels"],
        )
        expected, actual = jax.tree_util.tree_map(
            as_tensor,
            (
                hf_layer(**hf_inputs).logits,
                hf_layer_copy(**hf_inputs).logits,
            ),
        )
        self.assertNestedAllClose(expected, actual, atol=1e-5, rtol=1e-3)


class HFGPT2ModelConverterTest(TestCase):
    def setUp(self):
        super().setUp()
        self.hf_cfg = hf_gpt2.GPT2Config(
            vocab_size=64,
            n_positions=8,
            n_embd=16,
            n_layer=2,
            n_head=2,
            activation_function="relu",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-8,
            bos_token_id=1,
            eos_token_id=1,
        )

    @parameterized.parameters(True, False)
    def test_roundtrip(self, tie_word_embeddings):
        """Test Hugging Face GPT2 model to AXLearn and back."""
        self.hf_cfg.tie_word_embeddings = tie_word_embeddings
        hf_layer = hf_gpt2.GPT2LMHeadModel(self.hf_cfg)
        hf_layer_copy = hf_gpt2.GPT2LMHeadModel(self.hf_cfg)
        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=self.hf_cfg.n_layer,
            hidden_dim=self.hf_cfg.n_embd,
            num_heads=self.hf_cfg.n_head,
            vocab_size=self.hf_cfg.vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=self.hf_cfg.n_positions,
            layer_norm_epsilon=self.hf_cfg.layer_norm_epsilon,
        )
        if not self.hf_cfg.tie_word_embeddings:
            decoder_cfg = decoder_cfg.set(lm_head=decoder.LmHead.default_config())
        axlearn_gpt2_cfg = causal_lm.Model.default_config().set(decoder=decoder_cfg)
        layer = axlearn_gpt2_cfg.set(name="convert_test").instantiate(parent=None)

        params = parameters_from_torch_layer(hf_layer)
        axlearn_to_torch(layer, params, hf_layer_copy)
        input_ids = jax.random.randint(
            jax.random.PRNGKey(222),
            shape=(2, self.hf_cfg.max_position_embeddings),
            minval=1,
            maxval=self.hf_cfg.vocab_size,
            dtype=jnp.int32,
        )
        labels = jax.random.randint(
            jax.random.PRNGKey(333),
            shape=(2, self.hf_cfg.max_position_embeddings),
            minval=1,
            maxval=self.hf_cfg.vocab_size,
        )
        hf_inputs = dict(
            input_ids=as_torch_tensor(input_ids),
            labels=as_torch_tensor(labels).type(torch.LongTensor),
        )
        expected, actual = jax.tree_util.tree_map(
            as_tensor,
            (
                hf_layer(**hf_inputs).logits,
                hf_layer_copy(**hf_inputs).logits,
            ),
        )
        self.assertNestedAllClose(expected, actual)


class DeBERTaModelConverterTest(TestCase):
    @parameterized.product(
        share_projections=[True, False],
        num_directional_buckets=[8],
        max_distance=[8],
        query_len=[8],
    )
    def test_disentangled_self_attention_roundtrip(
        self,
        share_projections: bool,
        num_directional_buckets: int,
        max_distance: int,
        query_len: int,
    ):
        hf_cfg, axlearn_args = deberta_build_cfg(
            share_projections=share_projections,
            num_directional_buckets=num_directional_buckets,
            max_distance=max_distance,
            query_len=query_len,
        )

        hf_layer = hf_deberta_v2.DebertaV2Attention(hf_cfg).eval()
        hf_layer_copy = hf_deberta_v2.DebertaV2Attention(hf_cfg).eval()
        attention_cfg = TransformerAttentionLayer.default_config().set(
            name="disentangled_attn_test",
            target_dim=axlearn_args.hidden_dim,
            source_dim=axlearn_args.hidden_dim,
            attention=deberta_build_disentangled_attention_cfg(test_cfg=axlearn_args),
            structure="postnorm",
        )
        layer: DisentangledSelfAttention = attention_cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)
        axlearn_to_torch(layer=layer, src=layer_params, dst=hf_layer_copy)

        expected, actual = jax.tree_util.tree_map(
            as_tensor, (hf_layer.state_dict(), hf_layer_copy.state_dict())
        )
        self.assertNestedAllClose(expected, actual)

    @parameterized.product(
        share_projections=[True, False],
        num_directional_buckets=[None, 8],
        max_distance=[8],
        query_len=[8],
        stack_cls=[RepeatedTransformerLayer, StackedTransformerLayer],
    )
    def test_model_roundtrip(
        self,
        share_projections: bool,
        num_directional_buckets: int,
        max_distance: int,
        query_len: int,
        stack_cls: Type[BaseStackedTransformerLayer],
    ):
        hf_cfg, axlearn_args = deberta_build_cfg(
            share_projections=share_projections,
            num_directional_buckets=num_directional_buckets,
            max_distance=max_distance,
            query_len=query_len,
            num_classes=1,
            stack_cls=stack_cls,
        )
        hf_layer = hf_deberta_v2.DebertaV2Model(hf_cfg)
        hf_layer_copy = hf_deberta_v2.DebertaV2Model(hf_cfg)
        layer = deberta_build_model_config(test_cfg=axlearn_args).instantiate(parent=None)

        params = parameters_from_torch_layer(hf_layer, dst_layer=layer)
        axlearn_to_torch(layer=layer, src=params, dst=hf_layer_copy)

        # Test parameters are identical.
        expected, actual = jax.tree_util.tree_map(
            as_tensor, (hf_layer.state_dict(), hf_layer_copy.state_dict())
        )
        self.assertNestedAllClose(expected, actual)

        batch_size = 4
        input_ids = jax.random.randint(
            jax.random.PRNGKey(111),
            [batch_size, query_len],
            minval=0,
            maxval=axlearn_args.vocab_size,
        )
        hf_inputs = dict(
            input_ids=as_torch_tensor(input_ids),
            return_dict=False,
        )

        # Test we get same outputs.
        expected, actual = jax.tree_util.tree_map(
            as_tensor, (hf_layer(**hf_inputs), hf_layer_copy(**hf_inputs))
        )
        self.assertNestedAllClose(expected, actual)


class TestHFMultiStreamTextEmbeddingModel(TestCase):
    """Tests against HF MultiStreamTextEmbeddingModel."""

    def setUp(self):
        super().setUp()
        self.hf_cfg = {
            "query_encoder": BertConfig(
                vocab_size=8,
                hidden_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=8,
                type_vocab_size=2,
                hidden_act="gelu_new",
                projection_dim=0,
                add_pooling_layer=False,
                output_norm=None,
            ),
            "doc_encoder": BertConfig(
                vocab_size=8,
                hidden_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=8,
                type_vocab_size=2,
                hidden_act="gelu_new",
                projection_dim=0,
                add_pooling_layer=False,
                output_norm=None,
            ),
        }

    @parameterized.parameters((0, None), (0, hf_text_encoder.L2_NORM), (4, hf_text_encoder.L2_NORM))
    def test_axlearn_to_torch(self, projection_dim, output_norm):
        hf_stream_encoders = {}
        for encoder_name in self.hf_cfg:  #  pylint: disable=consider-using-dict-items
            hf_cfg = self.hf_cfg[encoder_name]
            hf_cfg.projection_dim = projection_dim
            hf_cfg.output_norm = output_norm
            hf_encoder_model = hf_text_encoder.BertTextEmbeddingEncoder(
                hf_cfg, add_pooling_layer=hf_cfg.add_pooling_layer
            )
            hf_encoder_model.eval()
            hf_stream_encoders[encoder_name] = hf_encoder_model
        hf_model = hf_text_encoder.MultiStreamTextEmbeddingModel(hf_stream_encoders)

        axlearn_stream_encoders = {}
        for encoder_name in self.hf_cfg:  #  pylint: disable=consider-using-dict-items
            pad_token_id = self.hf_cfg[encoder_name].vocab_size - 1
            hf_output_norm = self.hf_cfg[encoder_name].output_norm
            hidden_size = self.hf_cfg[encoder_name].hidden_size

            output_norm = None
            output_proj = None
            if hf_output_norm == hf_text_encoder.L2_NORM:
                output_norm = L2Norm.default_config()
            hf_output_proj = self.hf_cfg[encoder_name].projection_dim
            if hf_output_proj > 0:
                output_proj = Linear.default_config().set(
                    input_dim=hidden_size, output_dim=hf_output_proj, bias=True
                )
            axlearn_encoder_cfg = bert_text_embedding_stream_encoder_config(
                pad_token_id=pad_token_id,
                vocab_size=self.hf_cfg[encoder_name].vocab_size,
                max_seq_len=self.hf_cfg[encoder_name].max_position_embeddings,
                num_layers=self.hf_cfg[encoder_name].num_hidden_layers,
                num_heads=self.hf_cfg[encoder_name].num_attention_heads,
                hidden_dim=hidden_size,
                output_norm=output_norm,
                output_proj=output_proj,
                output_dim=hf_output_proj if hf_output_proj > 0 else hidden_size,
            )
            axlearn_stream_encoders[encoder_name] = axlearn_encoder_cfg

        axlearn_text_dual_encoder_cfg = TextEmbeddingDualEncoder.default_config().set(
            stream_encoder=axlearn_stream_encoders
        )
        axlearn_model = axlearn_text_dual_encoder_cfg.set(name="convert_test").instantiate(
            parent=None
        )
        params = axlearn_model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        axlearn_to_torch(axlearn_model, params, hf_model)

        hf_inputs = {}
        axlearn_inputs = {}
        batch_size = 2
        num_inputs_per_example = 4
        for encoder_name in self.hf_cfg:  #  pylint: disable=consider-using-dict-items
            max_seq_len = self.hf_cfg[encoder_name].max_position_embeddings

            attention_mask = as_torch_tensor(
                dummy_padding_mask(
                    batch_size=batch_size * num_inputs_per_example, max_seq_len=max_seq_len
                )
            )
            input_ids = torch.randint(
                0,
                self.hf_cfg[encoder_name].vocab_size - 1,
                (batch_size * num_inputs_per_example, max_seq_len),
            )
            input_ids = input_ids * attention_mask + pad_token_id * (1 - attention_mask.int())

            hf_inputs[encoder_name] = dict(
                input_ids=as_torch_tensor(input_ids),
                attention_mask=as_torch_tensor(attention_mask),
            )
            axlearn_inputs[encoder_name] = {
                POSITIVE_INPUT_IDS: as_tensor(input_ids.reshape((batch_size, -1, max_seq_len)))
            }
        out, _ = F(
            axlearn_model,
            method="predict",
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=params,
            inputs=[axlearn_inputs],
        )
        for encoder_name in self.hf_cfg:
            output_dim = axlearn_stream_encoders[encoder_name].output_dim
            self.assertNestedAllClose(
                out[encoder_name][POSITIVE_EMBEDDINGS].reshape([-1, output_dim]),
                hf_model(hf_inputs)[encoder_name],
            )


@pytest.mark.skipif(not os.path.exists(tokenizers_dir), reason="Missing testdata.")
class T5XModelConverterTest(TestCase):
    """Tests individual T5X -> AXLearn layer conversions.

    Tests are meant to be run on TPU for precision/rounding reasons.

    To simulate CI testing on CPU, use the following command:
        JAX_PLATFORMS="cpu" pytest -n 4 \
            axlearn/common/param_converter_test.py::T5XModelConverterTest

    Some tests are intended to be run on TPU.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_model = cls._model_config().set(name="t5").instantiate(parent=None)
        # For some tests, atol is set to 1e-5 for CPU. Test passses for atol=1e-6 on TPU.
        # This is due to precision/rounding errors.
        cls.atol = 1e-5 if "cpu" in str(jax.devices()[0]).lower() else 1e-6

    @classmethod
    def _model_config(cls, stack_cfg: Optional[BaseStackedTransformerLayer.Config] = None):
        # Construct a t5 v1.1 base model.
        return t5.t5_encoder_decoder_config(
            vocab_size=128,
            dim=48,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_attention_heads=4,
            dropout_rate=0,
            stack_cfg=stack_cfg,
        )

    def test_parameters_from_t5x_ff(self):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_ff.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_outputs, _ = F(
                self.test_model.decoder.transformer.layer0.feed_forward,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_ff(
                    testcase["params"],
                    # Avoid scaling to compare directly with T5X
                    src_norm=dict(scale=jnp.ones_like(testcase["inputs"])),
                ),
                inputs=dict(inputs=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    @parameterized.parameters([True, False])
    def test_parameters_from_t5x_rel_pos_emb(self, bidirectional: bool):
        testcase = jnp.load(
            os.path.join(
                testdata_dir, __name__, f"test_parameters_from_t5x_rel_pos_emb_{bidirectional}.npy"
            ),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_outputs, _ = F(
                self.test_model.encoder.relative_pos_emb
                if bidirectional
                else self.test_model.decoder.relative_pos_emb,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_rel_pos_emb(
                    testcase["params"],
                ),
                inputs=dict(attention_logit_biases=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    @parameterized.parameters([QKVLinear, FusedQKVLinear])
    def test_parameters_from_t5x_attention(self, proj_cls: Type[BaseQKVLinear]):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_attention.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            attention_cfg = self.test_model.decoder.transformer.layer0.cross_attention.config
            attention_cfg.attention.input_linear = proj_cls.default_config()
            attention_layer = attention_cfg.set(name="test").instantiate(parent=None)

            test_outputs, _ = F(
                attention_layer.attention,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_attention(
                    testcase["params"],
                    src_norm=dict(scale=jnp.ones_like(testcase["inputs"]["query"])),
                    dst_layer=attention_layer,
                )["attention"],
                inputs=testcase["inputs"],
            )

        self.assertNestedAllClose(test_outputs.data, testcase["outputs"])

    def test_parameters_from_t5x_layer_norm(self):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_layer_norm.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_outputs, _ = F(
                self.test_model.decoder.output_norm,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_layer_norm(testcase["params"]),
                inputs=dict(x=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    def test_parameters_from_t5x_dense(self):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_dense.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_model_cfg = Linear.default_config().set(
                bias=False,
                input_dim=testcase["inputs"].shape[0],
                output_dim=testcase["inputs"].shape[1],
            )
            test_model = test_model_cfg.set(name="linear").instantiate(parent=None)

            test_outputs, _ = F(
                test_model,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_linear_like(testcase["params"]),
                inputs=dict(x=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    def test_parameters_from_t5x_embedding(self):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_embedding.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            model_cfg = self.test_model.config
            test_model_cfg = Embedding.default_config().set(
                num_embeddings=model_cfg.shared_token_emb.num_embeddings,
                dim=model_cfg.shared_token_emb.dim,
            )
            test_model = test_model_cfg.set(name="embed").instantiate(parent=None)

            test_outputs, _ = F(
                test_model,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_linear_like(testcase["params"]),
                inputs=dict(x=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    def test_parameters_from_t5x_transformer_layer(self):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_transformer_layer.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_outputs, _ = F(
                self.test_model.decoder.transformer.layer0,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_transformer_layer(
                    {k: v for k, v in testcase["params"].items() if k != "relative_embedding"},
                    self.test_model.decoder.transformer.layer0,
                ),
                inputs=testcase["inputs"],
            )

        # Tolerance set to 1e-5 for CPU. Test passses for atol=1e-6 on TPU.
        self.assertNestedAllClose(test_outputs.data, testcase["outputs"], atol=self.atol)

    @parameterized.parameters(StackedTransformerLayer, RepeatedTransformerLayer)
    def test_parameters_from_t5x_decoder(self, stack_cls: Type[BaseStackedTransformerLayer]):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_decoder.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_model_cfg: t5.T5EncoderDecoderModel.Config = self._model_config(
                stack_cfg=t5.t5_transformer_stack_config(base_cfg=stack_cls.default_config())
            )
            test_model_cfg.decoder.emb.token_emb = Embedding.default_config().set(
                num_embeddings=test_model_cfg.shared_token_emb.num_embeddings,
                dim=test_model_cfg.shared_token_emb.dim,
            )
            test_model_cfg.decoder.transformer.layer.cross_attention.source_dim = (
                test_model_cfg.shared_token_emb.dim
            )
            test_model_cfg = test_model_cfg.decoder
            test_model = test_model_cfg.set(name="decoder").instantiate(parent=None)
            test_state = _parameters_from_t5x_decoder(
                {k: v for k, v in testcase["params"].items() if k != "shared_embedding"},
                test_model,
            )
            test_state.update(
                emb=dict(
                    token_emb=_parameters_from_t5x_linear_like(
                        testcase["params"]["shared_embedding"]
                    )
                )
            )
            test_outputs, _ = F(
                test_model,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=test_state,
                inputs=dict(
                    input_ids=testcase["inputs"]["target_ids"],
                    cross_attention_data=testcase["inputs"]["cross_attention_data"],
                ),
            )

        label_mask = (testcase["inputs"]["target_labels"] > 0)[:, :, None]
        self.assertGreater(label_mask.sum(), (1 - label_mask).sum())
        # Tolerance set to 1e-5 for CPU. Test passses for atol=1e-6 on TPU.
        self.assertNestedAllClose(
            test_outputs["logits"] * label_mask, testcase["outputs"] * label_mask, atol=self.atol
        )

    @parameterized.parameters([StackedTransformerLayer, RepeatedTransformerLayer])
    def test_parameters_from_t5x_encoder(self, stack_cls: Type[BaseStackedTransformerLayer]):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_encoder.npy"),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_model_cfg: t5.T5EncoderDecoderModel.Config = self._model_config(
                stack_cfg=t5.t5_transformer_stack_config(base_cfg=stack_cls.default_config())
            )
            test_model_cfg.encoder.emb.token_emb = Embedding.default_config().set(
                num_embeddings=test_model_cfg.shared_token_emb.num_embeddings,
                dim=test_model_cfg.shared_token_emb.dim,
            )
            test_model_cfg = test_model_cfg.encoder
            test_model = test_model_cfg.set(name="encoder").instantiate(parent=None)
            test_state = _parameters_from_t5x_encoder(
                {k: v for k, v in testcase["params"].items() if k != "shared_embedding"},
                test_model,
            )
            test_state.update(
                emb=dict(
                    token_emb=_parameters_from_t5x_linear_like(
                        testcase["params"]["shared_embedding"]
                    )
                )
            )
            test_outputs, _ = F(
                test_model,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=test_state,
                inputs=dict(input_ids=testcase["inputs"]["source_ids"]),
            )

        # Tolerance set to 1e-5 for CPU. Test passses for atol=1e-6 on TPU.
        source_mask = (testcase["inputs"]["source_ids"] > 0)[:, :, None]
        self.assertNestedAllClose(
            test_outputs * source_mask, testcase["outputs"] * source_mask, atol=self.atol
        )

    @parameterized.product(
        stack_cls=[StackedTransformerLayer, RepeatedTransformerLayer],
        proj_cls=[QKVLinear, FusedQKVLinear],
    )
    def test_parameters_from_t5x_encoder_decoder(
        self,
        stack_cls: Type[BaseStackedTransformerLayer],
        proj_cls: Type[BaseQKVLinear],
    ):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_parameters_from_t5x_encoder_decoder.npy"),
            allow_pickle=True,
        ).item()

        print(testcase.keys())

        with jax.default_matmul_precision("float32"):
            stack_cfg = stack_cls.default_config()
            stack_cfg.layer.self_attention.attention.input_linear = proj_cls.default_config()
            test_model_cfg: t5.T5EncoderDecoderModel.Config = self._model_config(
                stack_cfg=t5.t5_transformer_stack_config(base_cfg=stack_cfg),
            )
            test_model = test_model_cfg.set(name="t5").instantiate(parent=None)

            test_outputs, _ = F(
                test_model,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=parameters_from_t5x_encoder_decoder(testcase["params"], test_model),
                inputs=dict(
                    input_batch=dict(
                        source=dict(
                            input_ids=testcase["inputs"]["source_ids"],
                        ),
                        target=dict(
                            input_ids=testcase["inputs"]["target_ids"],
                        ),
                        target_labels=testcase["inputs"]["target_labels"],
                    ),
                ),
                method="predict",
            )

        # Tolerance set to 1e-5 for CPU. Test passses for atol=1e-6 on TPU.
        label_mask = (testcase["inputs"]["target_labels"] > 0)[:, :, None]
        self.assertGreater(label_mask.sum(), (1 - label_mask).sum())
        self.assertNestedAllClose(
            test_outputs["logits"] * label_mask, testcase["outputs"] * label_mask, atol=self.atol
        )
