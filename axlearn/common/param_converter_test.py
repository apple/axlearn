# Copyright © 2023 Apple Inc.

"""Tests param converter utils."""

# pylint: disable=too-many-lines
import os
from typing import Any, Callable, Optional

import jax
import pytest
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from transformers import PreTrainedModel
from transformers.models.bert import modeling_bert as hf_bert
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.utils import ModelOutput

from axlearn.common import bert, t5
from axlearn.common.attention import (
    BaseQKVLinear,
    BaseStackedTransformerLayer,
    FusedQKVLinear,
    LearnedPositionalEmbedding,
    MultiheadInputLinear,
    QKVLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.bert import BertSequenceClassificationHead
from axlearn.common.bert_test import bert_encoder_config_from_hf, dummy_inputs_for_mlm
from axlearn.common.ein_ops import rearrange
from axlearn.common.layers import (
    BaseClassificationHead,
    Conv3D,
    Embedding,
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
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")
_MODULE_NAME = "axlearn.common.param_converter_test"
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
    def _compute_layer_outputs(  # pytype: disable=signature-mismatch
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
            self.assertEqual(
                jax.tree_util.tree_structure(params_from_ref_layer),
                jax.tree_util.tree_structure(params),
            )
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
        return test_outputs, jax.tree.map(as_tensor, ref_outputs)


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
        self.hf_cfg = hf_roberta.RobertaConfig(attn_implementation="eager", **hf_cfg_kwargs)
        self.hf_bert_cfg = hf_bert.BertConfig(attn_implementation="eager", **hf_cfg_kwargs)

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

    @parameterized.product(test_torch_to_axlearn=[True, False], bias=[True, False])
    def test_conv3d(self, test_torch_to_axlearn, bias):
        b, t, h, w, c, out_c, k = 2, 3, 4, 5, 6, 8, 3
        cfg = Conv3D.default_config().set(
            input_dim=c, output_dim=out_c, window=(k, k, k), bias=bias
        )
        layer = cfg.set(name="convert_test").instantiate(parent=None)
        hf_layer = torch.nn.Conv3d(c, out_c, k, bias=bias)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(b, t, h, w, c), minval=-10, maxval=10
        )
        hf_inputs = rearrange(inputs, "b t h w c -> b c t h w")
        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(hf_inputs),
            test_torch_to_axlearn=test_torch_to_axlearn,
        )
        hf_out = rearrange(hf_out, "b c t h w -> b t h w c")
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
            test_inputs=dict(input_batch=dict(inputs=inputs)),
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
            test_inputs=dict(input_batch=dict(inputs=inputs)),
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
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_ff.npy"),
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
                testdata_dir,
                _MODULE_NAME,
                f"test_parameters_from_t5x_rel_pos_emb_{bidirectional}.npy",
            ),
            allow_pickle=True,
        ).item()

        with jax.default_matmul_precision("float32"):
            test_outputs, _ = F(
                (
                    self.test_model.encoder.relative_pos_emb
                    if bidirectional
                    else self.test_model.decoder.relative_pos_emb
                ),
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=_parameters_from_t5x_rel_pos_emb(
                    testcase["params"],
                ),
                inputs=dict(attention_logit_biases=testcase["inputs"]),
            )

        self.assertNestedAllClose(test_outputs, testcase["outputs"])

    @parameterized.parameters([QKVLinear, FusedQKVLinear])
    def test_parameters_from_t5x_attention(self, proj_cls: type[BaseQKVLinear]):
        testcase = jnp.load(
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_attention.npy"),
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
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_layer_norm.npy"),
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
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_dense.npy"),
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
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_embedding.npy"),
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
            os.path.join(
                testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_transformer_layer.npy"
            ),
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
    def test_parameters_from_t5x_decoder(self, stack_cls: type[BaseStackedTransformerLayer]):
        testcase = jnp.load(
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_decoder.npy"),
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
    def test_parameters_from_t5x_encoder(self, stack_cls: type[BaseStackedTransformerLayer]):
        testcase = jnp.load(
            os.path.join(testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_encoder.npy"),
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
        stack_cls: type[BaseStackedTransformerLayer],
        proj_cls: type[BaseQKVLinear],
    ):
        testcase = jnp.load(
            os.path.join(
                testdata_dir, _MODULE_NAME, "test_parameters_from_t5x_encoder_decoder.npy"
            ),
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


if __name__ == "__main__":
    absltest.main()
