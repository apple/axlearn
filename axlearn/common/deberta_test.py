# Copyright © 2023 Apple Inc.

"""Tests DeBERTa implementation."""
# pylint: disable=no-self-use
from types import SimpleNamespace
from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import parameterized
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
from transformers.models.deberta_v2 import modeling_deberta_v2 as hf_deberta_v2

from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
)
from axlearn.common.bert import BertModel, BertPooler, BertSequenceClassificationHead
from axlearn.common.deberta import (
    DeBERTaV2Encoder,
    DisentangledAttentionType,
    DisentangledSelfAttention,
    _deberta_make_log_bucket_position,
    deberta_relative_position_bucket,
    deberta_v2_encoder_config,
    deberta_v2_model_config,
    deberta_v2_self_attention_config,
)
from axlearn.common.layers import Embedding
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer


class RelativePositionTest(TestCase):
    def test_relative_position_bucket(self):
        seq_len = 10

        # When number of buckets are limited, multiple relative positions share the same bucket.
        expected = np.array(
            [
                [4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
                [5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
                [6, 5, 4, 3, 2, 1, 1, 1, 1, 1],
            ]
        )
        np.testing.assert_array_equal(
            expected,
            deberta_relative_position_bucket(
                query_len=3,
                key_len=seq_len,
                num_directional_buckets=4,
                max_distance=seq_len,
            ),
        )

        # Swap query and key.
        expected = np.array(
            [
                [4, 3, 2],
                [5, 4, 3],
                [6, 5, 4],
                [7, 6, 5],
                [7, 7, 6],
                [7, 7, 7],
                [7, 7, 7],
                [7, 7, 7],
                [7, 7, 7],
                [7, 7, 7],
            ]
        )
        np.testing.assert_array_equal(
            expected,
            deberta_relative_position_bucket(
                query_len=seq_len, key_len=3, num_directional_buckets=4, max_distance=seq_len
            ),
        )

        # When max_distance is limited, relative distances with magnitude >= max_distance share two
        # buckets.
        expected = np.array(
            [
                [5, 4, 3, 2, 2, 1, 1, 1, 1, 0],
                [6, 5, 4, 3, 2, 2, 1, 1, 1, 1],
                [7, 6, 5, 4, 3, 2, 2, 1, 1, 1],
            ]
        )
        np.testing.assert_array_equal(
            expected,
            deberta_relative_position_bucket(
                query_len=3, key_len=seq_len, num_directional_buckets=5, max_distance=5
            ),
        )

        # Swap query and key.
        expected = np.array(
            [
                [5, 4, 3],
                [6, 5, 4],
                [7, 6, 5],
                [8, 7, 6],
                [8, 8, 7],
                [9, 8, 8],
                [9, 9, 8],
                [9, 9, 9],
                [9, 9, 9],
                [10, 9, 9],
            ]
        )
        np.testing.assert_array_equal(
            expected,
            deberta_relative_position_bucket(
                query_len=seq_len, key_len=3, num_directional_buckets=5, max_distance=5
            ),
        )

    @parameterized.product(
        [
            dict(num_directional_buckets=4, max_distance=10),
            dict(num_directional_buckets=7, max_distance=9),
            dict(num_directional_buckets=10, max_distance=4),
            dict(num_directional_buckets=9, max_distance=7),
            dict(num_directional_buckets=10, max_distance=10),
        ],
        [
            dict(query_len=8, key_len=16),
            dict(query_len=16, key_len=8),
        ],
    )
    def test_relative_position_bucket_against_hf(
        self,
        *,
        num_directional_buckets: Optional[int],
        max_distance: int,
        query_len: int,
        key_len: int,
    ):
        query = jnp.arange(query_len)
        key = jnp.arange(key_len)
        relative_pos = query[:, None] - key[None, :]

        # Test make_log_bucket_position.
        test = jax.jit(
            _deberta_make_log_bucket_position,
            static_argnames=("num_directional_buckets", "max_distance"),
        )(
            relative_pos,
            num_directional_buckets=num_directional_buckets,
            max_distance=max_distance,
        )
        relative_pos = np.asarray(relative_pos)
        relative_pos = torch.from_numpy(relative_pos)
        ref = hf_deberta_v2.make_log_bucket_position(
            relative_pos, num_directional_buckets, max_distance
        )
        np.testing.assert_array_equal(test, ref)

        # Test relative_position_bucket.
        test = jax.jit(
            deberta_relative_position_bucket,
            static_argnames=("query_len", "key_len", "num_directional_buckets", "max_distance"),
        )(
            query_len=query_len,
            key_len=key_len,
            num_directional_buckets=num_directional_buckets,
            max_distance=max_distance,
        )
        ref = hf_deberta_v2.build_relative_position(
            query_len, key_len, num_directional_buckets, max_distance
        )
        # Our implementation returns values in [0, 2*num_directional_buckets], similar to T5, but
        # Hugging Face returns [-num_directional_buckets, num_directional_buckets].
        np.testing.assert_array_equal(test, (ref + num_directional_buckets).squeeze(0))


def build_cfg(
    *,
    share_projections: bool,
    max_distance: int,
    query_len: int,
    vocab_size: int = 26,
    hidden_dim: int = 32,
    num_heads: int = 8,
    num_layers: int = 12,
    num_directional_buckets: Optional[int] = None,
    position_biased_input: bool = True,  # When false, position embeddings are not used.
    num_classes: int = 2,  # Only used for sequence classification.
    stack_cls: Optional[Type[BaseStackedTransformerLayer]] = None,
):
    """Build ref and test flat-configs."""
    attention_type = [DisentangledAttentionType.C2P, DisentangledAttentionType.P2C]
    ref_cfg = hf_deberta_v2.DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_buckets=num_directional_buckets or max_distance,
        max_relative_positions=max_distance,
        position_biased_input=position_biased_input,
        max_position_embeddings=query_len,
        relative_attention=len(attention_type) > 0,
        pos_att_type=[t.name.lower() for t in attention_type],
        share_att_key=share_projections,
        intermediate_size=hidden_dim * 4,
        norm_rel_ebd="layer_norm",
        pad_token_id=0,
        pooler_hidden_act="gelu",
        num_labels=num_classes,
    )
    test_cfg = SimpleNamespace(
        vocab_size=ref_cfg.vocab_size,
        hidden_dim=ref_cfg.hidden_size,
        num_heads=ref_cfg.num_attention_heads,
        num_layers=ref_cfg.num_hidden_layers,
        attention_type=[DisentangledAttentionType[t.upper()] for t in ref_cfg.pos_att_type],
        max_distance=max_distance,
        num_directional_buckets=num_directional_buckets,
        num_pos_emb=ref_cfg.position_buckets,
        attention_dropout=ref_cfg.attention_probs_dropout_prob,
        hidden_dropout=ref_cfg.hidden_dropout_prob,
        share_projections=ref_cfg.share_att_key,
        pad_token_id=ref_cfg.pad_token_id,
        max_position_embeddings=ref_cfg.max_position_embeddings if position_biased_input else None,
        pooler_hidden_act="exact_gelu",
        num_classes=num_classes,
        stack_cls=stack_cls,
    )
    return ref_cfg, test_cfg


def build_attention_cfg(*, test_cfg: SimpleNamespace) -> DisentangledSelfAttention.Config:
    # Build our config.
    cfg = deberta_v2_self_attention_config(
        num_heads=test_cfg.num_heads,
        max_distance=test_cfg.max_distance,
        hidden_dropout=test_cfg.hidden_dropout,
        attention_dropout=test_cfg.attention_dropout,
        share_projections=test_cfg.share_projections,
        num_directional_buckets=test_cfg.num_directional_buckets,
        attention_type=test_cfg.attention_type,
    )
    cfg.set(
        query_dim=test_cfg.hidden_dim,
        key_dim=test_cfg.hidden_dim,
        value_dim=test_cfg.hidden_dim,
        pos_emb=Embedding.default_config().set(
            num_embeddings=2 * test_cfg.num_pos_emb,
            dim=test_cfg.hidden_dim,
        ),
    )
    return cfg


# Parameterize all tests in this class.
@parameterized.product(
    share_projections=[True, False],
    num_directional_buckets=[None, 32, 25, 8],
    max_distance=[64, 32, 25, 8],
    # TODO(markblee): Test query_len != key_len.
    query_len=[16, 8],
)
class DisentangledSelfAttentionTest(TestCase):
    def test_disentangled_attention_bias(
        self,
        query_len: int,
        **kwargs,
    ):
        """Test DisentangledSelfAttention.disentangled_attention_bias."""
        batch_size = 4
        key_len = query_len
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_attention_cfg(test_cfg=test_cfg).set(name="test")

        hf_layer = hf_deberta_v2.DebertaV2Attention(ref_cfg).eval()
        layer: DisentangledSelfAttention = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)
        layer_params = layer_params["attention"]

        # Build relative positions.
        rel_pos = deberta_relative_position_bucket(
            query_len=query_len,
            key_len=key_len,
            num_directional_buckets=layer.num_directional_buckets(),
            max_distance=cfg.max_distance,
        )
        rel_pos_hf = hf_deberta_v2.build_relative_position(
            query_len,
            key_len,
            bucket_size=layer.num_directional_buckets(),
            max_position=cfg.max_distance,
        )

        # Generate some dummy inputs.
        relative_pos_emb = jax.random.uniform(
            jax.random.PRNGKey(321),
            (2 * layer.num_directional_buckets(), layer.hidden_dim()),
            minval=0,
            maxval=10,
        )
        q_proj = jax.random.uniform(
            jax.random.PRNGKey(111),
            [batch_size, query_len, cfg.num_heads, layer.per_head_dim()],
            minval=-10,
            maxval=10,
        )
        k_proj = jax.random.uniform(
            jax.random.PRNGKey(222),
            [batch_size, key_len, cfg.num_heads, layer.per_head_dim()],
            minval=-10,
            maxval=10,
        )
        # Build inputs for HF.
        query_hf = (
            as_torch_tensor(q_proj).transpose(1, 2).reshape((-1, query_len, layer.per_head_dim()))
        )
        key_hf = (
            as_torch_tensor(k_proj).transpose(1, 2).reshape((-1, key_len, layer.per_head_dim()))
        )

        # Patch in the pos_emb weights.
        layer_params["pos_emb"] = dict(weight=relative_pos_emb)

        # Test just the disentangled attention bias output.
        layer_outputs, _ = F(
            layer,
            method="_disentangled_attention_bias",
            inputs=dict(
                q_proj=q_proj,
                k_proj=k_proj,
                relative_pos_emb=relative_pos_emb,
                relative_pos=rel_pos[None, ...],
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
        )
        ref_outputs = hf_layer.self.disentangled_attention_bias(
            query_hf,
            key_hf,
            rel_pos_hf,
            as_torch_tensor(relative_pos_emb),
            len(cfg.attention_type) + 1,
        )
        ref_outputs = ref_outputs.detach().reshape((batch_size, cfg.num_heads, query_len, key_len))

        # Occasionally, we may see a tiny fraction of outputs (e.g. 1/8192) exceed 1e-6.
        # This is likely due to fp error so we increase threshold to 1e-5.
        assert_allclose(layer_outputs, ref_outputs, atol=1e-5)

        # Test the output matches if we don't explicitly pass relative_pos_emb, relative_pos,
        # assuming no dropout on the embeddings.
        layer_outputs_pos_emb, _ = F(
            layer,
            method="_disentangled_attention_bias",
            inputs=dict(q_proj=q_proj, k_proj=k_proj),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
        )
        assert_allclose(layer_outputs, layer_outputs_pos_emb)

    def test_disentangled_self_attention(
        self,
        query_len: int,
        **kwargs,
    ):
        """Test the DisentangledSelfAttention layer output probs."""
        batch_size = 4
        key_len = query_len
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_attention_cfg(test_cfg=test_cfg).set(name="test")

        hf_layer = hf_deberta_v2.DebertaV2Attention(ref_cfg).eval()
        layer: DisentangledSelfAttention = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)
        layer_params = layer_params["attention"]

        # Generate some dummy inputs.
        query = jax.random.uniform(
            jax.random.PRNGKey(111),
            [batch_size, query_len, layer.hidden_dim()],
            minval=-10,
            maxval=10,
        )
        key = jax.random.uniform(
            jax.random.PRNGKey(222),
            [batch_size, key_len, layer.hidden_dim()],
            minval=-10,
            maxval=10,
        )
        attention_logit_biases = jnp.zeros([batch_size, 1, query_len, key_len], dtype=jnp.float32)
        relative_pos_emb = jax.random.uniform(
            jax.random.PRNGKey(333),
            [cfg.pos_emb.num_embeddings, cfg.pos_emb.dim],
        )
        return_aux = {"probs"}

        # Patch in the pos_emb weights.
        layer_params["pos_emb"] = dict(weight=relative_pos_emb)

        # TODO(markblee): Test passing in relative_pos_emb, relative_pos in attention forward.
        layer_outputs, _ = F(
            layer,
            inputs=dict(
                query=query,
                key=key,
                value=key,
                attention_logit_biases=attention_logit_biases,
                return_aux=return_aux,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
        )
        ref_outputs = hf_layer(
            as_torch_tensor(key),
            as_torch_tensor(1 - attention_logit_biases),
            query_states=as_torch_tensor(query),
            rel_embeddings=as_torch_tensor(relative_pos_emb),
            output_attentions=True,
        )
        # [batch, num_heads, query_len, key_len].
        _, ref_probs = ref_outputs

        # Note: Only compare self attention probs here. HF DebertaV2Attention is comparable to
        # TransformerAttentionLayer since it also applies dropout + residual + norm.
        assert_allclose(layer_outputs.probs, ref_probs.detach())


def build_encoder_config(*, test_cfg: SimpleNamespace) -> DeBERTaV2Encoder.Config:
    cfg = deberta_v2_encoder_config(
        dim=test_cfg.hidden_dim,
        vocab_size=test_cfg.vocab_size,
        num_layers=test_cfg.num_layers,
        max_distance=test_cfg.max_distance,
        max_position_embeddings=test_cfg.max_position_embeddings,
        num_directional_buckets=test_cfg.num_directional_buckets,
        stack_cls=test_cfg.stack_cls,
    )
    cfg.transformer.layer.self_attention.attention = deberta_v2_self_attention_config(
        base_cfg=cfg.transformer.layer.self_attention.attention,
        num_heads=test_cfg.num_heads,
        max_distance=test_cfg.max_distance,
        hidden_dropout=test_cfg.hidden_dropout,
        attention_dropout=test_cfg.attention_dropout,
        share_projections=test_cfg.share_projections,
        num_directional_buckets=test_cfg.num_directional_buckets,
        attention_type=test_cfg.attention_type,
    )
    return cfg.set(name="encoder", pad_token_id=test_cfg.pad_token_id)


# Parameterize all tests in this class.
@parameterized.product(
    share_projections=[True, False],
    num_directional_buckets=[None, 32, 25, 8],
    max_distance=[64, 32, 25, 8],
    # TODO(markblee): Test query_len != key_len.
    query_len=[16, 8],
    stack_cls=[StackedTransformerLayer, RepeatedTransformerLayer],
)
class DeBERTaEncoderTest(TestCase):
    def test_emb(self, query_len: int, **kwargs):
        batch_size = 4
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_encoder_config(test_cfg=test_cfg)

        hf_layer = hf_deberta_v2.DebertaV2Model(ref_cfg).eval()
        layer: DeBERTaV2Encoder = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, query_len], minval=0, maxval=test_cfg.vocab_size
        )
        test_outputs, _ = F(
            layer.emb,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
            state=layer_params["encoder"]["emb"],
            inputs=[input_ids],
        )
        ref_outputs = hf_layer.embeddings(as_torch_tensor(input_ids))
        self.assertNestedAllClose(test_outputs, ref_outputs)

    def test_rel_emb(self, query_len: int, **kwargs):
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_encoder_config(test_cfg=test_cfg)

        hf_layer = hf_deberta_v2.DebertaV2Model(ref_cfg).eval()
        layer: DeBERTaV2Encoder = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)

        test_outputs, _ = F(
            layer.relative_pos_emb,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
            state=layer_params["encoder"]["relative_pos_emb"],
            inputs=[],
            method="embeddings",
        )
        ref_outputs = hf_layer.encoder.get_rel_embedding()
        self.assertNestedAllClose(test_outputs, ref_outputs)

    def test_layers(self, query_len: int, **kwargs):
        batch_size = 4
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_encoder_config(test_cfg=test_cfg)

        hf_layer = hf_deberta_v2.DebertaV2Model(ref_cfg).eval()
        layer: DeBERTaV2Encoder = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, query_len], minval=0, maxval=test_cfg.vocab_size
        )
        padding_mask = input_ids != cfg.pad_token_id
        test_out, _ = F(
            layer,
            jax.random.PRNGKey(0),
            layer_params["encoder"],
            [input_ids],
            is_training=False,
            drop_output_collections=[],
        )
        ref_out = hf_layer(
            as_torch_tensor(input_ids),
            attention_mask=as_torch_tensor(padding_mask),
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        output_mask = padding_mask[:, :, None]
        self.assertNestedAllClose(
            ref_out.hidden_states[-1] * as_torch_tensor(output_mask),
            test_out * output_mask,
            atol=1e-5,
            rtol=1e-2,
        )

    def test_encoder(self, query_len: int, **kwargs):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        batch_size = 4
        ref_cfg, test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_encoder_config(test_cfg=test_cfg)

        hf_layer = hf_deberta_v2.DebertaV2Model(ref_cfg).eval()
        layer: DeBERTaV2Encoder = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_layer, dst_layer=layer)
        layer_params = layer_params["encoder"]

        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, query_len], minval=0, maxval=test_cfg.vocab_size
        )
        padding_mask = input_ids != cfg.pad_token_id
        test_outputs, _ = F(
            layer,
            inputs=[input_ids],
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
        )
        ref_outputs = hf_layer(
            as_torch_tensor(input_ids),
            attention_mask=as_torch_tensor(padding_mask),
            return_dict=True,
        )
        padding_mask = padding_mask[..., None]
        self.assertNestedAllClose(
            test_outputs * padding_mask,
            ref_outputs["last_hidden_state"] * as_torch_tensor(padding_mask),
        )


def build_model_config(*, test_cfg: SimpleNamespace) -> BertModel.Config:
    encoder_cfg = build_encoder_config(test_cfg=test_cfg)
    cfg = deberta_v2_model_config(
        encoder_cfg=encoder_cfg,
        head_cfg=BertSequenceClassificationHead.default_config().set(
            num_classes=test_cfg.num_classes,
            pooler=BertPooler.default_config().set(
                activation=test_cfg.pooler_hidden_act,
            ),
        ),
    )

    return cfg.set(name="deberta_model")


class DeBERTaModelTest(TestCase):
    def test_context_pooler(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        hf_config = DebertaV2Config()
        hf_context_pooler = hf_deberta_v2.ContextPooler(hf_config).eval()
        layer = (
            BertPooler.default_config()
            .set(name="pooler", input_dim=hf_config.hidden_size, activation="exact_gelu")
            .instantiate(parent=None)
        )

        input_ids = jax.random.uniform(jax.random.PRNGKey(111), [8, 64, hf_config.hidden_size])
        test_outputs, ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_context_pooler,
            test_inputs=dict(inputs=input_ids),
            ref_inputs=as_torch_tensor(input_ids),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        self.assertNestedAllClose(test_outputs, ref_outputs)

    @parameterized.parameters(
        dict(method="predict"),
        dict(method="forward"),
    )
    def test_deberta_v2_sequence_classification(self, method: str):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        batch_size, seq_len = 4, 64
        # We test this by using the head together with the encoder.
        hf_config, test_cfg = build_cfg(
            share_projections=True, max_distance=seq_len, query_len=seq_len, num_classes=2
        )
        hf_config.problem_type = "single_label_classification"
        cfg = build_model_config(test_cfg=test_cfg)
        hf_seq_classification_model = DebertaV2ForSequenceClassification(hf_config).eval()
        model: BertModel = cfg.instantiate(parent=None)
        layer_params = parameters_from_torch_layer(hf_seq_classification_model, dst_layer=model)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, seq_len], minval=0, maxval=test_cfg.vocab_size
        )
        padding_mask = input_ids != cfg.encoder.pad_token_id
        token_type_ids = jnp.where(
            jnp.tile(jnp.arange(seq_len), [batch_size, 1])
            >= jnp.tile(
                jax.random.randint(
                    jax.random.PRNGKey(112), [batch_size, 1], minval=0, maxval=seq_len - 1
                ),
                [1, seq_len],
            ),
            1,
            0,
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), [batch_size], minval=0, maxval=1
        )
        input_batch = dict(
            input_ids=input_ids, token_type_ids=token_type_ids, target_labels=target_labels
        )
        test_outputs, _ = F(
            model,
            inputs=[input_batch],
            state=layer_params,
            is_training=False,
            method=method,
            prng_key=jax.random.PRNGKey(1),
        )
        ref_outputs = hf_seq_classification_model(
            as_torch_tensor(input_ids),
            attention_mask=as_torch_tensor(padding_mask),
            labels=as_torch_tensor(target_labels).type(torch.LongTensor),
            output_hidden_states=True,
            return_dict=True,
        )
        padding_mask = padding_mask[..., None]
        if method == "predict":
            self.assertNestedAllClose(
                test_outputs["sequence_output"] * padding_mask,
                ref_outputs["hidden_states"][-1] * as_torch_tensor(padding_mask),
            )
            self.assertNestedAllClose(test_outputs["logits"], ref_outputs["logits"])
        elif method == "forward":
            self.assertAlmostEqual(test_outputs[0].item(), ref_outputs["loss"].item(), places=6)
