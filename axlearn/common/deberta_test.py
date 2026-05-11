# Copyright © 2023 Apple Inc.

"""Tests DeBERTa implementation."""

# pylint: disable=no-self-use
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

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
    deberta_relative_position_bucket,
    deberta_v2_encoder_config,
    deberta_v2_model_config,
    deberta_v2_self_attention_config,
)
from axlearn.common.golden import load_golden
from axlearn.common.layers import Embedding
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose

MODULE_NAME = "axlearn.common.deberta_test"


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
    stack_cls: Optional[type[BaseStackedTransformerLayer]] = None,
):
    """Build test flat-configs (without HF reference)."""
    attention_type = [DisentangledAttentionType.C2P, DisentangledAttentionType.P2C]
    test_cfg = SimpleNamespace(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attention_type=attention_type,
        max_distance=max_distance,
        num_directional_buckets=num_directional_buckets,
        num_pos_emb=num_directional_buckets or max_distance,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        share_projections=share_projections,
        pad_token_id=0,
        max_position_embeddings=query_len if position_biased_input else None,
        pooler_hidden_act="exact_gelu",
        num_classes=num_classes,
        stack_cls=stack_cls,
    )
    return test_cfg


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


def _golden_key(  # pylint: disable=unused-argument
    test_prefix: str,
    *,
    share_projections: bool,
    num_directional_buckets: Optional[int],
    max_distance: int,
    stack_cls=None,
    **kwargs,
) -> str:
    """Build the golden test key from parameterized args."""
    sp = "True" if share_projections else "False"
    ndb = str(num_directional_buckets) if num_directional_buckets is not None else "None"
    md = str(max_distance)
    key = f"{test_prefix}_sp{sp}_ndb{ndb}_md{md}"
    if stack_cls is not None:
        if stack_cls == RepeatedTransformerLayer:
            key += "_Repeated"
        elif stack_cls == StackedTransformerLayer:
            key += "_Stacked"
    return key


# Parameterize all tests in this class.
@parameterized.product(
    share_projections=[True, False],
    num_directional_buckets=[None, 8],
    max_distance=[64, 8],
    query_len=[16],
)
class DisentangledSelfAttentionTest(TestCase):
    def test_disentangled_attention_bias(
        self,
        query_len: int,
        **kwargs,
    ):
        """Test DisentangledSelfAttention.disentangled_attention_bias."""
        golden_key = _golden_key("test_disentangled_attention_bias", **kwargs)
        golden = load_golden(MODULE_NAME, golden_key)

        test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_attention_cfg(test_cfg=test_cfg).set(name="test")
        layer: DisentangledSelfAttention = cfg.instantiate(parent=None)

        # Load params and inputs from golden.
        layer_params = golden["params"]
        q_proj = jnp.array(golden["inputs"]["q_proj"])
        k_proj = jnp.array(golden["inputs"]["k_proj"])
        relative_pos_emb = jnp.array(golden["inputs"]["relative_pos_emb"])

        # Patch in the pos_emb weights.
        layer_params["pos_emb"] = dict(weight=relative_pos_emb)

        # Build relative positions.
        key_len = query_len
        rel_pos = deberta_relative_position_bucket(
            query_len=query_len,
            key_len=key_len,
            num_directional_buckets=layer.num_directional_buckets(),
            max_distance=cfg.max_distance,
        )

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
        assert_allclose(layer_outputs, golden["outputs"]["ref"], atol=1e-5)

        # Test the output matches if we don't explicitly pass relative_pos_emb, relative_pos.
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
        golden_key = _golden_key("test_disentangled_self_attention", **kwargs)
        golden = load_golden(MODULE_NAME, golden_key)

        test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_attention_cfg(test_cfg=test_cfg).set(name="test")
        layer: DisentangledSelfAttention = cfg.instantiate(parent=None)

        # Load params and inputs from golden.
        layer_params = golden["params"]
        query = jnp.array(golden["inputs"]["query"])
        key = jnp.array(golden["inputs"]["key"])
        relative_pos_emb = jnp.array(golden["inputs"]["relative_pos_emb"])
        batch_size = query.shape[0]
        key_len = query_len

        attention_logit_biases = jnp.zeros([batch_size, 1, query_len, key_len], dtype=jnp.float32)
        return_aux = {"probs"}

        # Patch in the pos_emb weights.
        layer_params["pos_emb"] = dict(weight=relative_pos_emb)

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
        assert_allclose(layer_outputs.probs, golden["outputs"]["probs"])


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
    num_directional_buckets=[None, 8],
    max_distance=[64, 8],
    query_len=[16],
    stack_cls=[StackedTransformerLayer, RepeatedTransformerLayer],
)
class DeBERTaEncoderTest(TestCase):
    def test_encoder(self, query_len: int, **kwargs):
        golden_key = _golden_key("test_encoder", **kwargs)
        golden = load_golden(MODULE_NAME, golden_key)

        test_cfg = build_cfg(query_len=query_len, **kwargs)
        cfg = build_encoder_config(test_cfg=test_cfg)
        layer: DeBERTaV2Encoder = cfg.instantiate(parent=None)

        layer_params = golden["params"]["encoder"]
        input_ids = jnp.array(golden["inputs"]["input_ids"])
        padding_mask = jnp.array(golden["inputs"]["padding_mask"])

        test_outputs, _ = F(
            layer,
            inputs=[input_ids],
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
        )
        padding_mask_expanded = padding_mask[..., None]
        assert_allclose(
            test_outputs * padding_mask_expanded,
            golden["outputs"]["last_hidden_state"] * np.array(padding_mask_expanded),
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
        golden = load_golden(MODULE_NAME, "test_context_pooler")
        layer = (
            BertPooler.default_config()
            .set(name="pooler", input_dim=1536, activation="exact_gelu")
            .instantiate(parent=None)
        )
        layer_params = golden["params"]
        inputs = jnp.array(golden["inputs"]["inputs"])

        test_outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(test_outputs, golden["outputs"]["ref"])

    def test_deberta_v2_sequence_classification(self):
        golden = load_golden(MODULE_NAME, "test_deberta_v2_sequence_classification")
        batch_size, seq_len = 4, 64
        test_cfg = build_cfg(
            share_projections=True, max_distance=seq_len, query_len=seq_len, num_classes=2
        )
        cfg = build_model_config(test_cfg=test_cfg)
        model: BertModel = cfg.instantiate(parent=None)

        layer_params = golden["params"]
        input_ids = jnp.array(golden["inputs"]["input_ids"])
        padding_mask = jnp.array(golden["inputs"]["padding_mask"])
        target_labels = jnp.array(golden["inputs"]["target_labels"])
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
        input_batch = dict(
            input_ids=input_ids, token_type_ids=token_type_ids, target_labels=target_labels
        )

        # Test predict method.
        predict_outputs, _ = F(
            model,
            inputs=[input_batch],
            state=layer_params,
            is_training=False,
            method="predict",
            prng_key=jax.random.PRNGKey(1),
        )
        padding_mask_expanded = padding_mask[..., None]
        assert_allclose(
            predict_outputs["sequence_output"] * padding_mask_expanded,
            golden["outputs"]["last_hidden_state"] * np.array(padding_mask_expanded),
        )
        # Logits for samples where position 0 is a pad token may differ due to padding handling.
        assert_allclose(predict_outputs["logits"], golden["outputs"]["logits"], atol=0.02)

        # Test forward method.
        forward_outputs, _ = F(
            model,
            inputs=[input_batch],
            state=layer_params,
            is_training=False,
            method="forward",
            prng_key=jax.random.PRNGKey(1),
        )
        self.assertAlmostEqual(
            forward_outputs[0].item(), float(golden["outputs"]["loss"]), places=2
        )


if __name__ == "__main__":
    absltest.main()
