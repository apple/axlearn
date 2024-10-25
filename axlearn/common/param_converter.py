# Copyright © 2023 Apple Inc.

# pylint: disable=too-many-lines
"""Utilities to convert parameters from one library to another.

For example, utilities to convert AXLearn parameters to/from torch modules live here.
"""

from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from typing import Any, Optional, Protocol, Union, cast

import jax
import numpy as np
import torch
from jax import numpy as jnp
from timm.models import vision_transformer as timm_vit
from transformers.models.bert import modeling_bert as hf_bert
from transformers.models.deberta_v2 import modeling_deberta_v2 as hf_deberta_v2
from transformers.models.distilbert import modeling_distilbert as hf_distilbert
from transformers.models.dpr import modeling_dpr as hf_dpr
from transformers.models.encoder_decoder import modeling_encoder_decoder as hf_encoder_decoder
from transformers.models.gpt2 import modeling_gpt2 as hf_gpt2
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mt5 import modeling_mt5 as hf_mt5
from transformers.models.opt import modeling_opt as hf_opt
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.roformer import modeling_roformer as hf_roformer
from transformers.models.t5 import modeling_t5 as hf_t5
from transformers.models.vit import modeling_vit as hf_vit
from transformers.models.vit_mae import modeling_vit_mae as hf_vit_mae
from transformers.models.xlnet import modeling_xlnet as hf_xlnet

from axlearn.common.attention import (
    BaseMultiheadLinear,
    BaseStackedTransformerLayer,
    FusedQKVLinear,
    LearnedPositionalEmbedding,
    MultiheadAttention,
    MultiheadInputLinear,
    QKVLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.bert import BertModel, BertPooler, BertSequenceClassificationHead
from axlearn.common.causal_lm import Model as CausalLMModel
from axlearn.common.deberta import DeBERTaV2Encoder
from axlearn.common.decoder import Decoder
from axlearn.common.dit import (
    AdaptiveLayerNormModulation,
    DiTAttentionLayer,
    DiTBlock,
    DiTFeedForwardLayer,
    DiTFinalLayer,
    LabelEmbedding,
    TimeStepEmbedding,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.encoder import Encoder, EncoderModel
from axlearn.common.layers import Conv2D, Embedding, LayerNorm, LayerNormStateless, Linear, RMSNorm
from axlearn.common.t5 import T5Decoder, T5Encoder, T5EncoderDecoderModel
from axlearn.common.text_encoder import TextEmbeddingEncoder
from axlearn.common.utils import NestedTensor, Tensor, VDict, as_tensor
from axlearn.huggingface import hf_text_encoder

NestedTorchTensor = Union[torch.Tensor, dict[str, Any]]


class AXLearnToTorchFn(Protocol):
    def __call__(self, *, src: BaseLayer, params: NestedTensor, dst: torch.nn.Module):
        ...


class TorchToAXLearnFn(Protocol):
    def __call__(self, *, src: torch.nn.Module, dst: Optional[Union[BaseLayer, type]]):
        ...


# Mapping from (src_type, dst_type) to converter.
_axlearn_to_torch_registry: dict[tuple[type, type], AXLearnToTorchFn] = {}
_torch_to_axlearn_registry: dict[tuple[type, type], TorchToAXLearnFn] = {}


def register_axlearn_to_torch(*, src: Sequence[type], dst: Sequence[type]):
    """Registers a converter for converting axlearn layers to torch."""

    def decorator(fn: AXLearnToTorchFn):
        for src_type in src:
            for dst_type in dst:
                if (src_type, dst_type) in _axlearn_to_torch_registry:
                    raise ValueError(f"Converter for {src_type} -> {dst_type} already registered.")
                _axlearn_to_torch_registry[(src_type, dst_type)] = fn
        return fn

    return decorator


def register_torch_to_axlearn(*, src: Sequence[type], dst: Sequence[type] = (type(None),)):
    """Registers a converter for converting torch layers to axlearn."""

    def decorator(fn: TorchToAXLearnFn):
        for src_type in src:
            for dst_type in dst:
                if (src_type, dst_type) in _torch_to_axlearn_registry:
                    raise ValueError(f"Converter for {src_type} -> {dst_type} already registered.")
                _torch_to_axlearn_registry[(src_type, dst_type)] = fn
        return fn

    return decorator


def as_torch_tensor(x: Union[np.ndarray, Tensor, NestedTorchTensor]) -> NestedTorchTensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x.copy())
    if isinstance(x, Tensor):
        return torch.as_tensor(np.asarray(x).copy())
    if hasattr(x, "numpy"):
        return torch.as_tensor(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_torch_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def _parameters_from_attention(
    src: NestedTensor,
    dst: Union[
        hf_bert.BertAttention, hf_roberta.RobertaAttention, hf_deberta_v2.DebertaV2Attention
    ],
    key_mapping: Iterable[tuple[str, str]] = (
        ("q_proj", "query"),
        ("k_proj", "key"),
        ("v_proj", "value"),
    ),
):
    # Copy Q,K,V. In Hugging Face, this dense layer is in *SelfAttention layer.
    dst_self = dst.self
    all_head_dim = dst_self.all_head_size
    for src_proj, dst_proj in key_mapping:
        src_dense = src["i_proj"][src_proj]
        dst_dense = getattr(dst_self, dst_proj)
        # pytype: disable=attribute-error
        dst_dense.weight.data = as_torch_tensor(
            src_dense["weight"].reshape(-1, all_head_dim)
        ).transpose(0, 1)
        # pytype: enable=attribute-error
        dst_dense.bias.data = as_torch_tensor(src_dense["bias"].reshape(all_head_dim))

    # Copy O. In Hugging Face, this dense layer is in *SelfOutput layer.
    dst_output = dst.output.dense
    src_output = src["o_proj"]
    # No transpose here, as o_proj is already in (output, ...) layout.
    dst_output.weight.data = as_torch_tensor(src_output["weight"].reshape(-1, all_head_dim))
    dst_output.bias.data = as_torch_tensor(src_output["bias"])


def _parameters_to_hf_t5_attention(src: NestedTensor, dst: hf_t5.T5Attention):
    # Copy Q,K,V.
    all_head_dim = dst.inner_dim
    per_head_dim = dst.key_value_proj_dim
    linear_state = src["i_proj"]

    # Handle fused qkv linear state.
    if "qkv_proj" in linear_state:
        linear_state = {
            proj: dict(weight=src["i_proj"]["qkv_proj"]["weight"][i])
            for i, proj in enumerate(["q_proj", "k_proj", "v_proj"])
        }

    for src_proj, dst_proj in (
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
    ):
        src_dense = linear_state[src_proj]
        dst_dense = getattr(dst, dst_proj)
        weight = src_dense["weight"]
        if src_proj == "q_proj":
            weight /= per_head_dim**0.5
        dst_dense.weight.data = as_torch_tensor(weight.reshape(-1, all_head_dim)).transpose(
            0, 1
        )  # pytype: disable=attribute-error

    # Copy O.
    dst_output = dst.o
    src_output = src["o_proj"]
    # No transpose here, as o_proj is already in (output, ...) layout.
    dst_output.weight.data = as_torch_tensor(src_output["weight"].reshape(-1, all_head_dim))


def _parameters_to_hf_gpt2_attention(src: NestedTensor, dst: hf_gpt2.GPT2Attention):
    # Copy Q,K,V.
    all_head_dim = dst.embed_dim
    c_attn = []
    for src_proj in ["q_proj", "k_proj", "v_proj"]:
        weight = src["i_proj"][src_proj]["weight"]
        c_attn.append(as_torch_tensor(weight.reshape(-1, all_head_dim)))
    c_attn = torch.cat(c_attn, dim=-1)
    dst.c_attn.weight.data = c_attn  # pytype: disable=attribute-error
    # Copy O.
    dst.c_proj.weight.data = as_torch_tensor(
        src["o_proj"]["weight"].reshape(-1, all_head_dim).transpose()
    )


# pylint: disable-next=too-many-branches
def flax_to_torch(src: NestedTensor, dst: torch.nn.Module):
    """Copies parameters from an AXLearn Flax layer into a compatible torch module.
    The destination module is modified in-place.

    Args:
        src: Params corresponding to the Flax layer.
        dst: Torch module to copy to. It is modified in-place.

    Raises:
        NotImplementedError: If the dst type is not supported.
    """

    if isinstance(dst, torch.nn.LayerNorm):
        dst.weight.data = as_torch_tensor(src["scale"])
        dst.bias.data = as_torch_tensor(src["bias"])
    elif isinstance(dst, torch.nn.Linear):
        dst.weight.data = as_torch_tensor(src["kernel"]).transpose(
            0, 1
        )  # pytype: disable=attribute-error
        dst.bias.data = as_torch_tensor(src["bias"])
    elif isinstance(dst, torch.nn.Embedding):
        dst.weight.data = as_torch_tensor(src["embedding"])
    elif isinstance(dst, hf_roberta.RobertaSelfAttention):
        flax_to_torch(src["query"], dst.query)
        flax_to_torch(src["key"], dst.key)
        flax_to_torch(src["value"], dst.value)
    elif isinstance(dst, hf_roberta.RobertaAttention):
        flax_to_torch(src["self"], dst.self)
        flax_to_torch(src["output"], dst.output)
    elif isinstance(dst, hf_roberta.RobertaIntermediate):
        flax_to_torch(src["dense"], dst.dense)
    elif isinstance(dst, hf_roberta.RobertaLayer):
        flax_to_torch(src["attention"], dst.attention)
        flax_to_torch(src["intermediate"], dst.intermediate)
        flax_to_torch(src["output"], dst.output)
    elif isinstance(dst, (hf_roberta.RobertaSelfOutput, hf_roberta.RobertaOutput)):
        flax_to_torch(src["dense"], dst.dense)
        flax_to_torch(src["LayerNorm"], dst.LayerNorm)
    elif isinstance(dst, hf_roberta.RobertaEmbeddings):
        flax_to_torch(src["word_embeddings"], dst.word_embeddings)
        flax_to_torch(src["position_embeddings"], dst.position_embeddings)
        flax_to_torch(src["token_type_embeddings"], dst.token_type_embeddings)
        flax_to_torch(src["LayerNorm"], dst.LayerNorm)
    elif isinstance(dst, hf_roberta.RobertaEncoder):
        for layer_id, layer_state in src["layer"].items():
            flax_to_torch(layer_state, dst.layer[int(layer_id)])
    elif isinstance(dst, hf_roberta.RobertaClassificationHead):
        flax_to_torch(src["dense"], dst.dense)
        flax_to_torch(src["out_proj"], dst.out_proj)
    elif isinstance(dst, hf_roberta.RobertaModel):
        flax_to_torch(src["embeddings"], dst.embeddings)
        flax_to_torch(src["encoder"], dst.encoder)
    else:
        raise NotImplementedError(f"{type(dst)}")


# pylint: disable-next=too-many-branches,too-many-statements
def axlearn_to_torch(layer: BaseLayer, src: NestedTensor, dst: torch.nn.Module):
    """Copies parameters from an AXLearn layer into a compatible torch module.
    The destination module is modified in-place.
    See also `torch_to_axlearn` for the inverse.

    Args:
        layer: AXLearn layer to extract from.
        src: Params corresponding to the AXLearn layer.
        dst: Torch module to copy to. It is modified in-place.

    Raises:
        NotImplementedError: If conversion between src, dst is not supported.
        ValueError: If conversion fails.
    """

    def check_supported(*supported_layers: type):
        if not isinstance(layer, supported_layers):
            raise NotImplementedError(
                f"Conversion from {type(layer)} to {type(dst)} is not yet supported. "
                f"Supported layers are {supported_layers}."
            )

    # Match the input src, dst against a registered converter.
    for (src_type, dst_type), convert_fn in _axlearn_to_torch_registry.items():
        if isinstance(layer, src_type) and isinstance(dst, dst_type):
            convert_fn(src=layer, params=src, dst=dst)
            return

    if isinstance(dst, torch.nn.LayerNorm):
        check_supported(LayerNorm, LayerNormStateless)
        if isinstance(layer, LayerNorm):
            if not dst.elementwise_affine:
                raise ValueError("elementwise_affine must be True for conversion from LayerNorm")
            dst.weight.data = as_torch_tensor(src["scale"])
            dst.bias.data = as_torch_tensor(src["bias"])
        else:
            if dst.elementwise_affine:
                raise ValueError(
                    "elementwise_affine must be False for conversion from LayerNormStateless"
                )
            # No parameter to set.
    elif isinstance(dst, torch.nn.Linear):
        check_supported(Linear, MultiheadInputLinear, BertSequenceClassificationHead)
        if isinstance(layer, MultiheadInputLinear):
            # Shape of src["weight"] is [model_dim, num_heads, per_head_dim].
            all_head_dim = src["weight"].shape[1] * src["weight"].shape[2]
            dst.weight.data = as_torch_tensor(src["weight"].reshape(-1, all_head_dim)).transpose(
                0, 1
            )  # pytype: disable=attribute-error
            if "bias" in src:
                dst.bias.data = as_torch_tensor(src["bias"].reshape(all_head_dim))
        elif isinstance(layer, BertSequenceClassificationHead):
            # We only convert the linear layer of BertSequenceClassificationHead here. The pooler
            # of BertSequenceClassificationHead is handled outside with BertModel conversion.
            # torch.nn.Linear.weight uses layout (output, input) while AXLearn uses (input,
            # output).
            dst.weight.data = as_torch_tensor(src["output"]["weight"]).transpose(
                0, 1
            )  # pytype: disable=attribute-error
            if "bias" in src:
                dst.bias.data = as_torch_tensor(src["output"]["bias"])
        else:
            # torch.nn.Linear.weight uses layout (output, input) while AXLearn uses
            # (input, output).
            dst.weight.data = as_torch_tensor(src["weight"]).transpose(
                0, 1
            )  # pytype: disable=attribute-error
            if "bias" in src:
                dst.bias.data = as_torch_tensor(src["bias"])
    elif isinstance(dst, torch.nn.Embedding):
        check_supported(Embedding, LearnedPositionalEmbedding)
        dst.weight.data = as_torch_tensor(src["weight"])
        # In the case of LearnedPositionalEmbedding, remove extra leading dim.
        if isinstance(layer, LearnedPositionalEmbedding):
            dst.weight.data = torch.squeeze(dst.weight.data, dim=0)
    elif isinstance(dst, (hf_bert.BertAttention, hf_roberta.RobertaAttention)):
        check_supported(TransformerAttentionLayer)
        _parameters_from_attention(src["attention"], dst)
        axlearn_to_torch(layer.norm, src["norm"], dst.output.LayerNorm)
    elif isinstance(dst, hf_text_encoder.BertTextEmbeddingEncoder):
        axlearn_to_torch(layer.text_encoder, src["text_encoder"], dst.bert)
        if "output_proj" in src:
            axlearn_to_torch(layer.output_proj, src["output_proj"], dst.encode_proj)
    elif isinstance(dst, hf_text_encoder.MultiStreamTextEmbeddingModel):
        for encoder_name in dst.stream_encoders:
            axlearn_to_torch(
                getattr(layer, encoder_name), src[encoder_name], dst.stream_encoders[encoder_name]
            )
    elif isinstance(dst, torch.nn.Conv2d):
        check_supported(Conv2D)
        dst.weight.data = as_torch_tensor(src["weight"]).permute(
            (3, 2, 0, 1)
        )  # pytype: disable=attribute-error
        if "bias" in src:
            dst.bias.data = as_torch_tensor(src["bias"])
    elif isinstance(dst, (hf_bert.BertEmbeddings, hf_roberta.RobertaEmbeddings)):
        check_supported(TransformerTextEmbeddings)
        axlearn_to_torch(layer.token_emb, src["token_emb"], dst.word_embeddings)
        pos_emb = src["pos_emb"]
        if isinstance(dst, hf_roberta.RobertaEmbeddings):
            # Note: Hugging Face RoBERTa embeddings are slightly different from BERT:
            # Positions start at pad_token_id+1 and padding positions are explicitly masked out.
            # Reference:
            # https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/roberta/modeling_roberta.py#L1573-L1575
            pos_emb = dict(
                weight=jnp.pad(pos_emb["weight"], [(0, 0), (dst.padding_idx + 1, 0), (0, 0)]),
            )
        axlearn_to_torch(layer.pos_emb, pos_emb, dst.position_embeddings)
        axlearn_to_torch(layer.norm, src["norm"], dst.LayerNorm)

        # Type embedding might be None, in which case we replace with zeros.
        type_emb_layer = getattr(layer, "type_emb", None)
        type_emb_params = src.get("type_emb", None)
        if type_emb_layer is None or type_emb_params is None:
            if (type_emb_layer is None) != (type_emb_params is None):
                raise ValueError(
                    "Input layer and params must either both have type embeddings, or both not."
                )
            dummy_weights = jnp.zeros(dst.token_type_embeddings.weight.size())
            type_emb_layer = (
                Embedding.default_config()
                .set(
                    name="dummy_type_emb",
                    num_embeddings=dummy_weights.shape[0],
                    dim=dummy_weights.shape[1],
                )
                .instantiate(parent=None)
            )
            type_emb_params = {"weight": dummy_weights}
        axlearn_to_torch(
            type_emb_layer,
            type_emb_params,
            dst.token_type_embeddings,
        )
    elif isinstance(
        dst,
        (
            hf_bert.BertIntermediate,
            hf_roberta.RobertaIntermediate,
            hf_deberta_v2.DebertaV2Intermediate,
        ),
    ):
        check_supported(TransformerFeedForwardLayer)
        axlearn_to_torch(layer.linear1, src["linear1"], dst.dense)
    elif isinstance(
        dst, (hf_bert.BertOutput, hf_roberta.RobertaOutput, hf_deberta_v2.DebertaV2Output)
    ):
        check_supported(TransformerFeedForwardLayer)
        axlearn_to_torch(layer.linear2, src["linear2"], dst.dense)
        axlearn_to_torch(layer.norm, src["norm"], dst.LayerNorm)
    elif isinstance(
        dst, (hf_bert.BertLayer, hf_roberta.RobertaLayer, hf_deberta_v2.DebertaV2Layer)
    ):
        check_supported(TransformerLayer)
        axlearn_to_torch(layer.self_attention, src["self_attention"], dst.attention)
        axlearn_to_torch(layer.feed_forward, src["feed_forward"], dst.intermediate)
        axlearn_to_torch(layer.feed_forward, src["feed_forward"], dst.output)
    elif isinstance(dst, (hf_bert.BertEncoder, hf_roberta.RobertaEncoder)):
        check_supported(StackedTransformerLayer, RepeatedTransformerLayer)
        for i, dst_layer in enumerate(dst.layer):
            if isinstance(layer, StackedTransformerLayer):
                axlearn_to_torch(getattr(layer, f"layer{i}"), src[f"layer{i}"], dst_layer)
            else:
                assert isinstance(layer, RepeatedTransformerLayer)
                axlearn_to_torch(
                    layer.repeat.layer,
                    jax.tree.map(lambda x, idx=i: x[idx], src["repeat"]["layer"]),
                    dst_layer,
                )
    elif isinstance(dst, (hf_bert.BertPooler, hf_roberta.RobertaPooler)):
        check_supported(BertPooler)
        axlearn_to_torch(layer.linear, src["linear"], dst.dense)
        # Note: always use tanh as activation here.
    elif isinstance(dst, hf_bert.BertModel):
        check_supported(TextEmbeddingEncoder, BertModel)
        axlearn_to_torch(layer.encoder.emb, src["encoder"]["emb"], dst.embeddings)
        axlearn_to_torch(layer.encoder.transformer, src["encoder"]["transformer"], dst.encoder)
        src_pooler = src.get("head", {}).get("pooler", None)
        if (src_pooler is not None) != (dst.pooler is not None):
            raise ValueError(
                "Input layer and output layer must either both have pooler, or both not."
            )
        if src_pooler:
            axlearn_to_torch(layer.head.pooler, src_pooler, dst.pooler)
    elif isinstance(dst, hf_roberta.RobertaModel):
        check_supported(TextEmbeddingEncoder, BertModel)
        axlearn_to_torch(layer.encoder.emb, src["encoder"]["emb"], dst.embeddings)
        axlearn_to_torch(layer.encoder.transformer, src["encoder"]["transformer"], dst.encoder)
        has_pooler = "pooler" in src and src["pooler"] != {}
        if has_pooler != (dst.pooler is not None):
            raise ValueError(
                "Input layer and output layer must either both have pooler, or both not."
            )
        if has_pooler:
            axlearn_to_torch(layer.pooler, src["pooler"], dst.pooler)
    elif isinstance(dst, hf_deberta_v2.DebertaV2ForSequenceClassification):
        check_supported(BertModel)
        axlearn_to_torch(layer.head.pooler, src["head"]["pooler"], dst.pooler)
        axlearn_to_torch(layer.head.output, src["head"]["output"], dst.classifier)
        axlearn_to_torch(layer, src, dst.deberta)
    elif isinstance(dst, hf_deberta_v2.ContextPooler):
        check_supported(BertPooler)
        axlearn_to_torch(layer.linear, src["linear"], dst.dense)
    elif isinstance(dst, hf_deberta_v2.DebertaV2Model):
        check_supported(BertModel)
        axlearn_to_torch(layer.encoder.emb, src["encoder"]["emb"], dst.embeddings)
        axlearn_to_torch(layer.encoder, src["encoder"], dst.encoder)
    elif isinstance(dst, hf_deberta_v2.DebertaV2Embeddings):
        check_supported(TransformerTextEmbeddings)
        axlearn_to_torch(layer.token_emb, src["token_emb"], dst.word_embeddings)
        if hasattr(layer, "pos_emb"):
            axlearn_to_torch(layer.pos_emb, src["pos_emb"], dst.position_embeddings)
        axlearn_to_torch(layer.norm, src["norm"], dst.LayerNorm)
    elif isinstance(dst, hf_deberta_v2.DebertaV2Encoder):
        check_supported(DeBERTaV2Encoder)
        axlearn_to_torch(layer.relative_pos_emb, src["relative_pos_emb"], dst.rel_embeddings)
        axlearn_to_torch(
            layer.relative_pos_emb.norm, src["relative_pos_emb"]["norm"], dst.LayerNorm
        )

        layer, src = layer.transformer, src["transformer"]
        check_supported(StackedTransformerLayer, RepeatedTransformerLayer)
        for i, dst_layer in enumerate(dst.layer):
            if isinstance(layer, StackedTransformerLayer):
                axlearn_to_torch(getattr(layer, f"layer{i}"), src[f"layer{i}"], dst_layer)
            else:
                assert isinstance(layer, RepeatedTransformerLayer)
                axlearn_to_torch(
                    layer.repeat.layer,
                    jax.tree.map(lambda x, idx=i: x[idx], src["repeat"]["layer"]),
                    dst_layer,
                )
    elif isinstance(dst, hf_deberta_v2.DebertaV2Attention):
        check_supported(TransformerAttentionLayer)
        _parameters_from_attention(
            src["attention"],
            dst,
            key_mapping=(
                ("q_proj", "query_proj"),
                ("k_proj", "key_proj"),
                ("v_proj", "value_proj"),
            ),
        )

        if hasattr(dst.self, "pos_key_proj") or hasattr(dst.self, "pos_query_proj"):
            axlearn_to_torch(
                layer.attention.pos_k_proj, src["attention"]["pos_k_proj"], dst.self.pos_key_proj
            )
            axlearn_to_torch(
                layer.attention.pos_q_proj, src["attention"]["pos_q_proj"], dst.self.pos_query_proj
            )

        axlearn_to_torch(layer.norm, src["norm"], dst.output.LayerNorm)
    elif isinstance(dst, hf_bert.BertForSequenceClassification):
        check_supported(BertModel)
        axlearn_to_torch(layer.head, src["head"], dst.classifier)
        axlearn_to_torch(layer, src, dst.bert)
    elif isinstance(dst, hf_roberta.RobertaClassificationHead):
        check_supported(BertSequenceClassificationHead)
        axlearn_to_torch(layer.pooler.linear, src["pooler"]["linear"], dst.dense)
        axlearn_to_torch(layer.output, src["output"], dst.out_proj)
    elif isinstance(dst, hf_roberta.RobertaForSequenceClassification):
        check_supported(BertModel)
        axlearn_to_torch(layer.head, src["head"], dst.classifier)
        axlearn_to_torch(layer, src, dst.roberta)
    elif isinstance(dst, hf_t5.T5LayerNorm):
        check_supported(RMSNorm)
        dst.weight.data = as_torch_tensor(src["scale"])
    elif isinstance(dst, hf_t5.T5Attention):
        check_supported(TransformerAttentionLayer)
        _parameters_to_hf_t5_attention(src["attention"], dst)
    elif isinstance(dst, hf_t5.T5LayerFF):
        check_supported(TransformerFeedForwardLayer)
        ff = dst.DenseReluDense
        if isinstance(ff, hf_t5.T5DenseActDense):
            axlearn_to_torch(getattr(layer, "linear1"), src["linear1"], dst.DenseReluDense.wi)
        # Compat with transformers<4.21.1
        else:
            if hasattr(hf_t5, "T5DenseGatedGeluDense"):
                dense_gated_cls = hf_t5.T5DenseGatedGeluDense
            else:
                dense_gated_cls = hf_t5.T5DenseGatedActDense
            if isinstance(ff, dense_gated_cls):
                axlearn_to_torch(
                    getattr(layer, "linear1_0"), src["linear1_0"], dst.DenseReluDense.wi_0
                )
                axlearn_to_torch(
                    getattr(layer, "linear1_1"), src["linear1_1"], dst.DenseReluDense.wi_1
                )
        axlearn_to_torch(layer.linear2, src["linear2"], dst.DenseReluDense.wo)
        axlearn_to_torch(layer.norm, src["norm"], dst.layer_norm)
    elif isinstance(dst, hf_t5.T5Stack):
        ff_layer_idx = 1
        for i, t5_block in enumerate(dst.block):
            if isinstance(layer.transformer, RepeatedTransformerLayer):
                transformer_layer = layer.transformer.repeat.layer
                src_layer = jax.tree.map(
                    lambda x, idx=i: x[idx], src["transformer"]["repeat"]["layer"]
                )
            else:
                transformer_layer = getattr(layer.transformer, f"layer{i}")
                src_layer = src["transformer"][f"layer{i}"]
            # self attention
            axlearn_to_torch(
                transformer_layer.self_attention,
                src_layer["self_attention"],
                t5_block.layer[0].SelfAttention,
            )
            axlearn_to_torch(
                transformer_layer.self_attention.norm,
                src_layer["self_attention"]["norm"],
                t5_block.layer[0].layer_norm,
            )
            if t5_block.layer[0].SelfAttention.has_relative_attention_bias:
                axlearn_to_torch(
                    layer.relative_pos_emb.emb,
                    src["relative_pos_emb"]["emb"],
                    t5_block.layer[0].SelfAttention.relative_attention_bias,
                )
            if t5_block.is_decoder:
                # cross attention
                axlearn_to_torch(
                    transformer_layer.cross_attention,
                    src_layer["cross_attention"],
                    t5_block.layer[1].EncDecAttention,
                )
                axlearn_to_torch(
                    transformer_layer.cross_attention.norm,
                    src_layer["cross_attention"]["norm"],
                    t5_block.layer[1].layer_norm,
                )
                ff_layer_idx = 2
            # feed forward
            axlearn_to_torch(
                transformer_layer.feed_forward,
                src_layer["feed_forward"],
                t5_block.layer[ff_layer_idx],
            )
        # t5.T5Encoder output norm
        if hasattr(layer, "output"):
            axlearn_to_torch(layer.output.norm, src["output"]["norm"], dst.final_layer_norm)
        # t5.T5Decoder output norm
        if hasattr(layer, "output_norm"):
            axlearn_to_torch(layer.output_norm, src["output_norm"], dst.final_layer_norm)

    elif isinstance(dst, (hf_t5.T5ForConditionalGeneration, hf_t5.T5Model)):
        check_supported(T5EncoderDecoderModel)
        axlearn_to_torch(layer.shared_token_emb, src["shared_token_emb"], dst.shared)
        axlearn_to_torch(layer.encoder, src["encoder"], dst.encoder)
        if hasattr(layer.decoder, "lm_head") and layer.decoder.lm_head is not None:
            dst.lm_head.weight.data = as_torch_tensor(src["decoder"]["lm_head"]["weight"])
        axlearn_to_torch(layer.decoder, src["decoder"], dst.decoder)
    elif isinstance(dst, hf_gpt2.GPT2MLP):
        check_supported(TransformerFeedForwardLayer)
        dst.c_fc.weight.data = as_torch_tensor(src["linear1"]["weight"])
        dst.c_fc.bias.data = as_torch_tensor(src["linear1"]["bias"])
        dst.c_proj.weight.data = as_torch_tensor(src["linear2"]["weight"])
        dst.c_proj.bias.data = as_torch_tensor(src["linear2"]["bias"])
    elif isinstance(dst, hf_gpt2.GPT2Attention):
        check_supported(MultiheadAttention)
        _parameters_to_hf_gpt2_attention(src, dst)
    elif isinstance(dst, hf_gpt2.GPT2Block):
        check_supported(TransformerLayer)
        axlearn_to_torch(layer.self_attention.norm, src["self_attention"]["norm"], dst.ln_1)
        axlearn_to_torch(
            layer.self_attention.attention, src["self_attention"]["attention"], dst.attn
        )
        axlearn_to_torch(layer.feed_forward.norm, src["feed_forward"]["norm"], dst.ln_2)
        axlearn_to_torch(layer.feed_forward, src["feed_forward"], dst.mlp)
    elif isinstance(dst, hf_gpt2.GPT2Model):
        check_supported(CausalLMModel)
        axlearn_to_torch(layer.decoder.emb.token_emb, src["decoder"]["emb"]["token_emb"], dst.wte)
        axlearn_to_torch(layer.decoder.emb.pos_emb, src["decoder"]["emb"]["pos_emb"], dst.wpe)
        axlearn_to_torch(layer.decoder.output_norm, src["decoder"]["output_norm"], dst.ln_f)
        for i, gpt2_block in enumerate(dst.h):
            transformer_layer = getattr(layer.decoder.transformer, f"layer{i}")
            axlearn_to_torch(
                transformer_layer, src["decoder"]["transformer"][f"layer{i}"], gpt2_block
            )
    elif isinstance(dst, hf_gpt2.GPT2LMHeadModel):
        check_supported(CausalLMModel)
        axlearn_to_torch(layer, src, dst.transformer)
        if hasattr(layer.decoder, "lm_head") and layer.decoder.lm_head is not None:
            dst.lm_head.weight.data = as_torch_tensor(src["decoder"]["lm_head"]["weight"])
    else:
        raise NotImplementedError(f"{type(dst)}")


# pylint: disable-next=too-many-branches,too-many-statements
def torch_to_axlearn(
    src: torch.nn.Module, *, dst_layer: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    """Extracts parameters from a torch module into a compatible format with AXLearn layers.
    See also `axlearn_to_torch` for the inverse.

    Args:
        src: pytorch (e.g., huggingface) module to extract from.
        dst_layer: (optional) the destination AXLearn layer.

    Returns:
        The AXLearn-compatible parameters.

    Raises:
        NotImplementedError: If src type is not supported.
    """

    # Match the input src, dst against a registered converter.
    for (src_type, dst_type), convert_fn in _torch_to_axlearn_registry.items():
        if isinstance(src, src_type) and isinstance(dst_layer, dst_type):
            return as_tensor(convert_fn(src=src, dst=dst_layer))

    if isinstance(src, torch.nn.LayerNorm):
        if src.elementwise_affine:
            dst = dict(scale=src.weight, bias=src.bias)
        else:
            dst = {}
    elif isinstance(src, torch.nn.Linear):
        # torch.nn.Linear.weight uses layout (output, input) while AXLearn uses (input, output).
        dst = dict(weight=src.weight.transpose(0, 1))
        if src.bias is not None:
            dst["bias"] = src.bias
    elif isinstance(src, torch.nn.Embedding):
        dst = dict(weight=src.weight)
    elif isinstance(src, torch.nn.Conv2d):
        # torch.nn.Conv2d.weight uses layout (output, input, H, W) while AXLearn uses
        # (H, W, input, output).
        dst = dict(weight=src.weight.permute(2, 3, 1, 0), bias=src.bias)
    elif isinstance(src, torch.nn.Conv1d) and all(s == 1 for s in src.kernel_size):  # pointwise
        # Return parameters for a Linear layer.
        dst = dict(weight=src.weight.view(src.out_channels, src.in_channels).permute(1, 0))
        if src.bias:
            dst["bias"] = src.bias
    elif isinstance(src, torch.nn.Conv1d) and src.groups == src.in_channels:  # depthwise
        # (window, 1, in_channels).
        dst = dict(weight=src.weight.permute(2, 1, 0))
    elif isinstance(src, torch.nn.BatchNorm1d):
        dst = dict(
            scale=src.weight,
            bias=src.bias,
            moving_mean=src.running_mean,
            moving_variance=src.running_var,
        )
    elif isinstance(src, hf_roberta.RobertaAttention):
        dst = _parameters_from_roberta_attention(src)
    elif isinstance(src, hf_roberta.RobertaLayer):
        dst = _parameters_from_roberta_layer(src)
    elif isinstance(src, hf_vit.ViTForImageClassification):
        dst = _parameters_from_vit_classification(src)
    elif isinstance(src, hf_vit.ViTLayer):
        dst = _parameters_from_vit_layer(src)
    elif isinstance(src, (hf_vit.ViTModel, hf_vit_mae.ViTMAEModel)):
        dst = _parameters_from_vit_encoder(src)
    elif isinstance(src, hf_gpt2.GPT2LMHeadModel):
        dst = _parameters_from_gpt2_layer(src)
    elif isinstance(src, hf_bert.BertForMaskedLM):
        dst = _parameters_from_bert_mlm_model(src)
    elif isinstance(src, (hf_bert.BertForSequenceClassification, hf_bert.BertForMultipleChoice)):
        dst = _parameters_from_bert_sequence_classification_model(src)
    elif isinstance(src, hf_roberta.RobertaForSequenceClassification):
        dst = _parameters_from_roberta_sequence_classification_model(src)
    elif isinstance(src, hf_bert.BertEmbeddings):
        dst = _parameters_from_bert_embeddings(src, dst_layer=dst_layer)
    elif isinstance(src, hf_roberta.RobertaEmbeddings):
        dst = _parameters_from_roberta_embeddings(src, dst_layer=dst_layer)
    elif isinstance(src, hf_encoder_decoder.EncoderDecoderModel):
        dst = _parameters_from_encoder_decoder_model(src)
    elif isinstance(src, hf_bert.BertModel):
        dst = _parameters_from_bert_model(src, dst_layer=dst_layer)
    elif isinstance(src, (hf_bert.BertEncoder, hf_roberta.RobertaEncoder)):
        dst = _parameters_from_bert_encoder(src, dst_layer=dst_layer)
    elif isinstance(src, hf_roberta.RobertaModel):
        dst = _parameters_from_roberta_model(src)
    elif isinstance(src, hf_xlnet.XLNetRelativeAttention):
        dst = _parameters_from_xlnet_attention(src)
    elif isinstance(src, hf_t5.T5LayerNorm):
        dst = dict(scale=src.weight)
    elif isinstance(src, hf_t5.T5EncoderModel):
        dst = _parameters_from_t5_encoder_model(src, dst_layer=dst_layer)
    elif isinstance(src, hf_t5.T5Stack):
        dst = _parameters_from_t5_stack(src, dst_layer=dst_layer)
    elif isinstance(src, hf_t5.T5ForConditionalGeneration):
        dst = _parameters_from_t5_for_conditional_generation(src, dst_layer=dst_layer)
    elif isinstance(src, (hf_mt5.MT5ForConditionalGeneration, hf_mt5.MT5Model)):
        dst = _parameters_from_t5_for_conditional_generation(src, dst_layer=dst_layer)
    elif isinstance(src, hf_opt.OPTModel):
        dst = _parameters_from_opt_model(src, dst_layer=dst_layer)
    elif isinstance(src, hf_opt.OPTForCausalLM):
        dst = _parameters_from_opt_causal_lm(src, dst_layer=dst_layer)
    elif isinstance(src, hf_dpr.DPRQuestionEncoder):
        dst = _parameters_from_dpr_question_encoder(src)
    elif isinstance(src, hf_dpr.DPRContextEncoder):
        dst = _parameters_from_dpr_context_encoder(src)
    elif isinstance(src, hf_dpr.DPREncoder):
        dst = _parameters_from_dpr_encoder(src)
    elif isinstance(src, hf_roformer.RoFormerAttention):
        dst = _parameters_from_roformer_attention(src)
    elif isinstance(src, hf_deberta_v2.ContextPooler):
        dst = _parameters_from_deberta_context_pooler(src)
    elif isinstance(src, hf_deberta_v2.DebertaV2Attention):
        dst = _parameters_from_deberta_attention(src)
    elif isinstance(src, hf_deberta_v2.DebertaV2Model):
        dst = _parameters_from_deberta_model(src, dst_layer=dst_layer)
    elif isinstance(src, hf_deberta_v2.DebertaV2ForSequenceClassification):
        dst = _parameters_from_deberta_for_sequence_classification(src, dst_layer=dst_layer)
    elif isinstance(src, hf_distilbert.DistilBertModel):
        dst = _parameters_from_distilbert_model(src)
    elif isinstance(src, timm_vit.Attention):
        dst = _parameters_from_timm_vit_attn(src, dst_layer=dst_layer)
    elif isinstance(src, timm_vit.PatchEmbed):
        dst = _parameters_from_timm_vit_patch_embed(src)
    elif isinstance(dst_layer, TimeStepEmbedding):
        dst = _parameters_from_dit_timestep_embedding(src)
    elif isinstance(dst_layer, LabelEmbedding):
        dst = _parameters_from_dit_label_embedding(src)
    elif isinstance(dst_layer, AdaptiveLayerNormModulation):
        dst = _parameters_from_dit_adaln(src)
    elif isinstance(dst_layer, DiTFeedForwardLayer):
        dst = _parameters_from_dit_ffn(src)
    elif isinstance(dst_layer, DiTAttentionLayer):
        dst = _parameters_from_dit_attn(src, dst_layer=dst_layer)
    elif isinstance(dst_layer, DiTBlock):
        dst = _parameters_from_dit_block(src, dst_layer=dst_layer)
    elif isinstance(dst_layer, DiTFinalLayer):
        dst = _parameters_from_dit_final_layer(src, dst_layer=dst_layer)
    else:
        raise NotImplementedError(f"{type(src)}: {src}")
    return as_tensor(dst)


def _parameters_from_deberta_model(
    src: hf_deberta_v2.DebertaV2Model,
    dst_layer: Optional[Union[EncoderModel, DeBERTaV2Encoder]] = None,
) -> NestedTensor:
    # Sometimes we directly convert HF DebertaV2Model to DeBERTaV2Encoder since in AXLearn
    # emb is inside the encoder.
    if isinstance(dst_layer, EncoderModel):
        dst_layer = dst_layer.encoder
    encoder = _parameters_from_deberta_encoder(src.encoder, dst_layer=dst_layer)
    encoder["emb"] = _parameters_from_deberta_embeddings(src.embeddings)
    return dict(encoder=encoder)


def _parameters_from_deberta_for_sequence_classification(
    src: hf_deberta_v2.DebertaV2ForSequenceClassification,
    dst_layer: Optional[Union[EncoderModel, DeBERTaV2Encoder]] = None,
) -> NestedTensor:
    model = _parameters_from_deberta_model(src.deberta, dst_layer=dst_layer)
    pooler = _parameters_from_deberta_context_pooler(src.pooler)
    classifier = torch_to_axlearn(src.classifier)
    return dict(encoder=model["encoder"], head=dict(pooler=pooler, output=classifier))


def _parameters_from_deberta_embeddings(
    src: hf_deberta_v2.DebertaV2Embeddings, *, dst_layer: Optional[BaseLayer] = None
) -> NestedTensor:
    if getattr(src, "embed_proj", None):
        raise NotImplementedError(
            "hf_deberta_v2.DebertaV2Embeddings.embed_proj conversion is not supported yet."
        )
    if getattr(src.config, "pad_token_id", None):
        raise NotImplementedError("Expected hf_deberta_v2.DebertaV2Embeddings.pad_token_id == 0")

    dst = dict(
        token_emb=torch_to_axlearn(src.word_embeddings),
        norm=torch_to_axlearn(src.LayerNorm),
        dropout={},
    )
    if getattr(src, "token_type_embeddings", None):
        dst["token_emb"] = torch_to_axlearn(src.token_type_embeddings)
    if getattr(src, "position_embeddings", None):
        dst["pos_emb"] = torch_to_axlearn(src.position_embeddings)
        # TODO(markblee): Move away from LearnedPositionalEmbedding.
        if dst_layer is None or isinstance(dst_layer.pos_emb, LearnedPositionalEmbedding):
            dst["pos_emb"]["weight"] = dst["pos_emb"]["weight"][None, ...]
    return dst


def _parameters_from_deberta_encoder(
    src: hf_deberta_v2.DebertaV2Encoder, dst_layer: Optional[DeBERTaV2Encoder] = None
) -> NestedTensor:
    if getattr(src, "conv", None):
        raise NotImplementedError("hf_deberta_v2.ConvLayer conversion is not supported yet.")

    dst = dict(
        transformer={},
        relative_pos_emb=dict(
            weight=src.rel_embeddings.weight,
            norm=torch_to_axlearn(src.LayerNorm),
        ),
        attention_mask={},
    )

    transformer_layers = {
        i: _parameters_from_deberta_layer(module) for i, module in enumerate(src.layer)
    }
    num_layers = len(transformer_layers)

    if dst_layer is None or isinstance(dst_layer.transformer, RepeatedTransformerLayer):
        # We assume the dst is RepeatedTransformerLayer if dst_layer is not specified.
        # RepeatedTransformerLayer is the default in deberta.py and in HuggingFacePreTrainedBuilder
        # we do not know the dst ahead of time.
        dst["transformer"]["repeat"] = VDict(
            # pylint: disable-next=no-value-for-parameter
            layer=jax.tree.map(
                lambda *inputs: jnp.stack(inputs),
                *[transformer_layers[i] for i in range(num_layers)],
            )
        )
    elif isinstance(dst_layer.transformer, StackedTransformerLayer):
        for i in range(num_layers):
            dst["transformer"][f"layer{i}"] = transformer_layers[i]
    else:
        raise NotImplementedError(type(dst_layer.transformer))

    return dst


def _parameters_from_deberta_layer(src: hf_deberta_v2.DebertaV2Layer) -> NestedTensor:
    return dict(
        self_attention=_parameters_from_deberta_attention(src.attention),
        feed_forward=_parameters_from_deberta_feed_forward(
            src.intermediate,
            src.output,
        ),
    )


def _parameters_from_deberta_feed_forward(
    intermediate: hf_deberta_v2.DebertaV2Intermediate, output: hf_deberta_v2.DebertaV2Output
) -> NestedTensor:
    return dict(
        linear1=torch_to_axlearn(intermediate.dense),
        linear2=torch_to_axlearn(output.dense),
        norm=torch_to_axlearn(output.LayerNorm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_deberta_context_pooler(src: hf_deberta_v2.ContextPooler) -> NestedTensor:
    return dict(linear=torch_to_axlearn(src.dense), dropout={})


def _parameters_from_deberta_attention(
    src: hf_deberta_v2.DebertaV2Attention,
) -> NestedTensor:
    return dict(
        norm=torch_to_axlearn(src.output.LayerNorm),
        attention=_parameters_from_deberta_self_attention(src.self, src.output),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_deberta_self_attention(
    src: hf_deberta_v2.DisentangledSelfAttention,
    output: hf_deberta_v2.DebertaV2SelfOutput,
) -> NestedTensor:
    num_heads = src.num_attention_heads
    per_head_dim = src.attention_head_size
    params = {}
    i_proj = {}
    for src_proj, dst_proj in (
        ("query_proj", "q_proj"),
        ("key_proj", "k_proj"),
        ("value_proj", "v_proj"),
        ("pos_key_proj", "pos_k_proj"),
        ("pos_query_proj", "pos_q_proj"),
    ):
        # pos_key_proj, pos_query_proj may not exist if C2P/P2C attention is not used, or if we are
        # sharing attention weights.
        if not hasattr(src, src_proj):
            continue
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        dense_params = dict(
            weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            bias=dense_params["bias"].reshape(num_heads, per_head_dim),
        )
        if src_proj in ("pos_key_proj", "pos_query_proj"):
            params[dst_proj] = dense_params
        else:
            i_proj[dst_proj] = dense_params
    output_dense = output.dense
    o_proj = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    params["i_proj"] = i_proj
    params["o_proj"] = o_proj
    params["dropout"] = {}
    params["pos_emb"] = {}
    params["pos_emb_dropout"] = {}
    params["scale_key"] = {}
    params["scale_query"] = {}
    if "pos_k_proj" not in params:
        params["pos_k_proj"] = {}
    if "pos_q_proj" not in params:
        params["pos_q_proj"] = {}
    return as_tensor(params)


# Note: OPT positional embeddings start at offset = pad_token_id+1.
# https://github.com/huggingface/transformers/blob/f62cb8313c2d7051e38f845823c1f4a7307aac3e/src/transformers/models/opt/modeling_opt.py#L99-L102
def _parameters_from_opt_embeddings(
    src: hf_opt.OPTDecoder, *, dst_layer: Optional[TransformerTextEmbeddings] = None
) -> NestedTensor:
    # Pad to multiple of 8 for even distribution across multiple TPUs.
    pos_emb_shape = src.embed_positions.weight.shape
    num_to_pad = ceil(pos_emb_shape[0] / 8) * 8 - pos_emb_shape[0]

    # Remap pad token embedding to 0, for compat with seqio and beam search decoding.
    # TODO(markblee): Revisit this once we lift the pad_token_id == 0 requirement.
    token_emb = src.embed_tokens.weight.clone()
    token_emb[[0, src.config.pad_token_id]] = token_emb[[src.config.pad_token_id, 0]]
    token_emb_params = dict(weight=token_emb)

    # OPT 350m supports token_emb.dim != pos_emb.dim, via input/output projections.
    # https://github.com/huggingface/transformers/blob/a26114777ee1c2802e91bd9cb26a3b39974d52ba/src/transformers/models/opt/modeling_opt.py#L495-L503
    if src.project_in is not None:
        token_emb_params["i_proj"] = torch_to_axlearn(src.project_in)
    if src.project_out is not None:
        token_emb_params["o_proj"] = torch_to_axlearn(src.project_out)

    dst = dict(
        token_emb=token_emb_params,
        pos_emb=dict(
            weight=torch.cat(
                [
                    # Move the first two "offset" embeddings to the end.
                    torch.roll(src.embed_positions.weight, -src.embed_positions.offset, 0),
                    # Pad to desired shape.
                    torch.zeros(
                        (num_to_pad,) + pos_emb_shape[1:], dtype=src.embed_positions.weight.dtype
                    ),
                ]
            )
        ),
        dropout={},
    )
    # By default, assume Embedding instead of LearnedPositionalEmbedding.
    if dst_layer is not None and isinstance(dst_layer.pos_emb, LearnedPositionalEmbedding):
        dst["pos_emb"]["weight"] = dst["pos_emb"]["weight"][None, ...]
    return as_tensor(dst)


def _parameters_from_opt_attention(src: hf_opt.OPTAttention) -> NestedTensor:
    num_heads = src.num_heads
    per_head_dim = src.head_dim
    weights = []
    biases = []
    for src_proj in ("q_proj", "k_proj", "v_proj"):
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        weights.append(dense_params["weight"].reshape(-1, num_heads, per_head_dim))
        if dense.bias is not None:
            biases.append(dense_params["bias"].reshape(num_heads, per_head_dim))

    # Assume we use FusedQKVLinear implementation.
    # TODO(markblee): Infer from dst_layer.
    qkv_proj = dict(weight=jnp.stack(weights), bias=None)
    if biases:
        qkv_proj["bias"] = jnp.stack(biases)

    output_dense = src.out_proj
    o_proj = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    dst = dict(i_proj=dict(qkv_proj=qkv_proj), o_proj=o_proj, dropout={})
    return as_tensor(dst)


def _parameters_from_opt_decoder_layer(src: hf_opt.OPTDecoderLayer) -> NestedTensor:
    return dict(
        self_attention=dict(
            attention=_parameters_from_opt_attention(src.self_attn),
            norm=torch_to_axlearn(src.self_attn_layer_norm),
        ),
        feed_forward=dict(
            linear1=torch_to_axlearn(src.fc1),
            dropout1={},
            linear2=torch_to_axlearn(src.fc2),
            dropout2={},
            norm=torch_to_axlearn(src.final_layer_norm),
        ),
    )


def _parameters_from_opt_decoder(
    src: hf_opt.OPTDecoder, dst_layer: Optional[Decoder] = None
) -> NestedTensor:
    transformer = None
    if dst_layer is not None:
        if isinstance(dst_layer.transformer, StackedTransformerLayer):
            transformer = {
                f"layer{i}": _parameters_from_opt_decoder_layer(layer)
                for i, layer in enumerate(src.layers)
            }

    # By default, assume RepeatedTransformerLayer.
    if transformer is None:
        layers = [_parameters_from_opt_decoder_layer(layer) for layer in src.layers]
        layers = jax.tree.map(as_tensor, layers)
        transformer = dict(
            repeat=dict(
                layer=jax.tree.map(lambda *inputs: jnp.stack(inputs), *layers),
            ),
        )

    # This is unused e.g. for 350m.
    # Note: This src.final_layer_norm is different from the one in OPTDecoderLayer.
    output_norm = None
    if src.final_layer_norm is not None:
        output_norm = torch_to_axlearn(src.final_layer_norm)

    return dict(
        emb=_parameters_from_opt_embeddings(src, dst_layer=dst_layer.emb if dst_layer else None),
        transformer=transformer,
        output_norm=output_norm,
        lm_head=None,  # Lives in OPTForCausalLM.
        attention_mask={},
        output_dropout={},
    )


def _parameters_from_opt_model(
    src: hf_opt.OPTModel, dst_layer: Optional[BaseLayer] = None
) -> NestedTensor:
    return dict(
        decoder=_parameters_from_opt_decoder(
            src.decoder, dst_layer=dst_layer.decoder if dst_layer else None
        )
    )


def _parameters_from_opt_causal_lm(
    src: hf_opt.OPTForCausalLM, dst_layer: Optional[BaseLayer] = None
) -> NestedTensor:
    # Note: OPTForCausalLM.lm_head uses tied weights with token emb, and no bias.
    # https://github.com/huggingface/transformers/blob/f62cb8313c2d7051e38f845823c1f4a7307aac3e/src/transformers/models/opt/modeling_opt.py#L811
    return _parameters_from_opt_model(src.model, dst_layer=dst_layer)


def _parameters_from_encoder_decoder_model(src: hf_encoder_decoder.EncoderDecoderModel):
    if src.decoder.config.model_type != "gpt2":
        raise NotImplementedError(
            f"{src.decoder.config.model_type} for encoder decoder parameters extraction"
        )
    if src.encoder.config.model_type != "bert":
        raise NotImplementedError(
            f"{src.encoder.config.model_type} for encoder decoder parameters extraction"
        )
    decoder = cast(hf_gpt2.GPT2LMHeadModel, src.decoder)
    decoder_transformer = {
        f"layer{i}": _parameters_from_gpt2_block_layer(l)
        for i, l in enumerate(decoder.transformer.h)
    }
    encoder = cast(hf_bert.BertModel, src.encoder)
    bert_parameters = _parameters_from_bert_model(encoder)
    all_parameters = dict(
        encoder=bert_parameters["encoder"],
        decoder=dict(
            emb=dict(
                token_emb=dict(weight=decoder.transformer.wte.weight),
                pos_emb=dict(weight=decoder.transformer.wpe.weight[None, ...]),
                dropout={},
            ),
            transformer=decoder_transformer,
            output_norm=torch_to_axlearn(decoder.transformer.ln_f),
            output_dropout={},
            attention_mask={},
        ),
    )
    return all_parameters


def _parameters_from_attention_dense(
    src: Union[hf_bert.BertSelfAttention, hf_roberta.RobertaSelfAttention, hf_vit.ViTSelfAttention],
    output: Union[hf_bert.BertOutput, hf_roberta.RobertaOutput, hf_vit.ViTOutput],
):
    num_heads = src.num_attention_heads
    per_head_dim = src.attention_head_size
    i_proj = {}
    for src_proj, dst_proj in (
        ("query", "q_proj"),
        ("key", "k_proj"),
        ("value", "v_proj"),
    ):
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        dense_params = dict(
            weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            bias=dense_params["bias"].reshape(num_heads, per_head_dim),
        )
        i_proj[dst_proj] = dense_params
    output_dense = output.dense
    o_proj = torch_to_axlearn(output_dense)
    o_proj = dict(
        weight=o_proj["weight"].transpose().reshape(-1, num_heads, per_head_dim),
        bias=o_proj["bias"],
    )
    return dict(i_proj=i_proj, o_proj=o_proj, dropout={}, scale_query={}, scale_key={})


def _parameters_from_roberta_attention(src: hf_roberta.RobertaAttention):
    return dict(
        attention=_parameters_from_attention_dense(src.self, src.output),
        norm=torch_to_axlearn(src.output.LayerNorm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_roformer_attention(src: hf_roformer.RoFormerAttention):
    return dict(
        attention=dict(
            i_proj=_parameters_from_attention_dense(src.self, src.output),
            o_proj=_parameters_from_attention_dense(src.self, src.output)["o_proj"],
        ),
        norm=torch_to_axlearn(src.output.LayerNorm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_roberta_feed_forward(
    intermediate: hf_roberta.RobertaIntermediate, output: hf_roberta.RobertaOutput
):
    return dict(
        linear1=torch_to_axlearn(intermediate.dense),
        linear2=torch_to_axlearn(output.dense),
        norm=torch_to_axlearn(output.LayerNorm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_roberta_layer(src: hf_roberta.RobertaLayer):
    return dict(
        self_attention=_parameters_from_roberta_attention(src.attention),
        feed_forward=_parameters_from_roberta_feed_forward(
            src.intermediate,
            src.output,
        ),
    )


def _parameters_from_dpr_encoder(src: hf_dpr.DPREncoder):
    return torch_to_axlearn(src.bert_model)


def _parameters_from_dpr_question_encoder(src: hf_dpr.DPRQuestionEncoder):
    dpr_question_encoder = torch_to_axlearn(src.question_encoder)
    return dict(text_encoder=dpr_question_encoder)


def _parameters_from_dpr_context_encoder(src: hf_dpr.DPRContextEncoder):
    dpr_ctx_encoder = torch_to_axlearn(src.ctx_encoder)
    return dict(text_encoder=dpr_ctx_encoder)


def _parameters_from_vit_classification(src: hf_vit.ViTForImageClassification):
    return dict(
        encoder=_parameters_from_vit_encoder(src.vit),
        classifier=torch_to_axlearn(src.classifier),
        classification_metric={},
    )


def _parameters_from_vit_embedding(src: hf_vit.ViTEmbeddings):
    return dict(conv=torch_to_axlearn(src.patch_embeddings.projection))


def _parameters_from_vit_encoder_1d(
    src_enc: hf_vit.ViTEncoder, src_emb: hf_vit.ViTEmbeddings, src_norm: torch.nn.LayerNorm
):
    dst = dict(
        drop_token={},
        input_dropout={},
        transformer={},
        pos_emb=dict(weight=src_emb.position_embeddings),
        output_norm=torch_to_axlearn(src_norm),
    )
    for layer_i, layer in enumerate(src_enc.layer):
        dst["transformer"][f"layer{layer_i}"] = torch_to_axlearn(layer)
    return dst


def _parameters_from_vit_attention(src: hf_vit.ViTAttention, src_norm: torch.nn.LayerNorm):
    return dict(
        attention=_parameters_from_attention_dense(src.attention, src.output),
        dropout={},
        norm=torch_to_axlearn(src_norm),
        stochastic_depth={},
    )


def _parameters_from_vit_feed_forward(
    intermediate: hf_vit.ViTIntermediate, output: hf_vit.ViTOutput, norm: torch.nn.LayerNorm
):
    return dict(
        linear1=torch_to_axlearn(intermediate.dense),
        linear2=torch_to_axlearn(output.dense),
        dropout1={},
        dropout2={},
        norm=torch_to_axlearn(norm),
        stochastic_depth={},
    )


def _parameters_from_vit_layer(src: hf_vit.ViTLayer):
    return dict(
        self_attention=_parameters_from_vit_attention(src.attention, src.layernorm_before),
        feed_forward=_parameters_from_vit_feed_forward(
            src.intermediate, src.output, src.layernorm_after
        ),
    )


def _parameters_from_vit_encoder(src: hf_vit.ViTModel):
    return dict(
        encoder=dict(
            visual_embed=dict(
                convert_to_sequence=_parameters_from_vit_embedding(src.embeddings),
            ),
            cls_tokens=src.embeddings.cls_token,
            encoder_1d=_parameters_from_vit_encoder_1d(
                src.encoder,
                src.embeddings,
                src.layernorm,
            ),
            pooler={},
        ),
    )


def _parameters_from_gpt2_feed_forward(src: hf_gpt2.GPT2MLP, norm: torch.nn.LayerNorm):
    return dict(
        norm=torch_to_axlearn(norm),
        linear1=dict(weight=src.c_fc.weight, bias=src.c_fc.bias),
        dropout1={},
        linear2=dict(weight=src.c_proj.weight, bias=src.c_proj.bias),
        dropout2={},
        stochastic_depth={},
    )


def _parameters_from_gpt2_attention(
    src: hf_gpt2.GPT2Attention, norm: torch.nn.LayerNorm, is_cross_attention: bool = False
):
    # GPT2 attention weights are concat into one array, break out head and q/k/v dims.
    num_heads = src.num_heads
    # Add empty state for the key/query scaling.
    attention = {"scale_key": {}, "scale_query": {}}
    # Head projection.
    c_attn_w = src.c_attn.weight
    c_attn_b = src.c_attn.bias
    if is_cross_attention:
        c_attn_w = torch.cat((src.q_attn.weight, src.c_attn.weight), dim=-1)
        c_attn_b = torch.cat((src.q_attn.bias, src.c_attn.bias), dim=-1)
    c_attn_w = c_attn_w.split(c_attn_w.shape[-1] // 3, dim=-1)
    c_attn_b = c_attn_b.split(c_attn_b.shape[-1] // 3, dim=-1)
    i_proj = {}
    for w, b, proj in zip(c_attn_w, c_attn_b, ("q_proj", "k_proj", "v_proj")):
        i_proj[proj] = dict(
            weight=w.reshape(w.shape[0], num_heads, -1), bias=b.reshape(num_heads, -1)
        )
    attention["i_proj"] = i_proj
    # Output projection.
    c_proj_w = src.c_proj.weight
    attention["o_proj"] = dict(
        weight=c_proj_w.T.reshape(c_proj_w.shape[0], num_heads, -1), bias=src.c_proj.bias
    )
    attention["dropout"] = {}
    return dict(norm=torch_to_axlearn(norm), attention=attention, dropout={}, stochastic_depth={})


def _parameters_from_gpt2_block_layer(src: hf_gpt2.GPT2Block):
    params = dict(
        self_attention=_parameters_from_gpt2_attention(src.attn, src.ln_1),
        feed_forward=_parameters_from_gpt2_feed_forward(src.mlp, src.ln_2),
    )
    if hasattr(src, "crossattention") and src.crossattention is not None:
        params.update(
            dict(
                cross_attention=_parameters_from_gpt2_attention(
                    src.crossattention, src.ln_cross_attn, is_cross_attention=True
                )
            )
        )
    return params


def _parameters_from_gpt2_layer(src: hf_gpt2.GPT2LMHeadModel):
    decoder_transformer = {
        f"layer{i}": _parameters_from_gpt2_block_layer(l) for i, l in enumerate(src.transformer.h)
    }
    parameters = dict(
        decoder=dict(
            emb=dict(
                token_emb=dict(weight=src.transformer.wte.weight),
                pos_emb=dict(weight=src.transformer.wpe.weight[None, ...]),
                dropout={},
            ),
            transformer=decoder_transformer,
            output_norm=torch_to_axlearn(src.transformer.ln_f),
            output_dropout={},
            attention_mask={},
        ),
    )
    # add lm_head weight
    if not src.config.tie_word_embeddings:
        parameters["decoder"]["lm_head"] = dict(weight=src.lm_head.weight)
    return parameters


# Note: Huggingface RoBERTa embeddings are slightly different from BERT:
# Positions start at pad_token_id+1 and padding positions are explicitly masked out.
def _parameters_from_roberta_embeddings(
    src: hf_roberta.RobertaEmbeddings, *, dst_layer: Optional[BaseLayer] = None
):
    dst = dict(
        token_emb=dict(weight=src.word_embeddings.weight),
        type_emb=dict(weight=src.token_type_embeddings.weight),
        pos_emb=dict(weight=src.position_embeddings.weight[src.padding_idx + 1 :]),
        norm=torch_to_axlearn(src.LayerNorm),
        dropout={},
    )
    if dst_layer is None or isinstance(dst_layer.pos_emb, LearnedPositionalEmbedding):
        dst["pos_emb"]["weight"] = dst["pos_emb"]["weight"][None, ...]
    return dst


def _parameters_from_bert_embeddings(
    src: hf_bert.BertEmbeddings, *, dst_layer: Optional[BaseLayer] = None
):
    dst = dict(
        token_emb=dict(weight=src.word_embeddings.weight),
        type_emb=dict(weight=src.token_type_embeddings.weight),
        pos_emb=dict(weight=src.position_embeddings.weight),
        norm=torch_to_axlearn(src.LayerNorm),
        dropout={},
    )
    if dst_layer is None or isinstance(dst_layer.pos_emb, LearnedPositionalEmbedding):
        dst["pos_emb"]["weight"] = dst["pos_emb"]["weight"][None, ...]
    return dst


def _parameters_from_bert_pooler(src: hf_bert.BertPooler):
    return dict(
        linear=torch_to_axlearn(src.dense),
        dropout={},
    )


def _parameters_from_bert_encoder(
    src: Union[hf_bert.BertEncoder, hf_roberta.RobertaEncoder],
    dst_layer: Optional[BaseStackedTransformerLayer] = None,
):
    transformer_layers = {
        i: _parameters_from_roberta_layer(module) for i, module in enumerate(src.layer)
    }
    num_layers = len(transformer_layers)
    if dst_layer is not None and isinstance(dst_layer, RepeatedTransformerLayer):
        return dict(
            repeat=VDict(
                # pylint: disable-next=no-value-for-parameter
                layer=jax.tree.map(
                    lambda *inputs: jnp.stack(inputs),
                    *[transformer_layers[i] for i in range(num_layers)],
                )
            )
        )
    else:
        return {f"layer{i}": transformer_layers[i] for i in range(num_layers)}


# Note: Hugging Face RoBERTa uses slightly different embeddings from BERT.
def _parameters_from_roberta_model(src: hf_roberta.RobertaModel):
    encoder = dict(
        emb=_parameters_from_roberta_embeddings(src.embeddings),
        transformer=_parameters_from_bert_encoder(src.encoder),
        attention_mask={},
    )
    params = dict(encoder=encoder)
    return params


def _parameters_from_bert_model(src: hf_bert.BertModel, dst_layer: Optional[Encoder] = None):
    encoder = dict(
        emb=_parameters_from_bert_embeddings(src.embeddings),
        transformer=_parameters_from_bert_encoder(
            src.encoder, dst_layer=dst_layer.transformer if dst_layer is not None else None
        ),
        attention_mask={},
    )
    params = dict(encoder=encoder)
    return params


def _parameters_from_bert_transform(src: hf_bert.BertPredictionHeadTransform):
    return dict(
        linear=torch_to_axlearn(src.dense),
        norm=torch_to_axlearn(src.LayerNorm),
    )


def _parameters_from_bert_lm_head(src: hf_bert.BertLMPredictionHead):
    return dict(
        inner_head={},
        transform=_parameters_from_bert_transform(src.transform),
        output_bias=src.bias.data,
        metric={},
    )


def _parameters_from_bert_mlm_model(src: hf_bert.BertForMaskedLM):
    encoder = _parameters_from_bert_model(src.bert)
    lm_head = _parameters_from_bert_lm_head(src.cls.predictions)
    return dict(head=lm_head, **encoder)


def _parameters_from_bert_sequence_classification_model(src: hf_bert.BertForSequenceClassification):
    encoder = _parameters_from_bert_model(src.bert)
    classifier = dict(
        pooler=_parameters_from_bert_pooler(src.bert.pooler),
        output=torch_to_axlearn(src.classifier),
        metric={},
    )
    return dict(head=classifier, **encoder)


def _parameters_from_roberta_classification_head(src: hf_roberta.RobertaClassificationHead):
    return dict(
        pooler=dict(linear=torch_to_axlearn(src.dense)),
        output=torch_to_axlearn(src.out_proj),
        metric={},
    )


# Note: Hugging Face RoBERTa groups the pooler and output projection into a classification head,
# like we do, but unlike what Hugging Face BERT does.
def _parameters_from_roberta_sequence_classification_model(
    src: hf_roberta.RobertaForSequenceClassification,
):
    encoder = _parameters_from_roberta_model(src.roberta)
    classifier = _parameters_from_roberta_classification_head(src.classifier)
    return dict(head=classifier, **encoder)


def _parameters_from_t5_attention(src: hf_t5.T5Attention, *, dst_layer: TransformerAttentionLayer):
    num_heads = src.n_heads
    per_head_dim = src.key_value_proj_dim
    i_proj, o_proj = {}, {}
    for src_proj, dst_proj in (
        ("q", "q_proj"),
        ("k", "k_proj"),
        ("v", "v_proj"),
        ("o", "o_proj"),
    ):
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        if dst_proj == "o_proj":
            dense_params = dict(
                weight=dense_params["weight"].transpose().reshape(-1, num_heads, per_head_dim),
            )
        else:
            dense_params = dict(
                weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            )
        if dst_proj == "q_proj":
            dense_params["weight"] *= per_head_dim**0.5
        if src_proj == "o":
            o_proj[dst_proj] = dense_params
        else:
            i_proj[dst_proj] = dense_params

    if isinstance(dst_layer.attention.i_proj, FusedQKVLinear):
        i_proj = VDict(
            qkv_proj=dict(
                jax.tree.map(
                    lambda *inputs: jnp.stack(inputs),
                    *[i_proj[proj_key] for proj_key in ("q_proj", "k_proj", "v_proj")],
                )
            ),
        )
    return dict(i_proj=i_proj, dropout={}, **o_proj, scale_query={}, scale_key={})


def _parameters_from_t5_self_attention(
    src: hf_t5.T5LayerSelfAttention, *, dst_layer: TransformerAttentionLayer
):
    return dict(
        attention=_parameters_from_t5_attention(src.SelfAttention, dst_layer=dst_layer),
        norm=torch_to_axlearn(src.layer_norm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_t5_cross_attention(
    src: hf_t5.T5LayerCrossAttention, *, dst_layer: TransformerAttentionLayer
):
    return dict(
        attention=_parameters_from_t5_attention(src.EncDecAttention, dst_layer=dst_layer),
        norm=torch_to_axlearn(src.layer_norm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_t5_feed_forward(src: hf_t5.T5LayerFF):
    if isinstance(src.DenseReluDense, hf_t5.T5DenseActDense):
        return dict(
            linear1=torch_to_axlearn(src.DenseReluDense.wi),
            linear2=torch_to_axlearn(src.DenseReluDense.wo),
            norm=torch_to_axlearn(src.layer_norm),
            dropout1={},
            dropout2={},
            stochastic_depth={},
        )
    ff = src.DenseReluDense
    # Compat with transformers<4.21.1
    if hasattr(hf_t5, "T5DenseGatedGeluDense"):
        dense_gated_cls = hf_t5.T5DenseGatedGeluDense
    else:
        dense_gated_cls = hf_t5.T5DenseGatedActDense
    if not isinstance(ff, dense_gated_cls):
        raise NotImplementedError(type(ff))
    return dict(
        linear1_0=torch_to_axlearn(ff.wi_0),
        linear1_1=torch_to_axlearn(ff.wi_1),
        linear2=torch_to_axlearn(ff.wo),
        norm=torch_to_axlearn(src.layer_norm),
        dropout1={},
        dropout2={},
        stochastic_depth={},
    )


def _parameters_from_t5_block(src: hf_t5.T5Block, *, dst_layer: TransformerLayer):
    """Returns parameters of a Transformer layer."""
    if src.is_decoder:
        self_atten, cross_atten, ff = src.layer.children()
    else:
        self_atten, ff = src.layer.children()
        cross_atten = None
    params = dict(
        self_attention=_parameters_from_t5_self_attention(
            self_atten, dst_layer=dst_layer.self_attention
        ),
        feed_forward=_parameters_from_t5_feed_forward(ff),
    )
    if cross_atten is not None:
        params["cross_attention"] = _parameters_from_t5_cross_attention(
            cross_atten, dst_layer=dst_layer.cross_attention
        )
    return params


def _parameters_from_t5_stack(src: hf_t5.T5Stack, *, dst_layer: Optional[BaseLayer] = None):
    num_layers = len(src.block)
    if isinstance(dst_layer, StackedTransformerLayer):
        params = {
            f"layer{i}": _parameters_from_t5_block(
                src.block[i], dst_layer=getattr(dst_layer, f"layer{i}")
            )
            for i in range(num_layers)
        }
    elif isinstance(dst_layer, RepeatedTransformerLayer):
        params = dict(
            repeat=VDict(
                layer=jax.tree.map(
                    lambda *inputs: jnp.stack(inputs),
                    *[
                        _parameters_from_t5_block(src.block[i], dst_layer=dst_layer.repeat.layer)
                        for i in range(num_layers)
                    ],
                )
            )
        )
    else:
        raise NotImplementedError(type(dst_layer))
    return params


def _parameters_from_t5_relative_pos_emb(src: hf_t5.T5Stack):
    relative_pos_emb = None
    for block in src.block:
        self_atten = list(block.layer)[0]
        atten: hf_t5.T5Attention = self_atten.SelfAttention
        if atten.has_relative_attention_bias:
            assert relative_pos_emb is None
            relative_pos_emb = atten.relative_attention_bias
    assert relative_pos_emb is not None
    return dict(emb=dict(weight=relative_pos_emb.weight))


def _parameters_from_t5_encoder_model(src: hf_t5.T5EncoderModel, *, dst_layer: Optional[BaseLayer]):
    """Returns parameters for a T5Encoder."""
    if dst_layer is None or not isinstance(dst_layer, T5Encoder):
        raise NotImplementedError("dst_layer should be T5Encoder")

    return dict(
        emb=dict(dropout={}, token_emb=torch_to_axlearn(src.shared)),
        transformer=torch_to_axlearn(src.encoder, dst_layer=dst_layer.transformer),
        output=dict(norm=torch_to_axlearn(src.encoder.final_layer_norm), dropout={}),
        relative_pos_emb=_parameters_from_t5_relative_pos_emb(src.encoder),
        attention_mask={},
    )


def _parameters_for_lm_head(src: torch.nn.Linear):
    if src.bias is not None:
        raise NotImplementedError("bias is not supported")
    params = torch_to_axlearn(src)
    return dict(weight=jnp.transpose(params["weight"], (1, 0)))


def _parameters_from_t5_for_conditional_generation(
    src: Union[hf_t5.T5ForConditionalGeneration, hf_mt5.MT5ForConditionalGeneration],
    dst_layer: Optional[BaseLayer],
):
    """Returns parameters for a T5EncoderDecoderModel."""
    if dst_layer is None or not isinstance(dst_layer, T5EncoderDecoderModel):
        raise NotImplementedError("dst_layer should be T5EncoderDecoderModel")

    return dict(
        shared_token_emb=torch_to_axlearn(src.shared),
        encoder=dict(
            relative_pos_emb=_parameters_from_t5_relative_pos_emb(src.encoder),
            transformer=_parameters_from_t5_stack(
                src.encoder, dst_layer=dst_layer.encoder.transformer
            ),
            output=dict(norm=torch_to_axlearn(src.encoder.final_layer_norm), dropout={}),
            emb=dict(dropout={}, token_emb={}),
            attention_mask={},
        ),
        decoder=dict(
            relative_pos_emb=_parameters_from_t5_relative_pos_emb(src.decoder),
            transformer=_parameters_from_t5_stack(
                src.decoder, dst_layer=dst_layer.decoder.transformer
            ),
            output_norm=torch_to_axlearn(src.decoder.final_layer_norm),
            lm_head=_parameters_for_lm_head(src.lm_head),
            attention_mask={},
            emb=dict(dropout={}, token_emb={}),
            output_dropout={},
        ),
    )


def _parameters_from_xlnet_attention(src: hf_xlnet.XLNetRelativeAttention):
    """Returns parameters for a TransformerAttentionLayer."""
    return dict(
        attention=dict(
            i_proj=VDict(
                qkv_proj=dict(
                    weight=torch.stack([src.q, src.k, src.v], dim=0),
                )
            ),
            o_proj=dict(weight=src.o),
            r_proj=dict(weight=src.r),
            u_bias=src.r_w_bias,
            v_bias=src.r_r_bias,
            dropout={},
            relative_pos_emb={},
            scale_query={},
            scale_key={},
        ),
        norm=torch_to_axlearn(src.layer_norm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_distilbert_embeddings(
    src: hf_distilbert.Embeddings, *, dst_layer: Optional[BaseLayer] = None
):
    dst = dict(
        token_emb=dict(weight=src.word_embeddings.weight),
        pos_emb=dict(weight=src.position_embeddings.weight),
        norm=torch_to_axlearn(src.LayerNorm),
        dropout={},
    )
    if dst_layer is None or isinstance(dst_layer.pos_emb, LearnedPositionalEmbedding):
        dst["pos_emb"]["weight"] = dst["pos_emb"]["weight"][None, ...]
    return dst


def _parameters_from_distilbert_feed_forward(
    ffn: hf_distilbert.FFN,
    output_layer_norm: torch.nn.LayerNorm,
):
    return dict(
        linear1=torch_to_axlearn(ffn.lin1),
        linear2=torch_to_axlearn(ffn.lin2),
        norm=torch_to_axlearn(output_layer_norm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_distilbert_attention_dense(
    src: hf_distilbert.MultiHeadSelfAttention,
):
    num_heads = src.n_heads
    per_head_dim = src.dim // src.n_heads
    i_proj = {}
    for src_proj, dst_proj in (
        ("q_lin", "q_proj"),
        ("k_lin", "k_proj"),
        ("v_lin", "v_proj"),
    ):
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        dense_params = dict(
            weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            bias=dense_params["bias"].reshape(num_heads, per_head_dim),
        )
        i_proj[dst_proj] = dense_params
    output_dense = src.out_lin
    o_proj = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    return dict(i_proj=i_proj, o_proj=o_proj, dropout={}, scale_query={}, scale_key={})


def _parameters_from_distilbert_attention(src: hf_distilbert.Transformer):
    return dict(
        attention=_parameters_from_distilbert_attention_dense(src.attention),
        norm=torch_to_axlearn(src.sa_layer_norm),
        dropout={},
        stochastic_depth={},
    )


def _parameters_from_distilbert_layer(src: hf_distilbert.Transformer):
    return dict(
        self_attention=_parameters_from_distilbert_attention(src),
        feed_forward=_parameters_from_distilbert_feed_forward(
            src.ffn,
            src.output_layer_norm,
        ),
    )


def _parameters_from_distilbert_encoder(src: hf_distilbert.DistilBertModel):
    return {
        f"layer{i}": _parameters_from_distilbert_layer(module) for i, module in enumerate(src.layer)
    }


def _parameters_from_distilbert_model(src: hf_distilbert.DistilBertModel):
    encoder = dict(
        emb=_parameters_from_distilbert_embeddings(src.embeddings),
        transformer=_parameters_from_distilbert_encoder(src.transformer),
        attention_mask={},
    )
    params = dict(encoder=encoder)
    return params


def _parameters_from_dit_timestep_embedding(src: torch.nn.Module):
    return dict(embed_proj=torch_to_axlearn(src.mlp[0]), output_proj=torch_to_axlearn(src.mlp[2]))


def _parameters_from_dit_label_embedding(src: torch.nn.Module):
    return dict(emb=torch_to_axlearn(src.embedding_table))


def _parameters_from_dit_adaln(src: torch.nn.Module):
    return dict(linear=torch_to_axlearn(src.adaLN_modulation[1]))


def _parameters_from_dit_ffn(src: torch.nn.Module):
    return dict(
        linear1=torch_to_axlearn(src.mlp.fc1),
        linear2=torch_to_axlearn(src.mlp.fc2),
        dropout1={},
        dropout2={},
        norm={},  # LayerNormStateless
    )


def _parameters_from_timm_vit_attn(src: timm_vit.Attention, dst_layer: MultiheadAttention):
    # src is a timm.models.vision_transformer.Attention layer.
    # Therefore, we cannot directly use the existed torch_to_axlearn to convert it.
    qkv_param = torch_to_axlearn(src.qkv)
    q_weight, k_weight, v_weight, *_ = np.split(qkv_param["weight"], 3, axis=1)
    weight = dict(q=q_weight, k=k_weight, v=v_weight)
    q_bias, k_bias, v_bias, *_ = np.split(qkv_param["bias"], 3, axis=0)
    bias = dict(q=q_bias, k=k_bias, v=v_bias)

    num_heads = dst_layer.hidden_dim() // dst_layer.per_head_dim()
    per_head_dim = dst_layer.per_head_dim()
    i_proj = {}
    for src_proj, dst_proj in (
        ("q", "q_proj"),
        ("k", "k_proj"),
        ("v", "v_proj"),
    ):
        dense_params = dict(
            weight=weight[src_proj].reshape(-1, num_heads, per_head_dim),
            bias=bias[src_proj].reshape(num_heads, per_head_dim),
        )
        i_proj[dst_proj] = dense_params
    output_dense = torch_to_axlearn(src.proj)
    # Transpose is required for the weight and bias converter.
    o_proj = dict(
        weight=output_dense["weight"].transpose().reshape(-1, num_heads, per_head_dim),
        bias=output_dense["bias"].transpose(),
    )
    return dict(i_proj=i_proj, o_proj=o_proj, dropout={})


def _parameters_from_dit_attn(src: torch.nn.Module, dst_layer: DiTAttentionLayer):
    return dict(
        norm={},  # LayerNormStateless
        attention=torch_to_axlearn(src.attn, dst_layer=dst_layer.attention),
    )


def _parameters_from_dit_block(src: torch.nn.Module, dst_layer: DiTBlock):
    return dict(
        adaln=torch_to_axlearn(src, dst_layer=dst_layer.adaln),
        attention=torch_to_axlearn(src, dst_layer=dst_layer.attention),
        feed_forward=torch_to_axlearn(src, dst_layer=dst_layer.feed_forward),
    )


def _parameters_from_dit_final_layer(src: torch.nn.Module, dst_layer: DiTFinalLayer):
    return dict(
        norm={},  # LayerNormStateless
        adaln=torch_to_axlearn(src, dst_layer=dst_layer.adaln),
        linear=torch_to_axlearn(src.linear),
    )


def _parameters_from_timm_vit_patch_embed(src: timm_vit.PatchEmbed):
    return dict(
        conv=torch_to_axlearn(src.proj),
    )


def _parameters_from_t5x_ff(src: NestedTensor, *, src_norm: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Mlp param dict into AXLearn TransformerFeedForwardLayer.

    Corresponding T5X layer is layer.MlpBlock.

    Args:
        src: Params corresponding to the T5X Feed Forward param dict.
        src_norm: Params corresponding to T5X pre-norm for feed forward.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Feed Forward.
    """
    dst = dict(dropout1={}, dropout2={}, stochastic_depth={})

    for layer, params in src.items():
        if layer == "wi_0":
            dst["linear1_0"] = _parameters_from_t5x_linear_like(params)
        elif layer == "wi_1":
            dst["linear1_1"] = _parameters_from_t5x_linear_like(params)
        elif layer == "wo":
            dst["linear2"] = _parameters_from_t5x_linear_like(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")
    dst["norm"] = _parameters_from_t5x_layer_norm(src_norm)

    return as_tensor(dst)


def _parameters_from_t5x_attn_proj(
    src: NestedTensor, *, dst_proj: str, dst_layer: BaseMultiheadLinear
) -> NestedTensor:
    """Imports params from T5X DenseGeneral param dict into AXLearn BaseMultiheadLinear model
    state.

    Corresponding T5X layer is just layer.DenseGeneral.

    Args:
        src: Params corresponding to the T5X Attention Projection param dict.
        dst_proj: AXLearn linear projection to transfer params to. Should be one of {q,k,v,o}_proj.
        dst_layer: AXLearn BaseMultiheadLinear model.
            This is usually MultiheadInputLinear or MultiheadOutputLinear.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If projection name is invalid.
    """
    if dst_proj not in [f"{x}_proj" for x in "qkvo"]:
        raise ValueError(f"Unsupported projection type. Got {dst_proj}")

    num_heads = dst_layer.config.num_heads
    per_head_dim = dst_layer.config.per_head_dim

    kernel = src["kernel"]
    if dst_proj[0] == "o":
        kernel = kernel.transpose(1, 0)
    # T5X bakes the projection scaling into initialization
    if dst_proj[0] == "q":
        kernel = kernel * per_head_dim**0.5

    # Note: T5X linear layers do not use bias.
    dst = dict(weight=kernel.reshape(-1, num_heads, per_head_dim))
    return as_tensor(dst)


def _parameters_from_t5x_lm_head(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X LM Head param dict into AXLearn LmHead model state.

    Args:
        src: Params corresponding to the T5X LM Head param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.
    """
    dst = dict(weight=src["kernel"].transpose(1, 0))
    return as_tensor(dst)


def _parameters_from_t5x_rel_pos_emb(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Relative Position Embeddings param dict
        into AXLearn T5RelativePositionalEmbedding model state.

    Corresponding T5X module is layers.RelativePositionBiases

    Args:
        src: Params corresponding to the T5X Rel Pos Emb param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.
    """
    dst = dict(emb=dict(weight=src["rel_embedding"].transpose(1, 0)))
    return as_tensor(dst)


def _parameters_from_t5x_attention(
    src: NestedTensor, *, src_norm: NestedTensor, dst_layer: TransformerAttentionLayer
) -> NestedTensor:
    """Imports params from T5X Attention param dict into AXLearn TransformerAttentionLayer model
    state.

    Corresponding T5X layer is layer.MultiHeadDotProductAttention.

    Args:
        src: Params corresponding to the T5X Attention param dict.
        src_norm: Params corresponding to T5X pre-norm for attention.
        dst_layer: AXLearn TransformerAttentionLayer model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Attention.
        NotImplementedError: If dst_layer.attention.i_proj is not supported.
    """
    dst = dict(dropout={}, stochastic_depth={}, attention=dict(i_proj={}, dropout={}))
    i_proj = {}

    for layer, params in src.items():
        if layer not in {"query", "key", "value", "out"}:
            raise ValueError(f"Unsupported layer: {layer}")

        # query, key, value, out => q_proj, k_proj, v_proj, o_proj
        proj_key = f"{layer[0]}_proj"

        if proj_key.startswith("o"):
            dst["attention"][proj_key] = _parameters_from_t5x_attn_proj(
                params, dst_proj=proj_key, dst_layer=dst_layer.attention.o_proj
            )
            continue

        if isinstance(dst_layer.attention.i_proj, QKVLinear):
            proj_layer = getattr(dst_layer.attention.i_proj, proj_key)
        elif isinstance(dst_layer.attention.i_proj, FusedQKVLinear):
            proj_layer = dst_layer.attention.i_proj.qkv_proj
        else:
            raise NotImplementedError(type(dst_layer.attention.i_proj))
        i_proj[proj_key] = _parameters_from_t5x_attn_proj(
            params, dst_proj=proj_key, dst_layer=proj_layer
        )

    if isinstance(dst_layer.attention.i_proj, QKVLinear):
        for proj_key in ("q_proj", "k_proj", "v_proj"):
            dst["attention"]["i_proj"][proj_key] = i_proj[proj_key]
    elif isinstance(dst_layer.attention.i_proj, FusedQKVLinear):
        dst["attention"]["i_proj"] = VDict(
            qkv_proj=dict(
                jax.tree.map(
                    lambda *inputs: jnp.stack(inputs),
                    *[i_proj[proj_key] for proj_key in ("q_proj", "k_proj", "v_proj")],
                )
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.attention.i_proj))

    dst["norm"] = _parameters_from_t5x_layer_norm(src_norm)
    return as_tensor(dst)


def _parameters_from_t5x_transformer_layer(
    src: NestedTensor, dst_layer: TransformerLayer
) -> NestedTensor:
    """Imports parameters from T5X Transformer param dict into AXLearn TransformerLayer model state.

    Corresponding T5X layer is network.EncoderLayer and network.DecoderLayer.

    Args:
        src: Params corresponding to the T5X Transformer param dict.
        dst_layer: AXLearn TransformerLayer model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Transformer.
    """
    dst = {}

    for layer_name in dst_layer.children:
        if layer_name == "cross_attention":
            dst["cross_attention"] = _parameters_from_t5x_attention(
                src["encoder_decoder_attention"],
                src_norm=src["pre_cross_attention_layer_norm"],
                dst_layer=dst_layer.cross_attention,
            )
        elif layer_name == "self_attention":
            dst["self_attention"] = _parameters_from_t5x_attention(
                src.get("self_attention") or src.get("attention"),
                src_norm=src.get("pre_self_attention_layer_norm")
                or src.get("pre_attention_layer_norm"),
                dst_layer=dst_layer.self_attention,
            )
        elif layer_name == "feed_forward":
            dst["feed_forward"] = _parameters_from_t5x_ff(
                src["mlp"], src_norm=src["pre_mlp_layer_norm"]
            )
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")

    return as_tensor(dst)


# pylint: disable-next=too-many-branches
def _parameters_from_t5x_decoder(src: NestedTensor, dst_layer: T5Decoder) -> NestedTensor:
    """Imports parameters from T5X Decoder param dict into AXLearn T5Decoder model state.

    Args:
        src: Params corresponding to the T5X Decoder param dict.
        dst_layer: AXLearn T5Decoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Decoder.
        NotImplementedError: If dst_layer.transformer is not supported.
    """
    dst = dict(
        transformer={}, emb=dict(dropout={}, token_emb={}), attention_mask={}, output_dropout={}
    )
    transformer_layers = {}

    for layer, params in src.items():
        if layer.startswith("layers_"):
            # T5X layers are named "layers_{i}".
            layer_index = int(layer.split("_")[-1])
            if isinstance(dst_layer.transformer, StackedTransformerLayer):
                # AXLearn stacked layers are named "layer{i}".
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, getattr(dst_layer.transformer, f"layer{layer_index}")
                )
            elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, dst_layer.transformer.repeat.layer
                )
        elif layer == "decoder_norm":
            dst["output_norm"] = _parameters_from_t5x_layer_norm(params)
        elif layer == "logits_dense":
            dst["lm_head"] = _parameters_from_t5x_lm_head(params)
        elif layer == "relpos_bias":
            dst["relative_pos_emb"] = _parameters_from_t5x_rel_pos_emb(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    num_axlearn_decoder_layers = dst_layer.transformer.config.num_layers
    num_t5x_decoder_layers = len(transformer_layers)
    if num_axlearn_decoder_layers != num_t5x_decoder_layers:
        raise ValueError(
            f"Number of decoder layers does not match. T5X: {num_t5x_decoder_layers}, "
            f"AXLearn: {num_axlearn_decoder_layers}."
        )

    # Traverse transformer_layers by index, to ensure that layers are ordered even if the source
    # layers are visited out-of-order.
    if isinstance(dst_layer.transformer, StackedTransformerLayer):
        for i in range(num_axlearn_decoder_layers):
            dst["transformer"][f"layer{i}"] = transformer_layers[i]
    elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
        dst["transformer"]["repeat"] = VDict(
            # pylint: disable-next=no-value-for-parameter
            layer=jax.tree.map(
                lambda *inputs: jnp.stack(inputs),
                *[transformer_layers[i] for i in range(num_axlearn_decoder_layers)],
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.transformer))

    return as_tensor(dst)


def _parameters_from_t5x_encoder(src: NestedTensor, dst_layer: T5Encoder) -> NestedTensor:
    """Imports parameters from T5X Encoder param dict into AXLearn T5Encoder model state.

    Args:
        src: Params corresponding to the T5X Encoder param dict.
        dst_layer: AXLearn T5Encoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
        NotImplementedError: If dst_layer.transformer is not supported.
    """
    dst = dict(transformer={}, emb=dict(dropout={}, token_emb={}), attention_mask={})
    transformer_layers = {}

    for layer, params in src.items():
        if layer.startswith("layers_"):
            # T5X layers are named "layers_{i}".
            layer_index = int(layer.split("_")[-1])
            if isinstance(dst_layer.transformer, StackedTransformerLayer):
                # AXLearn stacked layers are named "layer{i}".
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, getattr(dst_layer.transformer, f"layer{layer_index}")
                )
            elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, dst_layer.transformer.repeat.layer
                )
        elif layer == "encoder_norm":
            dst["output"] = dict(norm=_parameters_from_t5x_layer_norm(params), dropout={})
        elif layer == "relpos_bias":
            dst["relative_pos_emb"] = _parameters_from_t5x_rel_pos_emb(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    num_axlearn_encoder_layers = dst_layer.transformer.config.num_layers
    num_t5x_encoder_layers = len(transformer_layers)
    if num_axlearn_encoder_layers != num_t5x_encoder_layers:
        raise ValueError(
            f"Number of encoder layers does not match. T5X: {num_t5x_encoder_layers}, "
            f"AXLearn: {num_axlearn_encoder_layers}."
        )

    # Traverse transformer_layers by index, to ensure that layers are ordered even if the source
    # layers are visited out-of-order.
    if isinstance(dst_layer.transformer, StackedTransformerLayer):
        for i in range(num_axlearn_encoder_layers):
            dst["transformer"][f"layer{i}"] = transformer_layers[i]
    elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
        dst["transformer"]["repeat"] = VDict(
            # pylint: disable-next=no-value-for-parameter
            layer=jax.tree.map(
                lambda *inputs: jnp.stack(inputs),
                *[transformer_layers[i] for i in range(num_axlearn_encoder_layers)],
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.transformer))

    return as_tensor(dst)


def _parameters_from_t5x_layer_norm(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Layer Norm param dict into AXLearn RMSNorm model state.

    Corresponding T5X module is layers.LayerNorm

    Args:
        src: Params corresponding to the T5X LayerNorm param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer == "scale":
            dst["scale"] = params
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)


def _parameters_from_t5x_linear_like(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X linear param dict into AXLearn Embedding or Linear model state.

    Corresponding T5X modules are layers.Embed and layers.DenseGeneral.

    Args:
        src: Params corresponding to the T5X LayerNorm param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer in {"embedding", "kernel"}:
            dst["weight"] = params
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)


def parameters_from_t5x_encoder_decoder(
    src: NestedTensor, dst_layer: T5EncoderDecoderModel
) -> NestedTensor:
    """Imports parameters from T5X EncoderDecoder param dict into AXLearn T5EncoderDecoder model
    state.

    The corresponding T5X Flax model is network.Transformer.

    Args:
        src: Params corresponding to the T5X EncoderDecoder param dict.
        dst_layer: AXLearn T5EncoderDecoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X EncoderDecoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer == "decoder":
            dst["decoder"] = _parameters_from_t5x_decoder(params, dst_layer.decoder)
        elif layer == "encoder":
            dst["encoder"] = _parameters_from_t5x_encoder(params, dst_layer.encoder)
        elif layer == "token_embedder":
            dst["shared_token_emb"] = _parameters_from_t5x_linear_like(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)


def _permute_q_k_for_rope(vector: torch.Tensor) -> torch.Tensor:
    """Permutes q and k vector because transformers package has a different implementation of RoPE.

    The revert operation of the following:
    https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py#L136
    """
    n, h, d = vector.shape
    vector = vector.view(n, 2, h // 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def parameters_from_llama_3(llama: LlamaForCausalLM, state: dict) -> dict:
    """Converts llama weights from huggingface model to fuji state.

    The following model are supported and tested:
    - (fuji_model_name="fuji-1B-v3", llama_model_name="Llama-3.2-1B")
    - (fuji_model_name="fuji-3B-v3", llama_model_name="Llama-3.2-3B")
    - (fuji_model_name="fuji-8B-v3", llama_model_name="Llama-3.1-8B")
    - (fuji_model_name="fuji-70B-v3", llama_model_name="Llama-3.1-70B")

    Args:
        llama: A Llama model with type LlamaForCausalLM.
        state: The state of a fuji model.

    Returns:
        NestedTensor containing the same structure as state, but the weights are from llama.
    """
    # Copy the nested dict. No need to deep copy the data since it will be replaced.
    state = jax.tree.map(lambda x: x, state)
    if "lm_head" in state["decoder"]:
        if id(llama.model.embed_tokens.weight) == id(llama.lm_head.weight):
            raise ValueError("The embed_tokens and lm_head should not share weights.")
        state["decoder"]["lm_head"]["weight"] = llama.lm_head.weight
    elif id(llama.model.embed_tokens.weight) != id(llama.lm_head.weight):
        raise ValueError("The embed_tokens and lm_head should share weights")

    state["decoder"]["emb"]["token_emb"]["weight"] = llama.model.embed_tokens.weight
    gate_proj = []
    up_proj = []
    down_proj = []
    qkv = []
    o = []
    input_norm = []
    post_attention_norm = []
    o_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "o_proj"
    ]["weight"].shape
    i_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "i_proj"
    ]["i_proj"]["qkv_proj"][
        "weight"
    ].shape  # (n_layers, d, n, h)

    for layer in llama.model.layers:
        gate_proj.append(layer.mlp.gate_proj.weight.transpose(0, 1))
        up_proj.append(layer.mlp.up_proj.weight.transpose(0, 1))
        down_proj.append(layer.mlp.down_proj.weight.transpose(0, 1))

        vector = torch.concat(
            [
                _permute_q_k_for_rope(
                    layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                ),
                _permute_q_k_for_rope(
                    layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                ),
                layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
            ],
            dim=0,
        ).permute(2, 0, 1)
        qkv.append(vector)
        o.append(layer.self_attn.o_proj.weight.reshape(-1, o_shape[-2], o_shape[-1]))
        input_norm.append(layer.input_layernorm.weight)
        post_attention_norm.append(layer.post_attention_layernorm.weight)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"][
        "weight"
    ] = torch.stack(gate_proj)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"][
        "weight"
    ] = torch.stack(up_proj)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"][
        "weight"
    ] = torch.stack(down_proj)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"][
        "weight"
    ] = torch.stack(o)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["i_proj"][
        "i_proj"
    ]["qkv_proj"]["weight"] = torch.stack(qkv)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"][
        "scale"
    ] = torch.stack(input_norm)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"][
        "scale"
    ] = torch.stack(post_attention_norm)
    state["decoder"]["output_norm"]["scale"] = llama.model.norm.weight
    return as_tensor(state)
