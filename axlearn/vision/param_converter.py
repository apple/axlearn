# Copyright © 2023 Apple Inc.

"""Vision param converters."""
import copy
from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from transformers.models.clip import modeling_clip as hf_clip

from axlearn.common.attention import (
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.encoder import Encoder
from axlearn.common.multi_stream_model import MultiStreamModel
from axlearn.common.param_converter import (
    as_torch_tensor,
    axlearn_to_torch,
    register_axlearn_to_torch,
    register_torch_to_axlearn,
    torch_to_axlearn,
)
from axlearn.common.text_encoder import TextEmbeddingEncoder
from axlearn.common.utils import NestedTensor
from axlearn.common.vision_transformer import Encoder1D, VisionTransformer
from axlearn.vision.clip import CLIPImageStreamEncoder, CLIPTextStreamEncoder
from axlearn.vision.image_classification import ImageClassificationModel
from axlearn.vision.mobilenets_blocks import MobileBlockType


@register_axlearn_to_torch(src=[MultiStreamModel], dst=[hf_clip.CLIPModel])
def parameters_to_hf_clip_model(
    *, src: MultiStreamModel, params: NestedTensor, dst: hf_clip.CLIPModel
):
    axlearn_to_torch(
        src.textual_encoder.text_encoder,
        params["textual_encoder"]["text_encoder"],
        dst.text_model,
    )
    axlearn_to_torch(
        src.visual_encoder.image_encoder,
        params["visual_encoder"]["image_encoder"],
        dst.vision_model,
    )
    axlearn_to_torch(
        src.visual_encoder.output_proj,
        params["visual_encoder"]["output_proj"],
        dst.visual_projection,
    )
    axlearn_to_torch(
        src.textual_encoder.output_proj,
        params["textual_encoder"]["output_proj"],
        dst.text_projection,
    )


@register_axlearn_to_torch(src=[MultiStreamModel], dst=[hf_clip.CLIPVisionModelWithProjection])
def parameters_to_hf_clip_vision_model_with_projection(
    *, src: MultiStreamModel, params: NestedTensor, dst: hf_clip.CLIPVisionModelWithProjection
):
    axlearn_to_torch(
        src.visual_encoder.image_encoder,
        params["visual_encoder"]["image_encoder"],
        dst.vision_model,
    )
    axlearn_to_torch(
        src.visual_encoder.output_proj,
        params["visual_encoder"]["output_proj"],
        dst.visual_projection,
    )


@register_axlearn_to_torch(
    src=[TextEmbeddingEncoder, CLIPTextStreamEncoder], dst=[hf_clip.CLIPTextTransformer]
)
def parameters_to_hf_clip_text_transformer(
    *,
    src: Union[TextEmbeddingEncoder, CLIPTextStreamEncoder],
    params: NestedTensor,
    dst: hf_clip.CLIPTextTransformer,
):
    if isinstance(src, CLIPTextStreamEncoder):
        src = src.text_encoder
        params = params["text_encoder"]
    axlearn_to_torch(src.encoder.emb, params["encoder"]["emb"], dst.embeddings)
    axlearn_to_torch(src.encoder, params["encoder"], dst.encoder)
    axlearn_to_torch(
        src.encoder.output,
        params["encoder"]["output"],
        dst.final_layer_norm,
    )


@register_axlearn_to_torch(src=[Encoder, Encoder1D], dst=[hf_clip.CLIPEncoder])
def parameters_to_hf_clip_encoder(*, src: Encoder, params: NestedTensor, dst: hf_clip.CLIPEncoder):
    for i, dst_layer in enumerate(dst.layers):
        if isinstance(src.transformer, StackedTransformerLayer):
            axlearn_to_torch(
                getattr(src.transformer, f"layer{i}"),
                params["transformer"][f"layer{i}"],
                dst_layer,
            )
        else:
            assert isinstance(src.transformer, RepeatedTransformerLayer)
            axlearn_to_torch(
                src.transformer.repeat.layer,
                jax.tree.map(lambda x, idx=i: x[idx], params["transformer"]["repeat"]["layer"]),
                dst_layer,
            )


def _parameters_to_hf_clip_attention(src: NestedTensor, dst: hf_clip.CLIPAttention):
    # Copy Q,K,V. In Hugging Face, this dense layer is in *SelfAttention layer.
    all_head_dim = dst.embed_dim
    for src_proj, dst_proj in (
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
    ):
        src_dense = src["i_proj"][src_proj]
        dst_dense = getattr(dst, dst_proj)
        dst_dense.weight.data = as_torch_tensor(
            src_dense["weight"].reshape(-1, all_head_dim)
        ).transpose(
            0, 1
        )  # pytype: disable=attribute-error
        dst_dense.bias.data = as_torch_tensor(src_dense["bias"].reshape(all_head_dim))

    dst_output = dst.out_proj
    src_output = src["o_proj"]
    dst_output.weight.data = as_torch_tensor(src_output["weight"].reshape(-1, all_head_dim))
    dst_output.bias.data = as_torch_tensor(src_output["bias"].reshape(all_head_dim))


@register_axlearn_to_torch(src=[TransformerLayer], dst=[hf_clip.CLIPEncoderLayer])
def parameters_to_hf_clip_encoder_layer(
    *, src: TransformerLayer, params: NestedTensor, dst: hf_clip.CLIPEncoderLayer
):
    _parameters_to_hf_clip_attention(params["self_attention"]["attention"], dst.self_attn)
    axlearn_to_torch(src.self_attention.norm, params["self_attention"]["norm"], dst.layer_norm1)
    axlearn_to_torch(src.feed_forward, params["feed_forward"], dst.mlp)
    axlearn_to_torch(src.feed_forward.norm, params["feed_forward"]["norm"], dst.layer_norm2)


@register_axlearn_to_torch(src=[TransformerFeedForwardLayer], dst=[hf_clip.CLIPMLP])
def parameters_to_hf_clip_mlp(
    *, src: TransformerFeedForwardLayer, params: NestedTensor, dst: hf_clip.CLIPMLP
):
    axlearn_to_torch(src.linear1, params["linear1"], dst.fc1)
    axlearn_to_torch(src.linear2, params["linear2"], dst.fc2)


@register_axlearn_to_torch(src=[TransformerTextEmbeddings], dst=[hf_clip.CLIPTextEmbeddings])
def parameters_to_hf_clip_text_embeddings(
    *, src: TransformerTextEmbeddings, params: NestedTensor, dst: hf_clip.CLIPTextEmbeddings
):
    pos = copy.deepcopy(params["pos_emb"])
    pos["weight"] = jnp.squeeze(pos["weight"], 0)
    axlearn_to_torch(src.token_emb, params["token_emb"], dst.token_embedding)
    axlearn_to_torch(src.token_emb, pos, dst.position_embedding)


@register_axlearn_to_torch(
    src=[VisionTransformer, CLIPImageStreamEncoder], dst=[hf_clip.CLIPVisionTransformer]
)
def parameters_to_hf_clip_vision_transformer(
    *,
    src: Union[VisionTransformer, CLIPImageStreamEncoder],
    params: NestedTensor,
    dst: hf_clip.CLIPVisionTransformer,
):
    if isinstance(src, CLIPImageStreamEncoder):
        src = src.image_encoder
        params = params["image_encoder"]
    axlearn_to_torch(src, params, dst.embeddings)
    axlearn_to_torch(
        src.encoder_1d.input_norm,
        params["encoder_1d"]["input_norm"],
        dst.pre_layrnorm,
    )
    axlearn_to_torch(src.encoder_1d, params["encoder_1d"], dst.encoder)
    axlearn_to_torch(
        src.encoder_1d.output_norm,
        params["encoder_1d"]["output_norm"],
        dst.post_layernorm,
    )


@register_axlearn_to_torch(src=[VisionTransformer], dst=[hf_clip.CLIPVisionEmbeddings])
def parameters_to_hf_clip_vision_embeddings(
    *, src: VisionTransformer, params: NestedTensor, dst: hf_clip.CLIPVisionEmbeddings
):
    axlearn_to_torch(
        src.visual_embed.convert_to_sequence.conv,
        params["visual_embed"]["convert_to_sequence"]["conv"],
        dst.patch_embedding,
    )
    axlearn_to_torch(
        src.encoder_1d.pos_emb, params["encoder_1d"]["pos_emb"], dst.position_embedding
    )
    dst.class_embedding.data = as_torch_tensor(params["cls_token"]).reshape(
        -1
    )  # pytype: disable=attribute-error


def _parameters_from_clip_visual_embedding(src: hf_clip.CLIPVisionEmbeddings) -> NestedTensor:
    # The CLIP uses patch_embedding. Not patch_embeddings as ViT.
    params = dict(conv=torch_to_axlearn(src.patch_embedding))
    # bias is not existed in AXLearn CLIP conv layer.
    assert params["conv"].pop("bias") is None
    return params


def _parameters_from_clip_visual_encoder(
    src_enc: hf_clip.CLIPEncoder,
    src_emb: hf_clip.CLIPVisionEmbeddings,
    src_pre_norm: torch.nn.LayerNorm,
    src_output_norm: torch.nn.LayerNorm,
) -> NestedTensor:
    dst = dict(
        drop_token={},
        input_dropout={},
        transformer={},
        pos_emb=dict(weight=src_emb.position_embedding.weight[None, ...]),
        input_norm=torch_to_axlearn(src_pre_norm),
        output_norm=torch_to_axlearn(src_output_norm),
    )
    for layer_i, layer in enumerate(src_enc.layers):
        dst["transformer"][f"layer{layer_i}"] = torch_to_axlearn(layer)
    return dst


@register_torch_to_axlearn(src=[hf_clip.CLIPVisionTransformer])
def parameters_from_clip_vision_transformer(
    src: hf_clip.CLIPVisionTransformer, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    # The HF CLIP cls_token shape is [dim].
    # The AXLearn CLIP cls_token shape is [1, num_cls_tokens, dim].
    # num_cls_token = 1 for CLIP.
    cls_token = torch.reshape(src.embeddings.class_embedding, (1, 1, -1))
    return {
        "image_encoder": dict(
            cls_token=cls_token,
            visual_embed=dict(
                convert_to_sequence=_parameters_from_clip_visual_embedding(src.embeddings),
            ),
            encoder_1d=_parameters_from_clip_visual_encoder(
                src.encoder,
                src.embeddings,
                src.pre_layrnorm,
                src.post_layernorm,
                # pre_layrnorm is not a typo.
            ),
            pooler={},
        ),
    }


def _parameters_from_clip_attention_dense(src: hf_clip.CLIPAttention) -> NestedTensor:
    num_heads = src.num_heads
    per_head_dim = src.head_dim
    i_proj = {}
    for src_proj, dst_proj in (
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
    ):
        dense = getattr(src, src_proj)
        dense_params = torch_to_axlearn(dense)
        dense_params = dict(
            weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            bias=dense_params["bias"].reshape(num_heads, per_head_dim),
        )
        i_proj[dst_proj] = dense_params
    output_dense = src.out_proj
    o_proj = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    return dict(i_proj=i_proj, o_proj=o_proj, dropout={}, scale_key={}, scale_query={}, kv_cache={})


def _parameters_from_clip_attention(
    src: hf_clip.CLIPAttention, src_norm: torch.nn.LayerNorm
) -> NestedTensor:
    return dict(
        attention=_parameters_from_clip_attention_dense(
            src,
        ),
        dropout={},
        stochastic_depth={},
        norm=torch_to_axlearn(src_norm),
    )


def _parameters_from_clip_feed_forward(
    clip_mlp: hf_clip.CLIPMLP, norm: torch.nn.LayerNorm
) -> NestedTensor:
    return dict(
        linear1=torch_to_axlearn(clip_mlp.fc1),
        linear2=torch_to_axlearn(clip_mlp.fc2),
        dropout1={},
        dropout2={},
        stochastic_depth={},
        norm=torch_to_axlearn(norm),
    )


@register_torch_to_axlearn(src=[hf_clip.CLIPEncoderLayer])
def parameters_from_clip_layer(
    src: hf_clip.CLIPEncoderLayer, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    return dict(
        self_attention=_parameters_from_clip_attention(src.self_attn, src.layer_norm1),
        feed_forward=_parameters_from_clip_feed_forward(src.mlp, src.layer_norm2),
    )


@register_torch_to_axlearn(src=[hf_clip.CLIPTextEmbeddings])
def parameters_from_clip_text_embedding(
    src: hf_clip.CLIPTextEmbeddings, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    if src.token_embedding.weight.shape[0] == 49408:
        # This means we want to load the parameter for CLIP-OpenAI evaluation.
        # OpenAI vocab size is 49408.
        dim = src.token_embedding.weight.shape[1]
        # We append <PAD> token in the end of the vocab.
        # Therefore, we append an empty embedding to the OpenAI CLIP token_embedding.
        token_embedding_weight = torch.cat((src.token_embedding.weight, torch.zeros((1, dim))))
    else:
        # This means we want to load the parameter for AXLearn->HF round-trip unittest.
        token_embedding_weight = src.token_embedding.weight
    return dict(
        token_emb=dict(weight=token_embedding_weight),
        pos_emb=dict(weight=src.position_embedding.weight[None, ...]),
        dropout={},
    )


@register_torch_to_axlearn(src=[hf_clip.CLIPEncoder])
def parameters_from_clip_encoder(
    src_enc: hf_clip.CLIPEncoder, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    params = {}
    for layer_i, layer in enumerate(src_enc.layers):
        params[f"layer{layer_i}"] = torch_to_axlearn(layer)
    return params


@register_torch_to_axlearn(src=[hf_clip.CLIPTextTransformer])
def parameters_from_clip_text_transformer(
    src: hf_clip.CLIPTextTransformer, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    return {
        "text_encoder": dict(
            encoder=dict(
                emb=parameters_from_clip_text_embedding(src.embeddings),
                transformer=parameters_from_clip_encoder(src.encoder),
                output=torch_to_axlearn(src.final_layer_norm),
                attention_mask={},
            ),
            pooler={},
        ),
    }


@register_torch_to_axlearn(src=[hf_clip.CLIPModel])
def parameters_from_clip_model(
    src: hf_clip.CLIPModel, dst: Optional[Union[BaseLayer, type]] = None
) -> NestedTensor:
    del dst
    visual_encoder = torch_to_axlearn(src.vision_model)
    visual_encoder["output_proj"] = torch_to_axlearn(src.visual_projection)
    visual_encoder["output_norm"] = {}
    textual_encoder = torch_to_axlearn(src.text_model)
    textual_encoder["output_proj"] = torch_to_axlearn(src.text_projection)
    textual_encoder["output_norm"] = {}
    return dict(
        visual_encoder=visual_encoder,
        textual_encoder=textual_encoder,
        fusion_network=dict(
            log_logit_scale=src.logit_scale.reshape(1),
        ),
    )


class ClassificationModelParamConverter:
    """Base class for converting parameters of classification models from PyTorch to JAX."""

    @staticmethod
    def _params_from_conv(ref: torch.nn.Module) -> NestedTensor:
        param = {
            # torch conv uses layout (output, input, H, W)
            # while axlearn uses (H, W, input, output).
            "weight": ref.weight.permute(2, 3, 1, 0),
        }
        if hasattr(ref, "bias"):
            param["bias"] = ref.bias
        return param

    @staticmethod
    def _params_from_linear(ref: torch.nn.Module) -> NestedTensor:
        return {
            # torch linear uses layout (output, input)
            # while axlearn uses (input, output).
            "weight": ref.weight.transpose(1, 0),
            "bias": ref.bias,
        }

    @staticmethod
    def _params_from_bn(ref: torch.nn.Module) -> NestedTensor:
        return {
            "scale": ref.weight,
            "bias": ref.bias,
            "moving_mean": ref.running_mean,
            "moving_variance": ref.running_var,
        }


class MobileNetsParamConverter(ClassificationModelParamConverter):
    """Converter for parameters of MobileNets from PyTorch to JAX."""

    @staticmethod
    def _params_from_squeeze_excitation(ref: torch.nn.Module) -> NestedTensor:
        params = {}
        if hasattr(ref, "conv_expand"):
            params["expand"] = MobileNetsParamConverter._params_from_conv(ref.conv_expand)
        if hasattr(ref, "conv_reduce"):
            params["reduce"] = MobileNetsParamConverter._params_from_conv(ref.conv_reduce)
        return params

    @staticmethod
    def _params_from_block(block_type: MobileBlockType, ref: torch.nn.Module) -> NestedTensor:
        if block_type == MobileBlockType.CONV_BN_ACT:
            params = {
                "conv": MobileNetsParamConverter._params_from_conv(ref.conv),
                "bn": MobileNetsParamConverter._params_from_bn(ref.bn1),
            }
        elif block_type == MobileBlockType.DEPTHWISE_SEPARABLE:
            params = {
                "conv_dw": MobileNetsParamConverter._params_from_conv(ref.conv_dw),
                "bn_dw": MobileNetsParamConverter._params_from_bn(ref.bn1),
                "conv_pw": MobileNetsParamConverter._params_from_conv(ref.conv_pw),
                "bn_pw": MobileNetsParamConverter._params_from_bn(ref.bn2),
            }
        elif block_type == MobileBlockType.INVERTED_BOTTLENECK:
            params = {
                "conv_exp": MobileNetsParamConverter._params_from_conv(ref.conv_pw),
                "bn_exp": MobileNetsParamConverter._params_from_bn(ref.bn1),
                "conv_dw": MobileNetsParamConverter._params_from_conv(ref.conv_dw),
                "bn_dw": MobileNetsParamConverter._params_from_bn(ref.bn2),
                "conv_pw": MobileNetsParamConverter._params_from_conv(ref.conv_pwl),
                "bn_pw": MobileNetsParamConverter._params_from_bn(ref.bn3),
            }
        elif block_type == MobileBlockType.FUSED_INVERTED_BOTTLENECK:
            params = {
                "conv_exp": MobileNetsParamConverter._params_from_conv(ref.conv_exp),
                "bn_exp": MobileNetsParamConverter._params_from_bn(ref.bn1),
                "conv_pw": MobileNetsParamConverter._params_from_conv(ref.conv_pwl),
                "bn_pw": MobileNetsParamConverter._params_from_bn(ref.bn2),
            }
        if hasattr(ref, "se"):
            params["se"] = MobileNetsParamConverter._params_from_squeeze_excitation(ref.se)
        return params

    @staticmethod
    def _params_from_stem(ref: torch.nn.ModuleList) -> NestedTensor:
        return {
            "conv": MobileNetsParamConverter._params_from_conv(ref.conv_stem),
            "bn": MobileNetsParamConverter._params_from_bn(ref.bn1),
        }

    @staticmethod
    def _params_from_embedding_layer(ref: torch.nn.ModuleList) -> NestedTensor:
        params = {
            "conv_pw": MobileNetsParamConverter._params_from_conv(ref.conv_head),
        }
        if hasattr(ref, "bn2"):
            params["bn"] = MobileNetsParamConverter._params_from_bn(ref.bn2)
        return params

    @staticmethod
    def _params_from_backbone(
        cfg: ImageClassificationModel.Config, ref: torch.nn.ModuleList
    ) -> NestedTensor:
        params = {
            "stem": MobileNetsParamConverter._params_from_stem(ref),
            "embedding_layer": MobileNetsParamConverter._params_from_embedding_layer(ref),
        }
        block_defs = cfg.backbone.block_defs
        for stage_idx, (stage_defs, ref_stage) in enumerate(zip(block_defs, ref.blocks)):
            for block_idx, (stage_def, ref_block) in enumerate(zip(stage_defs, ref_stage)):
                block_type: MobileBlockType = stage_def[0]
                block_str = f"blocks_{stage_idx}_{block_idx}"
                params[block_str] = MobileNetsParamConverter._params_from_block(
                    block_type, ref_block
                )
        return params

    @staticmethod
    def _params_from_classifier(ref: torch.nn.ModuleList) -> NestedTensor:
        return MobileNetsParamConverter._params_from_linear(ref.classifier)

    @staticmethod
    def params_from_model(
        cfg: ImageClassificationModel.Config, ref: torch.nn.Module
    ) -> NestedTensor:
        return {
            "backbone": MobileNetsParamConverter._params_from_backbone(cfg, ref),
            "classifier": MobileNetsParamConverter._params_from_classifier(ref),
        }


class ResNetParamConverter(ClassificationModelParamConverter):
    """Converter for ResNet parameters from PyTorch to JAX."""

    @staticmethod
    def _params_from_downsample(ref: torch.nn.ModuleList) -> NestedTensor:
        return {
            "conv": ResNetParamConverter._params_from_conv(ref[0]),
            "norm": ResNetParamConverter._params_from_bn(ref[1]),
        }

    @staticmethod
    def _params_from_block(ref: torch.nn.Module) -> NestedTensor:
        params = {
            "conv1": ResNetParamConverter._params_from_conv(ref.conv1),
            "norm1": ResNetParamConverter._params_from_bn(ref.bn1),
            "conv2": ResNetParamConverter._params_from_conv(ref.conv2),
            "norm2": ResNetParamConverter._params_from_bn(ref.bn2),
        }
        if hasattr(ref, "conv3"):
            params["conv3"] = ResNetParamConverter._params_from_conv(ref.conv3)
        if hasattr(ref, "bn3"):
            params["norm3"] = ResNetParamConverter._params_from_bn(ref.bn3)
        if getattr(ref, "downsample"):
            params["downsample"] = ResNetParamConverter._params_from_downsample(ref.downsample)
        return params

    @staticmethod
    def _params_from_stage(ref: torch.nn.ModuleList) -> NestedTensor:
        return {
            f"block{i}": ResNetParamConverter._params_from_block(block)
            for i, block in enumerate(ref)
        }

    @staticmethod
    def _params_from_stem(ref: torch.nn.ModuleList) -> NestedTensor:
        return {
            "conv1": ResNetParamConverter._params_from_conv(ref.conv1),
            "norm1": ResNetParamConverter._params_from_bn(ref.bn1),
        }

    @staticmethod
    def _params_from_backbone(ref: torch.nn.ModuleList) -> NestedTensor:
        return {
            "stem": ResNetParamConverter._params_from_stem(ref),
            **{
                f"stage{i}": ResNetParamConverter._params_from_stage(getattr(ref, f"layer{i + 1}"))
                for i in range(4)
            },
        }

    @staticmethod
    def _params_from_classifier(ref: torch.nn.ModuleList) -> NestedTensor:
        return ResNetParamConverter._params_from_linear(ref.fc)

    @staticmethod
    def params_from_model(ref: torch.nn.Module) -> NestedTensor:
        return {
            "backbone": ResNetParamConverter._params_from_backbone(ref),
            "classifier": ResNetParamConverter._params_from_classifier(ref),
        }
