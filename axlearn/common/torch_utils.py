# Copyright © 2023 Apple Inc.

"""Utilities relating to PyTorch."""

from typing import Optional, Type, Union

import torch
from absl import logging

try:
    # pylint: disable=import-error
    from fairseq.modules import conformer_layer as fairseq_conformer  # type: ignore
    from fairseq.modules import (
        espnet_multihead_attention as fairseq_espnet_multihead_attention,  # type: ignore
    )

    # pylint: enable=import-error

    _FAIRSEQ_INSTALLED = True
except ModuleNotFoundError:
    logging.warning("fairseq is not installed -- param conversion for fairseq modules will fail.")
    _FAIRSEQ_INSTALLED = False

from axlearn.common.base_layer import BaseLayer
from axlearn.common.layers import Linear
from axlearn.common.param_converter import torch_to_axlearn
from axlearn.common.utils import NestedTensor, VDict, as_tensor

if _FAIRSEQ_INSTALLED:

    def _parameters_from_fairseq_conformer_layer(src: fairseq_conformer.ConformerEncoderLayer):
        """Returns parameters for a ConformerLayer."""
        return dict(
            ff_start=parameters_from_torch_layer(src.ffn1),
            self_attention=dict(
                attention=parameters_from_torch_layer(src.self_attn),
                norm=parameters_from_torch_layer(src.self_attn_layer_norm),
                dropout={},
                stochastic_depth={},
            ),
            lconv=parameters_from_torch_layer(src.conv_module),
            ff_end=parameters_from_torch_layer(src.ffn2),
            norm=parameters_from_torch_layer(src.final_layer_norm),
        )

    def _parameters_from_fairseq_conformer_ffn(src: fairseq_conformer.FeedForwardModule):
        """Returns parameters for a TransformerFeedForwardLayer."""
        return dict(
            linear1=parameters_from_torch_layer(src.w_1),
            linear2=parameters_from_torch_layer(src.w_2),
            norm=parameters_from_torch_layer(src.layer_norm),
            dropout1={},
            dropout2={},
            stochastic_depth={},
        )

    def _parameters_from_fairseq_conformer_conv(src: fairseq_conformer.ConvolutionModule):
        """Returns parameters for a LConvLayer."""
        linear1 = parameters_from_torch_layer(src.pointwise_conv1, dst_layer=Linear)
        linear1_channels = src.pointwise_conv1.out_channels // 2
        return dict(
            conv=parameters_from_torch_layer(src.depthwise_conv),
            conv_norm=parameters_from_torch_layer(src.batch_norm),
            linear1_0=dict(weight=linear1["weight"][:, :linear1_channels]),
            linear1_1=dict(weight=linear1["weight"][:, linear1_channels:]),
            linear1_norm=parameters_from_torch_layer(src.layer_norm),
            linear2=parameters_from_torch_layer(src.pointwise_conv2, dst_layer=Linear),
            dropout={},
        )

    def _parameters_from_fairseq_rel_pos_attention(
        src: fairseq_espnet_multihead_attention.RelPositionMultiHeadedAttention,
    ):
        """Returns parameters for a MultiheadAttentionXL."""
        qkv_weights = []
        qkv_biases = []
        for i in ("q", "k", "v"):
            src_proj = getattr(src, f"linear_{i}")
            qkv_weights.append(src_proj.weight.permute(1, 0).view(-1, src.h, src.d_k))
            qkv_biases.append(src_proj.bias.view(src.h, src.d_k))
        return dict(
            i_proj=VDict(
                qkv_proj=dict(
                    weight=torch.stack(qkv_weights, dim=0), bias=torch.stack(qkv_biases, dim=0)
                )
            ),
            o_proj=dict(
                weight=src.linear_out.weight.view(-1, src.h, src.d_k), bias=src.linear_out.bias
            ),
            r_proj=dict(weight=src.linear_pos.weight.permute(1, 0).view(-1, src.h, src.d_k)),
            u_bias=src.pos_bias_u,
            v_bias=src.pos_bias_v,
            dropout={},
            relative_pos_emb={},
            scale_query={},
            scale_key={},
        )


# pylint: disable-next=too-many-branches,too-many-statements
def parameters_from_torch_layer(
    src: torch.nn.Module, *, dst_layer: Optional[Union[BaseLayer, Type]] = None
) -> NestedTensor:
    """Extracts parameters from a torch module into a compatible format with AXLearn layers.

    TODO(markblee): Add a util to complete the src params, so that we don't have to manually patch
    in layers with no params like dropout.

    Args:
        src: torch module to extract from.
        dst_layer: (optional) the destination AXLearn layer.

    Returns:
        The AXLearn-compatible parameters.
    """
    dst = None
    if _FAIRSEQ_INSTALLED:
        if isinstance(src, fairseq_conformer.ConformerEncoderLayer):
            dst = _parameters_from_fairseq_conformer_layer(src)
        elif isinstance(src, fairseq_conformer.FeedForwardModule):
            dst = _parameters_from_fairseq_conformer_ffn(src)
        elif isinstance(src, fairseq_conformer.ConvolutionModule):
            dst = _parameters_from_fairseq_conformer_conv(src)
        elif isinstance(src, fairseq_espnet_multihead_attention.RelPositionMultiHeadedAttention):
            dst = _parameters_from_fairseq_rel_pos_attention(src)
    if dst is None:
        dst = torch_to_axlearn(src, dst_layer=dst_layer)
    return as_tensor(dst)
