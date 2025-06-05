# Copyright © 2023 Apple Inc.

"""CyCLIP implementation.

Ref: https://github.com/goel-shashank/CyCLIP/blob/main/src/train.py
"""
# pylint: disable=duplicate-code

from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import InstantiableConfig, config_class
from axlearn.common.loss import contrastive_logits, mean_squared_error
from axlearn.common.module import Module
from axlearn.common.multi_stream_model import FusionNetwork
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.vision.clip import CLIPFusionNetwork, CLIPModel


class CyCLIPFusionNetwork(FusionNetwork):
    """CyCLIP fusion network. See also CLIPModel."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures CyCLIPFusionNetwork.

        Ref: https://arxiv.org/pdf/2205.14459.pdf pp.4 Sect. 2.3.3
        """

        cross_modal_weight: Optional[float] = 0.0
        in_modal_weight: Optional[float] = 0.0
        clip_fusion_network: CLIPFusionNetwork.Config = CLIPFusionNetwork.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("clip_fusion_network", cfg.clip_fusion_network)

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function of cyCLIP.

        Args:
            input_batch: A dictionary containing:
                *"visual_encoder":
                    *"output_features": A Tensor with shape [batch_size, num_images, dim]
                *"textual_encoder":
                    *"output_features": A Tensor with shape [batch_size, num_sentences, dim]

        Returns:
            loss: A Tensor representing the loss
            A dictionary containing:
                *"similarity": A Tensor representing the similarity between
                    the text and image.
        """
        cfg = self.config
        clip_loss, metrics_dict = self.clip_fusion_network(input_batch)

        x = input_batch["visual_encoder"]["output_features"]
        y = input_batch["textual_encoder"]["output_features"]

        batch_size = x.shape[0]
        num_images = x.shape[1]
        num_sentences = y.shape[1]

        x = x.reshape(batch_size * num_images, *x.shape[2:])
        y = y.reshape(batch_size * num_sentences, *y.shape[2:])

        similarity = metrics_dict["similarity"]
        in_modal_loss = in_modal_reg_from_similarity(similarity)
        cross_modal_loss = cross_modal_reg(x, y)
        loss = (
            clip_loss
            + cfg.cross_modal_weight * cross_modal_loss
            + cfg.in_modal_weight * in_modal_loss
        )
        self.add_summary("loss_in_modal", in_modal_loss)
        self.add_summary("loss_cross_modal", cross_modal_loss)
        self.add_summary("loss_info_nce", clip_loss)
        self.add_summary("loss", loss)
        return loss, metrics_dict


def in_modal_reg_from_similarity(similarity: Tensor) -> Tensor:
    """Compute in-modal consistency regularization.

    in-modal consistency regularizer reduces the gap in the similarity scores between
    the embeddings of all combinations of image pairs and their corresponding text pairs in a batch

    Args:
        similarity: A float Tensor of shape [batch_size, batch_size]

    Returns:
        A float Tensor represents the cross-modal loss.
    """
    assert (
        len(similarity.shape) == 2 and similarity.shape[0] == similarity.shape[1]
    ), f"similarity must be a 2-D square matrix, but got similarity.shape={similarity.shape}"
    similarity_t = similarity.T
    return mean_squared_error(similarity, similarity_t).mean


def cross_modal_reg(x: Tensor, y: Tensor) -> Tensor:
    """Compute cross-modal consistency regularization.

    Cross-modal consistency regularizer reduces the gap in the similarity scores between
    the embeddings of all the mismatched image-text pairs in a batch, two at a time

    Args:
        x: A float Tensor of embedding with shape [batch_size, dimension] from modality 1
        y: A float Tensor of embedding with shape [batch_size, dimension] from modality 2

    Returns:
        A float Tensor represents the cross-modal loss.
    """
    assert (
        x.shape == y.shape
    ), f"x, y must have same shape but got x.shape={x.shape}, y.shape={y.shape}"
    sim1 = contrastive_logits(x, x)
    sim2 = contrastive_logits(y, y)
    return mean_squared_error(sim1, sim2).mean


def set_cyclip_fusion_network_config(
    *,
    cross_modal_weight: float,
    in_modal_weight: float,
):
    cyclip_fusion_network = CyCLIPFusionNetwork.default_config().set(
        cross_modal_weight=cross_modal_weight,
        in_modal_weight=in_modal_weight,
    )
    return cyclip_fusion_network


def set_cyclip_config(
    *,
    cross_modal_weight: float,
    in_modal_weight: float,
    text_encoder_cfg: InstantiableConfig,
    vision_encoder_cfg: InstantiableConfig,
):
    cyclip_fusion_network_cfg = set_cyclip_fusion_network_config(
        cross_modal_weight=cross_modal_weight,
        in_modal_weight=in_modal_weight,
    )
    clip_stream_encoder = {
        "visual_encoder": vision_encoder_cfg,
        "textual_encoder": text_encoder_cfg,
    }
    clip_fusion_network = {"fusion_network": cyclip_fusion_network_cfg}
    cyclip_model = CLIPModel.default_config().set(
        stream_encoder=clip_stream_encoder, fusion_network=clip_fusion_network
    )

    return cyclip_model
