# Copyright © 2023 Apple Inc.

"""Feature tokenizers."""
from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.layers import BaseNormalizationLayer, Linear
from axlearn.common.module import Module
from axlearn.common.utils import Tensor
from axlearn.common.vision_transformer import VisionTransformer


class CLIPFeatureTokenizer(BaseLayer):
    """A feature tokenizer using CLIP image encoder."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures CLIPFeatureTokenizer."""

        # The dim of the output embeddings.
        output_dim: Required[int] = REQUIRED
        # The hidden dim of `image_encoder`. If None, uses output_dim.
        hidden_dim: Optional[int] = None
        image_encoder: VisionTransformer.Config = VisionTransformer.default_config()
        output_proj: Optional[Linear.Config] = None
        # Coca uses LayerNorm instead.
        output_norm: Optional[BaseNormalizationLayer.Config] = None
        # A boolean for enabling the norm in image encoding.
        apply_output_norm: bool = False

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim or cfg.output_dim
        self._add_child("image_encoder", cfg.image_encoder.set(output_dim=hidden_dim))
        if cfg.output_proj:
            self._add_child(
                "output_proj", cfg.output_proj.set(input_dim=hidden_dim, output_dim=cfg.output_dim)
            )
        if cfg.output_norm:
            if "input_dim" in cfg.output_norm:
                cfg.output_norm.set(input_dim=cfg.output_dim)
            self._add_child("output_norm", cfg.output_norm)

    # pylint:disable-next=arguments-renamed
    def forward(self, inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """The forward function for the CLIPTokenizer.

        Args:
            inputs: A Tensor with shape [batch_size, height, width, channel].

        Returns:
            A tuple contains (targets, aux):
            - targets: the prediction targets in shape [batch_size, length, dim];
            - aux: the output dictionary from the image encoder.
        """
        cfg = self.config
        output_dict = self.image_encoder(inputs)
        # [batch_size, length, dim].
        x = output_dict["patch_features"]
        if "output_proj" in self.children:
            x = self.output_proj(x)
        if cfg.apply_output_norm and "output_norm" in self.children:
            x = self.output_norm(x)

        return x, output_dict
