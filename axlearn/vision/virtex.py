# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# kdexd/virtex:
# Copyright (c) 2020, Karan Desai.
# Licensed under the MIT license.

"""VirTex model implementation.

https://arxiv.org/abs/2006.06666
"""

from typing import Optional, Union

import jax
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.attention import TransformerAttentionLayer
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.decoder import Decoder
from axlearn.common.decoding import BeamSearchOutputs
from axlearn.common.layers import Linear
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, child_context
from axlearn.common.utils import NestedTensor, get_recursively, tree_paths, validate_contains_paths
from axlearn.common.vision_transformer import VisionTransformer as ViTModel
from axlearn.common.vision_transformer import named_model_configs as vit_named_model_configs
from axlearn.vision.resnet import ResNet

Tensor = jnp.ndarray


class ImageBackboneModelMixin(Module):
    """A mixin for models that have an image encoder backbone."""

    # The config of subclasses must contain the following fields (and types):
    #   - visual (InstantiableConfig): The config for the visual backbone.
    #   - visual_feature_layer_name (str): The layer in the visual backbone whose
    #     outputs will be used as visual features for the decoder.
    #   - visual_feature_size (int): The size of visual feature.

    @classmethod
    def resnet_backbone_config(cls, resnet_type: str) -> BaseLayer.Config:
        """Build a config with a ResNet visual backbone.

        Args:
            resnet_type: The name of the resnet model.
                One of 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.

        Returns:
            A BaseLayer.Config of this class.

        Raises:
            ValueError: If resnet_type is unsupported.
        """
        assert issubclass(cls, BaseLayer), f"{cls.__name__} must be a subclass of BaseLayer."

        fn_name = f"{resnet_type}_config"
        if not hasattr(ResNet, fn_name):
            raise ValueError(f"`ResNet` does not have a config for '{resnet_type}'")
        cfg = cls.default_config()
        cfg.visual = getattr(ResNet, fn_name)()
        # Extract features from 'stage3', the final conv layer,
        # in the resnet architecture. Dimensions of the output
        # features are 512 channels for resnet18 and resnet34 and
        # 2048 channels for resnet50, resnet101, and resnet152
        cfg.visual_feature_layer_name = "stage3/forward"
        cfg.visual_feature_size = 512 if resnet_type in ["resnet18", "resnet34"] else 2048

        return cfg

    @classmethod
    def vit_backbone_config(cls, vit_type: Union[str, ViTModel.Config]) -> BaseLayer.Config:
        """Build a config with a ViT visual backbone.

        Args:
            vit_type: The name of the ViT model.
                See vision_transformer.py for default names.

        Returns:
            A BaseLayer.Config of this class.
        """
        assert issubclass(cls, BaseLayer), f"{cls.__name__} must be a subclass of BaseLayer."

        cfg = cls.default_config()
        cfg.visual = (
            vit_named_model_configs()[vit_type] if isinstance(vit_type, str) else vit_type.clone()
        )
        cfg.visual_feature_layer_name = "pooled_features"
        cfg.visual_feature_size = cfg.visual.output_dim

        return cfg

    def embed_image(self, image: NestedTensor) -> Tensor:
        """Embed the image into features.

        Args:
            image: The input image. Shape: [batch, height, width, channels].

        Returns:
            a float 3D Tensor of shape [batch, -1, dim]
        """
        pass  # pylint: disable=unnecessary-pass

    def embed_image_batch(self, input_batch: NestedTensor) -> Tensor:
        """Wrapper for embed_image that takes an input batch."""
        return self.embed_image(input_batch["image"])

    def _get_visual_features(self, visual_outputs: NestedTensor) -> Tensor:
        """Gets the visual features.

        The visual features will be extracted from a NestedTensor based on a path specified in
        `cfg.visual_feature_layer_name`.

        The NestedTensor can be either `visual_outputs` or self.get_module_outputs()["visual"].

        Args:
            visual_outputs: The outputs of the visual layer.

        Returns:
            visual_features: The features from the visual backbone.
                Shape: (batch, height, width, channels)

        Raises:
            ValueError: If visual_feature_layer_name cannot be found.
        """
        cfg = self.config
        try:
            return get_recursively(visual_outputs, cfg.visual_feature_layer_name)
        except KeyError:
            pass

        try:
            return get_recursively(
                self.get_module_outputs(),
                f"visual/{cfg.visual_feature_layer_name}",
            )
        except KeyError:
            pass

        def _paths(x):
            return jax.tree_util.tree_leaves(tree_paths(x))

        raise ValueError(
            f"Cannot find visual features at {cfg.visual_feature_layer_name}. "
            f"visual_outputs={_paths(visual_outputs)}, "
            f"module_outputs={_paths(self.get_module_outputs().get('visual'))}"
        )


class VirTexModel(ImageBackboneModelMixin, BaseLayer):
    """The generic VirTex model."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures VirTexModel."""

        # The visual backbone.
        visual: Required[InstantiableConfig] = REQUIRED
        # The layer in the visual backbone whose outputs will be used
        # as visual features for the decoder.
        visual_feature_layer_name: Required[str] = REQUIRED
        # The size of visual feature
        visual_feature_size: Required[int] = REQUIRED
        # The text decoder.
        textual: Decoder.Config = Decoder.default_config()

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        cfg.textual.transformer.layer.cross_attention = TransformerAttentionLayer.default_config()
        # TODO: Determine what the default should be.
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            init_by_param_name={
                param_init.PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                    fan="fan_avg", distribution="uniform"
                )
            }
        )
        # Normal distribution initialization as done in VirTex.
        # https://github.com/kdexd/virtex/blob/ae67b23f86ab10934f43df239666592acbd74631/virtex/modules/textual_heads.py#L98
        cfg.textual.param_init = param_init.DefaultInitializer.default_config().set(
            init_by_param_name={
                param_init.PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                    # Equivalent to normal_(std_dev=0.02).
                    distribution="normal",
                    scale=0.02,
                )
            }
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("visual", cfg.visual)
        self._add_child("textual", cfg.textual)
        self._add_child(
            "visual_projection",
            Linear.default_config().set(
                input_dim=cfg.visual_feature_size, output_dim=cfg.textual.dim
            ),
        )

    def forward(
        self,
        input_batch: dict[str, Tensor],
        return_aux: bool = False,
    ) -> Tensor:
        """
        Args:
            input_batch: a dict with the following entries:
                image: The input image. Shape: [batch, height, width, channels].
                caption_tokens: The tokenized caption for the image.
                    Note this includes both BOS and EOS tokens. Thus the sequence length
                    should be +1 of the maximum sequence length.
                    Shape: [batch, max_sequence_length]. Values should be in the range
                    [0, vocab_size].
            return_aux: Whether to return auxiliary outputs, which includes
                visual backbone features and text decoder logits (and hidden state
                if the decoder is a transformer).

        Returns:
            (loss, Dict): The loss and dictionary of auxiliary outputs (if `return_aux=True`).
                If `return_aux=False`, an empty dictionary will be returned.
        """
        image = input_batch["image"]
        caption_tokens: Tensor = input_batch["caption_tokens"]

        projected_visual_features = self.embed_image(image)

        # Decode caption.
        decoder_ids, decoder_labels = caption_tokens[:, :-1], caption_tokens[:, 1:]
        predictions: dict[str, Tensor] = self.textual(
            input_batch=dict(input_ids=decoder_ids),
            cross_attention_data=projected_visual_features,
        )
        metrics = self._metrics(predictions["logits"], decoder_labels)

        aux_outputs = dict(visual_features=projected_visual_features, **predictions)
        return metrics["loss"], aux_outputs if return_aux else {}

    def caption(
        self,
        *,
        image: Tensor,
        prefix: Tensor,
        max_sequence_length: int,
        num_decodes: int,
    ) -> BeamSearchOutputs:
        """Autoregressively generate a caption for the image via beam search.

        Args:
            image: The input image. Shape: [batch, height, width, channels].
            prefix: The prefix to use for prompting. a Tensor of shape [batch, max_prefix_length].
                This can be a vector of [BOS] tokens.
                Currently, all prefixes must have max_prefix_length=1.
            max_sequence_length: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sqeuences to return.

        Returns:
            The beam search outputs.
        """
        projected_visual_features = self.embed_image(image)
        with child_context("beam_search_decode", module=self.textual):
            output: BeamSearchOutputs = self.textual.beam_search_decode(
                input_batch=dict(prefix=prefix),
                max_sequence_length=max_sequence_length,
                num_decodes=num_decodes,
                cross_attention_data=projected_visual_features,
            )
        return output

    def beam_search_decode(
        self,
        input_batch: NestedTensor,
        *,
        max_sequence_length: int,
        num_decodes: int,
    ) -> BeamSearchOutputs:
        """Autoregressively generate a caption for the image via beam search.

        Args:
            input_batch: The batched input. Keys should include:
                * image: The input image. Shape: [batch, height, width, channels].
                * prefix: The prefix to use for prompting.
                    a Tensor of shape [batch, max_prefix_length].
                    This can be a vector of [BOS] tokens.
            max_sequence_length: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sqeuences to return.

        Returns:
            The beam search outputs.
        """
        validate_contains_paths(input_batch, paths=["image", "prefix"])
        return self.caption(
            image=input_batch["image"],
            prefix=input_batch["prefix"],
            max_sequence_length=max_sequence_length,
            num_decodes=num_decodes,
        )

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed the image into features.

        Args:
            image: The input image. Shape: [batch, height, width, channels].

        Returns:
            a float Tensor of shape [batch, -1, text_dim]
        """
        visual_features = self._get_visual_features(self.visual(image))

        # If visual features are 2D (i.e. shape: [batch, height, width, channels]),
        # collapse them into a single dimension (i.e. shape: [batch, height*width, channels]).
        batch_size, channels = visual_features.shape[0], visual_features.shape[-1]
        visual_features = visual_features.reshape((batch_size, -1, channels))

        # Project visual features to same shape as textual features.
        # Shape: (batch, -1, channels) -> (batch, -1, textual.get_textual_feature_size())
        projected_visual_features = self.visual_projection(visual_features)
        return projected_visual_features

    def _metrics(self, logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
        cfg = self.config

        live_targets = (targets != cfg.textual.pad_token_id).astype(jnp.float32)
        num_targets = live_targets.sum()
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets).sum() / num_targets
        self.add_summary("accuracy", WeightedScalar(accuracy, num_targets))
        if logits.dtype in (jnp.bfloat16, jnp.float16):
            # Cast to fp32 for softmax loss.
            # TODO(tom_gunter): Implement a more stable cross entropy loss like:
            # <https://github.com/google-research/t5x/blob/90d74fa703075d8b9808ae572602bc48759f8bcc/t5x/losses.py#L26>
            logits = logits.astype(jnp.float32)
        # [batch, source_length].
        per_token_loss = (
            -(jax.nn.log_softmax(logits) * jax.nn.one_hot(targets, cfg.textual.vocab_size)).sum(
                axis=-1
            )
            * live_targets
        )
        # We cannot use max(1, num_targets) here because jax cannot use
        # an abstract tracer value when a concrete value is expected.
        # See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
        loss = per_token_loss.sum() / num_targets
        self.add_summary("loss", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))
        return dict(
            loss=loss,
            per_token_loss=per_token_loss,
            live_targets=live_targets,
            num_targets=num_targets,
        )
