# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/unilm:
# Copyright (c) 2021 Microsoft.
# Licensed under The MIT License.

"""A masked image modeling implementation.

References:
- Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., & Cao, Y.  (2022). Eva: Exploring the
limits of masked visual representation learning at scale. https://arxiv.org/abs/2211.07636
"""
from typing import Callable

from jax import numpy as jnp

from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Linear, cross_entropy
from axlearn.common.module import Module, NestedTensor, Tensor, child_context


# pylint: disable=no-self-use
class MaskedImageModel(BaseModel):
    """An implementation for masked image modeling."""

    @config_class
    class Config(BaseModel.Config):
        """Configures MaskedImageModel."""

        # The tokenizer takes the image as input and generates prediction labels in form of
        # tokens, features, etc.
        tokenizer: Required[InstantiableConfig] = REQUIRED
        # The encoder takes (preprocessed) image as input and generates embeddings.
        encoder: Required[InstantiableConfig] = REQUIRED
        # The head takes feature embeddings as input and generates predictions.
        head: InstantiableConfig = Linear.default_config().set(
            bias=True,
            param_partition_spec=("model", None),
        )
        loss_fn: Callable[..., Tensor] = cross_entropy

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("tokenizer", cfg.tokenizer)
        self._add_child("encoder", cfg.encoder)
        self._add_child(
            "head",
            cfg.head.clone(input_dim=self.encoder.endpoints_dims["embedding"]),
        )

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    # TODO(xianzhi): use plural from for input arguments.
    def generate_labels(self, image: Tensor) -> Tensor:
        """Generates labels for masked positions.

        Args:
            image: The input image. Shape: (batch, height_pixel, width_pixel, channels).

        Returns:
            A patch-wise target sequence in shape (batch, length) or (batch, length, dim).
            The output size is determined by the tokenizer implementation, examples:
            * Vocabulary-based tokenizer generates targets in shape (batch, length);
            * Feature-based tokenizer generates targets in shape (batch, length, dim), where
                `dim` is the `output_dim` of the tokenizer.
        """

        # Tokenizers used here need to be set in inference mode.
        # E.g. the tokenizer used for ViT-EVA is a pre-trained and frozen ViT-CLIP,
        # so ViT-EVA targets are 'image-text aligned vision features' (Fang et al., 2022)
        with child_context("tokenizer", module=self.tokenizer, is_training=False):
            targets, _ = self.tokenizer(image)
        return targets

    def compute_loss(self, logits: Tensor, labels: Tensor, is_masked: Tensor) -> Tensor:
        """Computes loss for the masked positions.

        Args:
            logits: The prediction logits in shape (batch, length, num_classes).
            labels: The prediction labels in shape (batch, length), with int values.
            is_masked: a boolean Tensor in shape (batch, length), representing the masked positions
                for the logits.

        Returns:
            The final loss.
        """
        # TODO(xianzhi): the reference code removes masked tokens before computing loss.
        # https://github.com/microsoft/unilm/blob/master/beit2/engine_for_pretraining.py
        loss, _ = self.config.loss_fn(
            logits.astype(jnp.float32),
            labels.astype(jnp.float32),
            live_targets=is_masked,
        )
        return loss

    def predict(self, image: Tensor, is_masked: Tensor) -> dict[str, Tensor]:
        """Generates model predictions.

        Args:
            image: The input image. Shape: (batch, height_pixel, width_pixel, channels).
            is_masked: a boolean Tensor in shape (batch, length), representing the masked positions
                for the patchifie input image.

        Returns:
            A dictionary representing the endpoints from the encoder.
        """
        endpoints = self.encoder(image, is_masked=is_masked)
        # We assume backbone outputs patch_features in shape (batch, num_patches, dim).
        x = endpoints["patch_features"]
        endpoints["logits"] = self.head(x)
        return endpoints

    # pylint: disable-next=arguments-differ
    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, NestedTensor]:
        """Runs forward pass.

        Args:
            input_batch: a dictionary that contains:
            * image: The input image. Shape: (batch, height_pixel, width_pixel, channels).
            * is_masked: a boolean Tensor in shape (batch, height_patch, width_patch),
                representing the masked positions for the patchifie input image.

        Returns:
            A tuple of (loss, outputs):
            * `loss` is a float scalar with total loss
            * `outptus` is a dictionary containing encoder endpoints, logits, labels and masks.
        """
        image, is_masked = input_batch["image"], input_batch.get("is_masked")
        if is_masked is not None:
            # [batch, length_in_patch]. Flatten the 2D mask.
            is_masked = jnp.reshape(is_masked, (is_masked.shape[0], -1))

        outputs = self.predict(image, is_masked)
        outputs["labels"] = self.generate_labels(image)
        outputs["is_masked"] = is_masked

        loss = self.compute_loss(
            logits=outputs["logits"],
            labels=outputs["labels"],
            is_masked=is_masked,
        )
        return loss, outputs
