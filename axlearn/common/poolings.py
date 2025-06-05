# Copyright © 2023 Apple Inc.

"""Layers for pooling operations.

On `paddings`:
`Paddings` is a Tensor with shape: (batch, seq_len).
    It represents the padded token masks.
    0 (False) means valid token and 1 (True) means padded token.
    paddings only take 0 / 1 or False / True as values.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention import (
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    scaled_hidden_dim,
)
from axlearn.common.attention_bias import NEG_INF, make_segment_mask
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Linear
from axlearn.common.module import Module
from axlearn.common.utils import Tensor, safe_not


class BasePoolingLayer(BaseLayer):
    """The base class of a pooling layer."""

    @config_class
    class Config(BaseLayer.Config):
        # Input and output embedding dimensions.
        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED

        # Number of outputs from pooling functions.
        num_outputs: int = 1

    def forward(self, tokens: Tensor, paddings: Tensor = None) -> Tensor:
        """
        Args:
            tokens: The image tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token masks. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, num_outputs, output_dim).
        """
        raise NotImplementedError(type(self))


class AttentionPooling(BasePoolingLayer):
    """Attention-based pooling.

    Reference:
    https://arxiv.org/pdf/2205.01917.pdf (Section 3.2)
    """

    @config_class
    class Config(BasePoolingLayer.Config):
        # The cross attention layer config.
        cross_attention: InstantiableConfig = TransformerAttentionLayer.default_config()
        feed_forward: InstantiableConfig = TransformerFeedForwardLayer.default_config()

    @classmethod
    def default_config(cls) -> Config:
        cfg: AttentionPooling.Config = super().default_config()
        # pylint: disable=no-member  # pytype: disable=attribute-error
        cfg.cross_attention.attention.num_heads = 1
        cfg.feed_forward.hidden_dim = scaled_hidden_dim(scale=4)
        # pylint: enable=no-member  # pytype: enable=attribute-error
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        cfg.cross_attention.set(source_dim=cfg.input_dim, target_dim=cfg.output_dim)
        self._add_child("cross_attention", cfg.cross_attention)
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.output_dim))

    def forward(self, tokens: Tensor, paddings: Tensor = None) -> Tensor:
        """
        Args:
            tokens: The input tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token masks. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, num_outputs, output_dim).
        """
        cfg = self.config
        targets: Tensor = jnp.tile(
            jnp.expand_dims(self.parameters["query_weight"], 0), (tokens.shape[0], 1, 1)
        )
        self.vlog(3, "targets shape: %s", targets.shape)

        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool)

        source_masks = safe_not(paddings)
        target_masks = jnp.ones((tokens.shape[0], cfg.num_outputs), dtype=jnp.bool)
        masks = make_segment_mask(source_segments=source_masks, target_segments=target_masks)

        targets = self.cross_attention(
            target=targets, source=tokens, attention_logit_biases=masks
        ).data
        targets = self.feed_forward(targets)
        return targets

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        # The "weight" suffix is needed to match the strings for initialization
        # in DefaultInitializer.
        return dict(
            query_weight=ParameterSpec(
                shape=(cfg.num_outputs, cfg.output_dim),
                mesh_axes=None,
            )
        )


class AveragePooling(BasePoolingLayer):
    """Average pooling layer."""

    @config_class
    class Config(BasePoolingLayer.Config):
        # eps is added to avoid divided by zero.
        eps: float = 1e-8

    def forward(self, tokens: Tensor, paddings: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            tokens: The image tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token indicators. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, 1, output_dim).

        Raises:
            ValueError: If cfg.num_outputs > 1 or cfg.input_dim != cfg.output_dim.
        """
        cfg = self.config

        if cfg.num_outputs > 1:
            raise ValueError("AveragePooling doesn't support more than 1 query.")

        if cfg.input_dim != cfg.output_dim:
            raise ValueError("AveragePooling requrires input_dim == output_dim.")

        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool)
        input_masks = safe_not(paddings)
        input_masks = jnp.expand_dims(input_masks, axis=-1)
        embeddings_sum = jnp.sum(tokens * input_masks, axis=1, keepdims=True)
        masks_sum = input_masks.sum(axis=1, keepdims=True) + self.config.eps
        pooled_embeddings = embeddings_sum / masks_sum
        return pooled_embeddings


class MaxPooling(BasePoolingLayer):
    """Max pooling layer."""

    def forward(self, tokens: Tensor, paddings: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            tokens: The image tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token indicators. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, 1, output_dim).

        Raises:
            ValueError: If cfg.num_outputs > 1 or cfg.input_dim != cfg.output_dim.
        """
        cfg = self.config

        if cfg.num_outputs > 1:
            raise ValueError("AveragePooling doesn't support more than 1 query.")

        if cfg.input_dim != cfg.output_dim:
            raise ValueError("AveragePooling requrires input_dim == output_dim.")

        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        input_masks = jnp.expand_dims(paddings, axis=-1) * NEG_INF
        pooled_embeddings = jnp.max(tokens + input_masks, axis=1, keepdims=True)
        return pooled_embeddings


class FirstNTokenPooling(BasePoolingLayer):
    """Take the first N tokens as the pooler output."""

    def forward(self, tokens: Tensor, paddings: Optional[Tensor] = None) -> Tensor:
        """Computes pooling from first N tokens.

        If the number of not padded tokens is smaller than the num_outputs,
        FirstNTokenPooling returns a tensor with shape (batch, num_outputs, dim).
        But, the padded item will be filled with zeros.

        Example:
            tokens with shape (2, 2, 3)= [[[0.1, 0.2 ,0.3],
                                           [0.2, 0.3, 0.4]],
                                          [[0.4, 0.5, 0.6],
                                           [0.7, 0.8, 0.9]]]
             paddings with shape (2, 2) = [[0, 1],
                                           [0, 0]]
             num_outputs = 2
             The output = [[[0.1, 0.2, 0.3], <-- The first token.
                            [0,   0,   0]], <-- 2nd token (padded).
                           [[0.7, 0.8, 0.9], <-- The first token.
                            [0.4, 0.5, 0.6]]] <-- 2nd token (not padded).

        Args:
            tokens: The image tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token indicators. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, num_outputs, dim).
        """
        n = self.config.num_outputs
        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        return tokens[:, :n, :] * safe_not(paddings)[:, :n, None]


class LastNTokenPooling(BasePoolingLayer):
    """Take the last N tokens as the pooler output."""

    def forward(self, tokens: Tensor, paddings: Optional[Tensor] = None) -> Tensor:
        """Computes pooling from last N tokens.

        If num_outputs < input_masks valid tokens per line.
        LastNTokenPooling returns a tensor with shape (batch, num_outputs, dim).
        But, the invalid item will be filled with zeros.

        Example:
            tokens with shape (2, 2, 3)= [[[0.1, 0.2 ,0.3],
                                           [0.2, 0.3, 0.4]],
                                          [[0.4, 0.5, 0.6],
                                           [0.7, 0.8, 0.9]]]
             paddings with shape (2, 2) = [[0, 1],
                                           [0, 0]]
             num_outputs = 2
             The output = [[[0.1, 0.2, 0.3], <-- The last token.
                            [0,   0,   0]], <-- 2nd to the last token (invalid).
                           [[0.7, 0.8, 0.9], <-- The last token.
                            [0.4, 0.5, 0.6]]] <-- 2nd to the last token (valid).

        Args:
            tokens: The image tokens. Shape: (batch, seq_len, source_dim).
            paddings: The padded token masks. Shape: (batch, seq_len).
                See ``On paddings`` in the file comments.

        Returns:
            A float Tensor of shape (batch, num_outputs, dim)

        TODO(bwzhang@): This only support one segment. Add multi segments support later.
        """

        cfg = self.config
        num_outputs = cfg.num_outputs
        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool)
        input_masks = safe_not(paddings)
        # Determine the last N tokens via input_masks.
        # The idea is to obtain the last N positions per line with input_masks==1.
        # Concretely, we count the position of the input_masks==1 per line
        # with the flipped input_masks.
        input_masks = input_masks[:, ::-1]
        input_masks_cumsum = input_masks.cumsum(axis=1)[:, ::-1]
        dispatch = jax.nn.one_hot(input_masks_cumsum - 1, num_outputs, dtype=tokens.dtype)
        chosen_tokens = jnp.einsum("bsd,bso->bod", tokens, dispatch)

        return chosen_tokens


class PoolingWithProjection(BasePoolingLayer):
    """Composite pooler containing a regular pooler followed by a projection."""

    @config_class
    class Config(BasePoolingLayer.Config):
        # Arbitrary pooling layer. Note that if `pooler.output_dim` is not set, we will
        # set it to `pooler.input_dim`.
        pooler: Required[BasePoolingLayer.Config] = REQUIRED
        # Optional projection layer that maps to a different output dimensionality.
        proj: InstantiableConfig = Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        cfg.pooler.output_dim = cfg.pooler.output_dim or cfg.input_dim
        self._add_child(
            "pooler",
            cfg.pooler.set(
                input_dim=cfg.input_dim,
            ),
        )
        self._add_child(
            "proj",
            cfg.proj.set(
                input_dim=cfg.pooler.output_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, tokens: Tensor, paddings: Tensor = None) -> Tensor:
        """See BasePoolingLayer.forward docstring for details."""
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens.ndim=3, but got ndim={tokens.ndim} for "
                f"tokens with shape={tokens.shape}."
            )
        pooled_embeddings = self.pooler(tokens, paddings=paddings)
        pooled_embeddings = self.proj(pooled_embeddings)
        return pooled_embeddings
