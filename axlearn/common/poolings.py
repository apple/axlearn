# Copyright © 2023 Apple Inc.

"""Layers for pooling operations.

On `paddings`:
`Paddings` is a Tensor with shape: (batch, seq_len).
    It represents the padded token masks.
    0 (False) means valid token and 1 (True) means padded token.
    paddings only take 0 / 1 or False / True as values.
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention import (
    NEG_INF,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    make_segment_mask,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import LayerNorm, Linear, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


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
        cfg.cross_attention.attention.num_heads = 1  # pylint: disable=no-member
        cfg.feed_forward.hidden_dim = scaled_hidden_dim(scale=4)  # pylint: disable=no-member
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
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)

        source_masks = 1 - paddings
        target_masks = jnp.ones((tokens.shape[0], cfg.num_outputs), dtype=tokens.dtype)
        masks = make_segment_mask(source_segments=source_masks, target_segments=target_masks)

        targets = self.cross_attention(
            target=targets, source=tokens, attention_logit_biases=masks
        ).data
        targets = self.feed_forward(targets)
        return targets

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
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
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        input_masks = 1 - paddings
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
        return tokens[:, :n, :] * (1 - paddings[:, :n, None])


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
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        input_masks = 1 - paddings
        # Determine the last N tokens via input_masks.
        # The idea is to obtain the last N positions per line with input_masks==1.
        # Concretely, we count the position of the input_masks==1 per line
        # with the flipped input_masks.
        input_masks = input_masks[:, ::-1]
        input_masks_cumsum = input_masks.cumsum(axis=1)[:, ::-1]
        dispatch = jax.nn.one_hot(input_masks_cumsum - 1, num_outputs)
        chosen_tokens = jnp.einsum("bsd,bso->bod", tokens, dispatch)

        return chosen_tokens


class SpladeMapping(BaseLayer):
    """SpladeMapping is a simplified version of BertLM head.

    The SpladeMapping maps the embedding from input_dim to intermediate_dim via input_mapping.
    The embedding is further mapped into output_dim via vocab_mapping.
    """

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED
        intermediate_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        input_mapping: Linear.Config = Linear.default_config()
        activation_fn: str = "nn.gelu"
        input_norm: LayerNorm.Config = LayerNorm.default_config()
        vocab_mapping: InstantiableConfig = Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "input_mapping",
            cfg.input_mapping.set(input_dim=cfg.input_dim, output_dim=cfg.intermediate_dim),
        )
        self._add_child(
            "input_norm",
            cfg.input_norm.set(input_dim=cfg.intermediate_dim),
        )
        self._add_child(
            "vocab_mapping",
            cfg.vocab_mapping.set(input_dim=cfg.intermediate_dim, output_dim=cfg.output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.input_mapping(inputs)
        x = get_activation_fn(self.config.activation_fn)(x)
        x = self.input_norm(x)
        output_emb = self.vocab_mapping(x)
        return output_emb


class SpladePooling(BasePoolingLayer):
    """Splade pooling layer."""

    @config_class
    class Config(BasePoolingLayer.Config):
        # Splade activation function. ReLU by default.
        splade_activation_fn: str = "nn.relu"
        splade_mode: str = "max"
        vocab_mapping: InstantiableConfig = SpladeMapping.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "vocab_mapping",
            cfg.vocab_mapping.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(  # pylint:disable=arguments-renamed
        self, tokens: Tensor, paddings: Tensor = None
    ) -> Tensor:
        """Calculate the Splade Pooler.

        Args:
            tokens: A Tensor of shape [batch_size, seq_len, hidden_dim].
            paddings: A Tensor of shape [batch_size, seq_len].

        Returns:
            A Tensor of shape [batch_size, num_outputs, vocab_size] representing Splade features,
             where num_outputs is determined by the pooling mode.
             Currently cfg.splade_mode supports max and sum. For both, num_outputs = 1.

        Raises:
            ValueError: If cfg.splade_mode is not supported.
            NotImplementedError: If cfg.num_outputs > 1.
        """
        cfg = self.config
        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        # paddings shape is expanded to [batch_size, seq_len, 1].
        paddings = jnp.expand_dims(paddings, -1)
        if cfg.splade_mode not in ["max", "sum"]:
            raise ValueError(f"({cfg.splade_mode}) is not supported in Splade pooling.")
        if cfg.num_outputs != 1:
            raise NotImplementedError(
                f"SPLADE pooling currently doesn't support num_outputs = ({cfg.num_outputs})."
            )
        # The output of MLM should have shape [batch_size, seq_len, vocab_size]
        # BertLMHead uses predict instead of forward.
        x = self.vocab_mapping(inputs=tokens)
        # Splade = max( log(1 + relu(x)), dim=1)
        # (yinfeiy@) reports doing max pooling first will reduce memory in pytorch.
        # TODO(bwzhang@) Check the pooling order within AXLearn.
        splade_output = jnp.log1p(get_activation_fn(cfg.splade_activation_fn)(x))
        if cfg.splade_mode == "max":
            splade_output += paddings * NEG_INF  # set padded values to -inf
            splade_output = jnp.max(splade_output, axis=1, keepdims=True)
        elif cfg.splade_mode == "sum":
            splade_output *= 1 - paddings  # set padded values to 0
            splade_output = jnp.sum(splade_output, axis=1, keepdims=True)
        return splade_output


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
