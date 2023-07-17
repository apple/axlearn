# Copyright © 2023 Apple Inc.

"""Embedding layers."""
from typing import Optional

from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Dropout, Embedding
from axlearn.common.module import Module, Tensor, child_context


class TransformerTextEmbeddings(BaseLayer):
    """Textual embeddings from token id, position and token type embeddings."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures TransformerTextEmbeddings."""

        dim: Required[int] = REQUIRED  # Embedding dimension.
        vocab_size: Required[int] = REQUIRED  # Input embedding vocab size.
        token_emb: InstantiableConfig = Embedding.default_config()  # Input embedding lookup.
        type_emb: Optional[InstantiableConfig] = None  # Optional token type embedding lookup.
        pos_emb: Optional[InstantiableConfig] = None  # Optional positional embedding lookup.
        norm: Optional[InstantiableConfig] = None  # Optional layer norm for embeddings.
        dropout: InstantiableConfig = Dropout.default_config()  # Embedding dropout.
        # Optional soft output logits capping (only affect 'attend').
        # Enabled by setting a positive value.
        soft_cap_logits: Optional[float] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("token_emb", cfg.token_emb.set(dim=cfg.dim, num_embeddings=cfg.vocab_size))
        if cfg.type_emb is not None:
            self._add_child("type_emb", cfg.type_emb.set(dim=cfg.dim))
        if cfg.pos_emb is not None:
            self._add_child("pos_emb", cfg.pos_emb.set(dim=cfg.dim))
        if cfg.norm is not None:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.dim))
        self._add_child("dropout", cfg.dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        token_type_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes input embeddings with positional embeddings.

        If token_type_ids is provided, we also add input type embeddings.

        Args:
            inputs: arbitrary input tensor with general shape [batch_size, seq_len, ...] that
                will be fed directly to `self.token_emb`.
            token_type_ids: An optional int Tensor of shape [batch_size, seq_len].
            positions: An optional int Tensor of shape [batch_size, seq_len].
                If None, assumed to be jnp.arange(seq_len) for each sequence.

        Returns:
            A float Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        x = self.token_emb(inputs)
        if self.config.type_emb is not None:
            if token_type_ids is None:
                token_type_ids = jnp.zeros_like(inputs)
            x = x + self.type_emb(token_type_ids)
        if self.config.pos_emb is not None:
            if positions is None:
                positions = jnp.arange(x.shape[1])
            x += self.pos_emb(positions)
        if self.config.norm is not None:
            x = self.norm(x)
        x = self.dropout(x)
        return x

    def attend(self, x: Tensor) -> Tensor:
        """Computes logits with token embedding.

        Args:
            x: an int Tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            A float Tensor of shape [batch_size, seq_len, cfg.token_emb.num_embeddings]
        """
        cfg = self.config
        with child_context("token_emb", module=self.token_emb):
            logits = self.token_emb.attend(x)
            # Applies soft logits capping if set.
            if not cfg.soft_cap_logits or cfg.soft_cap_logits <= 0.0:
                return logits
            cap = jnp.array(cfg.soft_cap_logits, dtype=logits.dtype)
            return cap * jnp.tanh(logits / cap)
