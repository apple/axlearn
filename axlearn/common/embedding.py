# Copyright © 2023 Apple Inc.

"""Embedding layers."""

from dataclasses import dataclass
from typing import Optional

from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Dropout, Embedding
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.utils import Nested, validate_contains_paths


class BaseEmbedding(BaseLayer):
    """The base class of a embedding layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseEmbedding.

        Attributes:
            dim: Output emb dim.
        """

        dim: Required[int] = REQUIRED

    def forward(self, input_batch: Nested[Tensor]):
        """Computes embeddings.

        Args:
            input_batch: A nested Tensor where leaves have a leading batch dim.

        Returns:
            Outputs of this embeeding.
        """
        raise NotImplementedError(type(self))

    def attend(self, x: Tensor):
        """Computes logits with token embedding.

        Args:
            x: A Tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size].
        """
        raise NotImplementedError(type(self))


class TransformerTextEmbeddings(BaseEmbedding):
    """Textual embeddings from token id, position and token type embeddings."""

    @config_class
    class Config(BaseEmbedding.Config):
        """Configures TransformerTextEmbeddings."""

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

    def forward(self, input_batch: Nested[Tensor]) -> Tensor:
        """Computes input embeddings with positional embeddings.

        If token_type_ids is provided, we also add input type embeddings.

        Args:
            input_batch: A dict containing:
                * inputs: An input tensor with general shape [batch_size, seq_len, ...] that will be
                    fed directly to `self.token_emb`.
                * token_type_ids: An optional int Tensor of shape [batch_size, seq_len].
                * positions: An optional int Tensor of shape [batch_size, seq_len].
                    If None, assumed to be jnp.arange(seq_len) for each sequence.

        Returns:
            A float Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        validate_contains_paths(input_batch, paths=["inputs"])
        inputs = input_batch["inputs"]
        token_type_ids = input_batch.get("token_type_ids", None)
        positions = input_batch.get("positions", None)

        cfg: TransformerTextEmbeddings.Config = self.config

        x = self.token_emb(inputs)
        if cfg.type_emb is not None:
            if token_type_ids is None:
                token_type_ids = jnp.zeros_like(inputs)
            x = x + self.type_emb(token_type_ids)
        if cfg.pos_emb is not None:
            if positions is None:
                positions = jnp.arange(x.shape[1])
            x += self.pos_emb(positions)
        if cfg.norm is not None:
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


@dataclass
class ModalityVocabInfo:
    """A modality-specific vocab info.

    The range [placeholder_start, placeholder_end) denotes the global range of placeholder tokens
    for this modality in the raw input send to the `ModalityEmbedding.forward()` function.

    The range [vocab_start, vocab_end) denotes the range of tokens id after replacing
    the placeholder tokens with real tokens generated from the embedding in the
    `ModalityEmbedding.forward()` function. Usually this range contains `input_placeholder_range`,
    but we do not enforce this constraint. There can be cases that we use separate ranges to
    represent placeholders and real token ids.

    In a `ModalityEmbedding.forward()` pass, the placeholder tokens in
    [placeholder_start, placeholder_end) range will be used to gather embeddings (if available in
    the output). Then the placeholder tokens will be replaced with real token ids in range
    [vocab_start, vocab_end) (if available in the output).

    Attributes:
        modality_name: The name of this modality.
        placeholder_start: The start of the placeholders of this modality.
        placeholder_end: The end of the placeholders of this modality.
        vocab_start: The start of the vocab of this modality.
        vocab_ebd: The end of the vocab of this modality.
        generate_logits: If true, this modality will invoke `attend()` function to generate
            logits with a size of `vocab_size`. We set this to ease checking generation
            capabilities and infering output vocab size without instantitating the modality.
    """

    modality_name: str
    placeholder_start: int
    placeholder_end: int
    vocab_start: int
    vocab_end: int
    generate_logits: bool = False

    @property
    def vocab_size(self):
        return self.vocab_end - self.vocab_start


class ModalityEmbedding(BaseEmbedding):
    """The base class of a modality-specific embedding layer."""

    @config_class
    class Config(BaseEmbedding.Config):
        """Configures ModalityEmbedding.

        Attributes:
            modality_vocab_info: The ModalityVocabInfo.
            in_partition_spec: An optional input partition spec.
                This is useful when the per-modality batch size is smaller than the original batch
                partition, in which case it may be necessary to constrain over a subset of batch
                axes. If not None, partition specs should be specified for all modalities; set
                PartitionSpec.UNSPECIFIED explicitly if certain modalities should be unconstrained.
            out_partition_spec: An optional output partition spec.
                This is useful for constraining the output to the original batch partitioning.
        """

        modality_vocab_info: Required[ModalityVocabInfo] = REQUIRED
        in_partition_spec: Optional[PartitionSpec] = None
        out_partition_spec: Optional[PartitionSpec] = None

    @dataclass
    class Output:
        """Modality-specific outputs.

        All fields are optional.

        Attributes:
            ids: An int Tensor of shape [batch, max_num_modality, max_modality_len].
                Token IDs for the target modality. Th value should fall in
                [input_vocab_start, input_vocab_end).
            embeddings: A float Tensor of shape [batch, max_num_modality, max_modality_len, dim].
                Embeddings for the target modality.
            paddings: A 0/1 Tensor of same shape as `ids`. 1's represent padded positions.
            batch_idx: An int Tensor of shape [batch], with values in
                [0, global_batch_size) indicating batch positions for the output embeddings.
                By default, None means that embeddings are returned for all batch elements, i.e.,
                `batch == global_batch_size`.
                This is useful in situations where only a subset of input batch contains samples for
                the given modality.
        """

        ids: Optional[Tensor]
        embeddings: Optional[Tensor]
        paddings: Optional[Tensor]
        batch_idx: Optional[Tensor]

    def lookup_modality_embeddings(self, input_ids: Tensor, accum: Tensor) -> Tensor:
        """Looks up modality embeddings.

        During auto-regressive generation, backbone LLM may generate token in this modality
        but there is no input for this modalty. In this case, we still need to
        replace the tokens whose indices are out-of-bound for text vocab, but fall into
        the vocab index range of this modality.

        Args:
            input_ids: A [batch_size, max_length] index Tensor
            accum: A [batch_size, max_length, ...] embedding Tensor

        Returns:
            An updated accum in which if its corresponding input_ids fall in
                the current modality, accum[batch_size, max_length, ..] will
                be replaced by modality_embedding[input_ids[batch_size, max_length], ...].
        """
        raise NotImplementedError(type(self))
