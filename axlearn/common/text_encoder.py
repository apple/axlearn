# Copyright © 2023 Apple Inc.

"""Text encoder module."""
from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.encoder import Encoder
from axlearn.common.module import Module
from axlearn.common.poolings import BasePoolingLayer, FirstNTokenPooling
from axlearn.common.utils import NestedTensor, Tensor

ENCODED_HIDDEN_STATES = "encoded_hidden_states"
TEXT_EMBEDDINGS = "text_embeddings"


class TextEmbeddingEncoder(BaseLayer):
    """A text embedding encoder that could output a single embedding vector for each input sequence.

    It consists of an encoder and a pooler:
        Encoder handles the following procedure:
            1. Mapping tokens into embedding.
            2. Adding positional embeddings.
            3. Applying Transformer on the embeddings.

        The pooler pools the output of encoder into an embedding. Pooler's input_dim is assumed to
        be the same as output_dim.
    """

    @config_class
    class Config(BaseLayer.Config):
        # Output dimension of TextEmbeddingEncoder.
        output_dim: Required[int] = REQUIRED
        # Hidden dimension of encoder. If set to None, the hidden_dim = output_dim.
        hidden_dim: Optional[int] = None
        # Value of input id corresponding to padding.
        pad_token_id: Required[int] = REQUIRED
        encoder: Encoder.Config = Encoder.default_config()
        pooler: BasePoolingLayer.Config = FirstNTokenPooling.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim or cfg.output_dim
        self._add_child("encoder", cfg.encoder.set(dim=hidden_dim, pad_token_id=cfg.pad_token_id))
        self._add_child("pooler", cfg.pooler.set(input_dim=hidden_dim, output_dim=cfg.output_dim))

    def forward(self, input_ids: Tensor) -> NestedTensor:
        """Computes text embedding.

        Text -> Text Token Emb. -> Transformer -> Pooler -> Output.

        Args:
            input_ids: An int Tensor with shape: [batch, length] representing token ids. Paddings
                are represented by cfg.pad_token_id.

        Returns:
            A dictionary containing:
                * ENCODED_HIDDEN_STATES: with shape (batch, length, hidden_dim);
                * TEXT_EMBEDDINGS: with shape (batch, num_outputs, output_dim).
        """
        cfg = self.config
        # Text token emb. + Transformer.
        x = self.encoder(input_ids)
        # Pooler.
        paddings = input_ids == cfg.pad_token_id
        pooled_output = self.pooler(x, paddings=paddings)
        return {ENCODED_HIDDEN_STATES: x, TEXT_EMBEDDINGS: pooled_output}
