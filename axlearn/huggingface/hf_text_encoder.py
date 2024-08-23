# Copyright © 2023 Apple Inc.

"""HuggingFace text encoders."""
import os
from enum import Enum, unique

import torch
from torch import nn
from transformers import AutoConfig, BertModel, BertPreTrainedModel, PretrainedConfig

L2_NORM = "l2_norm"


@unique
class EmbeddingPooler(Enum):
    # pylint: disable=invalid-name
    FirstNTokenPooling = 0
    AveragePooling = 1
    # pylint: enable=invalid-name


class TextEmbeddingEncoder(nn.Module):
    """A text embedding encoder that outputs a single embedding vector for each input sequence.

    It does following steps in forward():
        1. Getting encoded outputs from base model, e.g., BERT, RoBERTa.
        2. Taking representation of the first token as encoded embedding.
        3. Applying linear projection when specified.
        4. Applying normalization when specified. Currently only L2 normalization is supported.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        embedding_pooler: EmbeddingPooler = EmbeddingPooler.FirstNTokenPooling,
    ):
        super().__init__(config)  # pylint: disable=too-many-function-args
        if getattr(config, "projection_dim", 0) > 0:
            self.encode_proj = nn.Linear(config.hidden_size, config.projection_dim)
        self.embedding_pooler = embedding_pooler

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            input_ids: Input tokenized ids in shape of [batch_size, max_seq_len].
            attention_mask: Attention mask in shape of [batch_size, max_seq_len].

        Returns:
            Output embeddings in shape of [batch_size, output_dim] where output_dim==hidden_size
                when there is no linear projection, projection_dim otherwise.
        """
        base_model_outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        if self.embedding_pooler == EmbeddingPooler.FirstNTokenPooling:
            # Taking representation of the first token as output embedding.
            output = base_model_outputs["last_hidden_state"][:, 0, :]
        else:
            assert self.embedding_pooler == EmbeddingPooler.AveragePooling
            unpadded_output = base_model_outputs["last_hidden_state"] * torch.unsqueeze(
                attention_mask, -1
            )
            sum_unpadded_otuput = torch.sum(unpadded_output, 1)
            num_unpadded_tokens = torch.sum(attention_mask, 1)
            output = sum_unpadded_otuput / torch.unsqueeze(num_unpadded_tokens, -1)
        if hasattr(self, "encode_proj"):
            output = self.encode_proj(output)
        if getattr(self.config, "output_norm", None) == L2_NORM:
            output = nn.functional.normalize(output, p=2, dim=-1)
        return output


class MultiStreamTextEmbeddingModel(nn.Module):
    """A text embedding model consisted of multiple TextEmbeddingEncoder stream encoders."""

    def __init__(
        self,
        stream_encoders: dict[str, TextEmbeddingEncoder],
    ):
        super().__init__()
        self.stream_encoders = stream_encoders

    def save_pretrained(self, save_directory: str, *, safe_serialization: bool = True):
        """Save a model and its configuration file to a directory, so that it can be re-loaded
        using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        for encoder_name in self.stream_encoders:
            encoder_output_dir = os.path.join(save_directory, encoder_name)
            os.makedirs(encoder_output_dir, exist_ok=True)
            self.stream_encoders[encoder_name].save_pretrained(
                encoder_output_dir, safe_serialization=safe_serialization
            )

    def forward(
        self, input_batch: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward function.

        Args:
            input_batch: A dict where key is encoder name and value is a dict of per-encoder
                input_batch. Each per-encoder input_batch is a dict containing input_ids and
                attention_mask tensors. input_ids and attention_mask have shape of
                [batch_size, max_seq_len].

        Returns:
            encoder_outputs: A dict where key is encoder name and value is per-encoder output
                embeddings. Each output embedding tensor has shape of [batch_size, output_dim].
        """
        encoder_outputs = {}
        for encoder_name in self.stream_encoders:
            encoder_outputs[encoder_name] = self.stream_encoders[encoder_name](
                **input_batch[encoder_name]
            )
        return encoder_outputs

    @classmethod
    def from_pretrained(cls, output_dir: str, pooler: dict[str, str]):
        """Instantiate a pretrained pytorch model from a pre-trained model configuration."""
        stream_encoders = {}
        for encoder_name in os.listdir(output_dir):
            encoder_output_dir = os.path.join(output_dir, encoder_name)
            config = AutoConfig.from_pretrained(encoder_output_dir)
            if config.model_type == "bert":
                add_pooling_layer = getattr(config, "add_pooling_layer", True)
                stream_encoders[encoder_name] = BertTextEmbeddingEncoder.from_pretrained(
                    encoder_output_dir,
                    config=config,
                    embedding_pooler=pooler[encoder_name],
                    add_pooling_layer=add_pooling_layer,
                )
            else:
                raise ValueError("Unsupported model_type!")

        return cls(stream_encoders)


class BertTextEmbeddingEncoder(
    TextEmbeddingEncoder, BertPreTrainedModel
):  # pylint: disable=too-many-ancestors
    """Bert-based TextEmbeddingEncoder model."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        embedding_pooler: EmbeddingPooler = EmbeddingPooler.FirstNTokenPooling,
        **kwargs,
    ):
        super().__init__(config, embedding_pooler)
        # base_model_prefix of BertPreTrainedModel is "bert".
        self.bert = BertModel(config, **kwargs)
