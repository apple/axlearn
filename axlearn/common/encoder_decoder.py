# Copyright © 2023 Apple Inc.

"""Encoder-Decoder model."""

from typing import Callable, Dict, Optional, Tuple

from jax import numpy as jnp

from axlearn.common.attention import NEG_INF, make_segment_mask
from axlearn.common.base_encoder_decoder import BaseEncoderDecoderModel
from axlearn.common.config import ConfigOr, config_class
from axlearn.common.decoder import Decoder
from axlearn.common.decoding import BeamSearchOutputs, SampleOutputs
from axlearn.common.encoder import Encoder
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.loss import cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.utils import Nested


class EncoderDecoderModel(BaseEncoderDecoderModel):
    """Constructs an Encoder-Decoder model to output loss and logits based on input ids."""

    @config_class
    class Config(BaseEncoderDecoderModel.Config):
        z_loss_scale: float = 0.0
        label_smoothing: float = 0.0

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.encoder = Encoder.default_config()
        cfg.decoder = Decoder.default_config()
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("encoder", cfg.encoder)
        # TODO(markblee): set source dim without nesting.
        cfg.decoder.transformer.layer.cross_attention.source_dim = cfg.encoder.dim
        self._add_child("decoder", cfg.decoder)

    def predict(
        self,
        input_batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Produces encoder-decoder logits and hidden states.

        Args:
            input_batch: A dict with the following entries:
                source: A dict containing keyword arguments for the encoder:
                    input_ids: An int Tensor of shape [batch_size, source_len].
                        Values should be in the range [0, vocab_size).
                    input_segment_ids: An optional Tensor of same shape as input_ids with values
                        in [0, num_segments). Tokens are only allowed to attend to other tokens
                        within the same segment. By default, input_segment_ids == 0 represents
                        paddings; if input_segment_ids is not provided, it will be inferred from
                        input_ids != encoder.config.pad_token_id. See the corresponding `encoder`
                        implementation for details.
                    token_type_ids: An optional int Tensor of shape [batch_size, source_len].
                        Values should be in the range [0, type_vocab_size).
                    positions: An optional int Tensor of shape [batch_size, source_len].
                        If None, assumed to be jnp.arange(source_len) for each sequence.
                target: A dict containing keyword arguments for the decoder:
                    input_ids: An int Tensor of shape [batch_size, target_len].
                        Values should be in the range [0, vocab_size).
                    input_segment_ids: An optional Tensor of same shape as input_ids with values
                        in [0, num_segments). Tokens are only allowed to attend to other tokens
                        within the same segment. By default, input_segment_ids == 0 represents
                        paddings; if input_segment_ids is not provided, it will be inferred from
                        input_ids != decoder.config.pad_token_id. See the corresponding `decoder`
                        implementation for details.
                    positions: An optional int Tensor of shape [batch_size, target_len].
                        If None, assumed to be jnp.arange(target_len) for each sequence.

        Returns:
            A dict containing:
                hidden_states: A float Tensor of shape [batch_size, target_len, hidden_dim].
                logits: A float Tensor of shape [batch_size, target_len, num_classes], where
                    num_classes depends on the configured lm_head.

        Raises:
            ValueError: If source_segment_ids and target_segment_ids are not provided together.
        """
        self._validate_input_batch(input_batch, paths=["source/input_ids", "target/input_ids"])
        source_batch: Dict[str, Tensor] = input_batch["source"]
        target_batch: Dict[str, Tensor] = input_batch["target"]
        source_segment_ids: Optional[Tensor] = source_batch.get("input_segment_ids")
        target_segment_ids: Optional[Tensor] = target_batch.get("input_segment_ids")

        # Encoder hidden states: [batch_size, source_len, hidden_dim].
        encoder_output = self.encoder(**source_batch)
        # Cross-attention logit biases: [batch_size, target_len, source_len].
        cross_attention_logit_biases = self.compute_attention_logit_biases(
            source_batch["input_ids"]
        )
        if source_segment_ids is not None and target_segment_ids is not None:
            cross_attention_logit_biases += make_segment_mask(
                source_segments=source_segment_ids, target_segments=target_segment_ids
            )
        elif source_segment_ids is not None or target_segment_ids is not None:
            raise ValueError(
                "source_segment_ids and target_segment_ids should either both be None, or both not."
            )
        # Decoder hidden states: [batch_size, target_len, hidden_dim].
        decoder_output = self.decoder(
            **target_batch,
            cross_attention_data=encoder_output,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        return decoder_output

    def compute_attention_logit_biases(self, source_ids: Tensor) -> Tensor:
        """Produces cross-attention logit biases.

        Args:
            source_ids: An int Tensor of shape [batch_size, source_len].

        Returns:
            Attention logit biases of shape [batch_size, num_heads, target_len, source_len].
        """
        # Compute padding logit biases: [batch_size, num_heads=1, target_len=1, source_len].
        cross_attention_biases = (source_ids == self.encoder.config.pad_token_id) * NEG_INF
        cross_attention_biases = cross_attention_biases[:, None, None, :]
        return cross_attention_biases

    def _metrics(
        self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor]
    ) -> Tuple[Tensor, Nested[Tensor]]:
        """Computes metrics from logits and target labels like loss and per token loss.

        Args:
            input_batch: See parent `forward` for details.
            predict_outputs: A dict containing:
                logits: A float Tensor of shape [batch_size, target_len, vocab_size].

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar float Tensor.
                aux_outputs: A dict containing auxiliary metrics:
                    per_token_loss: A float Tensor of shape [batch_size, target_len].
        """
        self._validate_input_batch(input_batch, paths=["target_labels"])
        self._validate_input_batch(predict_outputs, paths=["logits"])
        logits: Tensor = predict_outputs["logits"]
        target_labels: Tensor = input_batch["target_labels"]

        num_classes = logits.shape[-1]
        live_targets = jnp.logical_and(0 <= target_labels, target_labels < num_classes)
        live_targets = jnp.logical_and(
            live_targets, target_labels != self.decoder.config.pad_token_id
        )
        num_targets = live_targets.sum()
        cfg = self.config
        loss, loss_dict = cross_entropy(
            logits=logits,
            target_labels=target_labels,
            live_targets=live_targets,
            z_loss_scale=cfg.z_loss_scale,
            label_smoothing=cfg.label_smoothing,
        )
        per_token_loss = loss_dict["pre_mask_loss"] * live_targets
        self.add_summary("loss", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))
        self.add_summary("token_accuracy", WeightedScalar(loss_dict["accuracy"], num_targets))
        return loss, dict(per_token_loss=per_token_loss)

    def beam_search_decode(
        self,
        input_batch: Dict[str, Tensor],
        max_sequence_length: int,
        num_decodes: int,
        brevity_penalty: Optional[Callable[[jnp.array, Tensor], jnp.array]] = None,
    ) -> BeamSearchOutputs:
        """Performs beam search decoding given prefix prompt.

        Args:
            input_batch: A dict with the following entries:
                prefix: An int Tensor of shape [batch_size, prefix_len] where
                    prefix_len <= `max_len`. See parent docstring for details.
                source: A dict containing keyword arguments for the encoder:
                    inputs_ids: An int Tensor of shape [batch_size, source_len].
                        Values should be in the range [0, vocab_size).
                        Paddings are inferred from input_ids == encoder.config.pad_token_id.
            max_sequence_length: See parent docstring for details.
            num_decodes: See parent docstring for details.
            brevity_penalty: See parent docstring for details.

        Returns:
            Beam search outputs. See parent docstring for details.
        """
        self._validate_input_batch(input_batch, paths=["prefix", "source/input_ids"])
        prefix: Tensor = input_batch["prefix"]
        input_ids: Tensor = input_batch["source"]["input_ids"]
        encoder_output = self.encoder(input_ids=input_ids)
        cross_attention_logit_biases = self.compute_attention_logit_biases(input_ids)

        with child_context("beam_search_decode", module=self.decoder):
            return self.decoder.beam_search_decode(
                prefix=prefix,
                max_sequence_length=max_sequence_length,
                num_decodes=num_decodes,
                cross_attention_data=encoder_output,
                cross_attention_logit_biases=cross_attention_logit_biases,
                brevity_penalty=brevity_penalty,
            )

    def sample_decode(
        self,
        input_batch: Dict[str, Tensor],
        max_sequence_length: int,
        num_decodes: int,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> SampleOutputs:
        """Perform sample-based decoding.

        Args:
            input_batch: A dict with the following entries:
                prefix: An int Tensor of shape [batch_size, prefix_len] where
                    prefix_len <= `max_len`. See parent docstring for details.
                source: A dict containing keyword arguments for the encoder:
                    inputs_ids: An int Tensor of shape [batch_size, source_len].
                        Values should be in the range [0, vocab_size).
                        Paddings are inferred from input_ids == encoder.config.pad_token_id.
            max_sequence_length: See parent docstring for details.
            num_decodes: See parent docstring for details.
            logits_modifier: See parent docstring for details.

        Returns:
            Sample decoding outputs. See parent docstring for details.
        """
        self._validate_input_batch(input_batch, paths=["prefix", "source/input_ids"])
        prefix: Tensor = input_batch["prefix"]
        input_ids: Tensor = input_batch["source"]["input_ids"]
        encoder_output = self.encoder(input_ids=input_ids)
        cross_attention_logit_biases = self.compute_attention_logit_biases(input_ids)

        with child_context("sample_decode", module=self.decoder):
            return self.decoder.sample_decode(
                prefix=prefix,
                max_sequence_length=max_sequence_length,
                num_decodes=num_decodes,
                cross_attention_data=encoder_output,
                cross_attention_logit_biases=cross_attention_logit_biases,
                logits_modifier=logits_modifier,
            )
