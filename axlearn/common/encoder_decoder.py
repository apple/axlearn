"""Encoder decoder model."""
from typing import Callable, Dict, Optional, Tuple

from jax import numpy as jnp

from axlearn.common.attention import NEG_INF, make_segment_mask
from axlearn.common.base_model import BaseModel
from axlearn.common.config import ConfigOr, config_class
from axlearn.common.decoder import Decoder
from axlearn.common.decoding import BeamSearchOutputs, SampleOutputs
from axlearn.common.encoder import Encoder
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.loss import cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, NestedTensor, Tensor, child_context


class EncoderDecoderModel(BaseModel):
    """Constructs an encoder decoder model to output loss and logits based on input ids."""

    @config_class
    class Config(BaseModel.Config):
        encoder: Encoder.Config = Encoder.default_config()
        decoder: Decoder.Config = Decoder.default_config()
        z_loss_scale: float = 0.0
        label_smoothing: float = 0.0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("encoder", cfg.encoder)
        # TODO(markblee): set source dim without nesting.
        cfg.decoder.transformer.layer.cross_attention.source_dim = cfg.encoder.dim
        self._add_child("decoder", cfg.decoder)

    # We drop the kwargs from BaseModel, since they aren't used here.
    # pylint: disable-next=arguments-differ
    def forward(
        self,
        input_batch: Dict[str, Tensor],
        return_aux: Optional[bool] = False,
    ) -> Tuple[Tensor, NestedTensor]:
        """Produces encoder-decoder loss and predictions including logits and decoder hidden states
        in auxiliary outputs.

        Args:
            input_batch: A dict with the following entries:
                source_ids: An int Tensor of shape [batch_size, source_len].
                    Used as encoder input ids. Values should be in the range [0, vocab_size).
                target_ids: An int Tensor of shape [batch_size, target_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size).
                target_labels: An optional int Tensor of shape [batch_size, target_len].
                    Used to calculate loss. Values should be in the range [0, vocab_size).
                    Out-of-class labels are ignored.
                source_token_type_ids: An optional int Tensor of shape [batch_size, source_len].
                    Values should be in the range [0, type_vocab_size).
                source_segment_ids: An optional Tensor of same shape as `source_ids` with values in
                    [0, num_segments). Tokens are only allowed to attend to other tokens within the
                    same segment.
                target_segment_ids: An optional Tensor of same shape as `target_ids` with values in
                    [0, num_segments). Tokens are only allowed to attend to other tokens within the
                    same segment.
                source_positions: An optional int Tensor of shape [batch_size, source_len].
                    If None, assumed to be jnp.arange(source_len) for each sequence.
                target_positions: An optional int Tensor of shape [batch_size, target_len].
                    If None, assumed to be jnp.arange(target_len) for each sequence.
            return_aux: Boolean to determine whether logits, per_token_loss and decoder
                hidden states are returned.

        Returns:
            loss: A scalar float Tensor representing the cross-entropy loss.
            decoder_output a dict containing:
                logits: A float Tensor of shape [batch_size, target_len, vocab_size]
                hidden_states: A float Tensor of shape [batch_size, target_len, hidden_dim]
        """
        decoder_output = self.predict(input_batch)
        loss_dict = self._metrics(decoder_output["logits"], input_batch["target_labels"])
        aux_output = dict(
            **decoder_output,
            per_token_loss=loss_dict["per_token_loss"],
        )
        # If return_aux, return the logits and pre LM head hidden states (useful for downstream
        # tasks).
        #
        # N.B. Do not enable for large-scale training since auxiliary outputs are not partitioned.
        # TODO(rpang): support partitioning of auxiliary outputs.
        return loss_dict["loss"], aux_output if return_aux else {}

    def predict(
        self,
        input_batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Produces encoder-decoder logits and hidden states.

        Args:
            input_batch: A dict with the following entries:
                source_ids: An int Tensor of shape [batch_size, source_len].
                    Used as encoder input ids. Values should be in the range [0, vocab_size).
                target_ids: An int Tensor of shape [batch_size, target_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size).
                source_token_type_ids: An optional int Tensor of shape [batch_size, source_len].
                    Values should be in the range [0, source_type_vocab_size).
                source_segment_ids: An optional Tensor of same shape as `source_ids` with values in
                    [0, num_segments). Tokens are only allowed to attend to other tokens within the
                    same segment.
                target_segment_ids: An optional Tensor of same shape as `target_ids` with values in
                    [0, num_segments). Tokens are only allowed to attend to other tokens within the
                    same segment.
                source_positions: An optional int Tensor of shape [batch_size, source_len].
                    If None, assumed to be jnp.arange(source_len) for each sequence.
                target_positions: An optional int Tensor of shape [batch_size, target_len].
                    If None, assumed to be jnp.arange(target_len) for each sequence.

        Returns:
            A dict containing:
                hidden_states: A float Tensor of shape [batch_size, target_len, hidden_dim].
                logits: A float Tensor of shape [batch_size, target_len, num_classes], where
                    num_classes depends on the configured lm_head.

        Raises:
            ValueError: If source_segment_ids and target_segment_ids are not provided together.
        """
        source_ids: Tensor = input_batch["source_ids"]
        target_ids: Tensor = input_batch["target_ids"]
        source_segment_ids: Optional[Tensor] = input_batch.get("source_segment_ids")
        target_segment_ids: Optional[Tensor] = input_batch.get("target_segment_ids")

        # Encoder hidden states: [batch_size, source_len, hidden_dim].
        encoder_output = self.encoder(
            input_ids=source_ids,
            input_segment_ids=source_segment_ids,
            token_type_ids=input_batch.get("source_token_type_ids"),
            positions=input_batch.get("source_positions"),
        )
        # Cross-attention logit biases: [batch_size, target_len, source_len].
        cross_attention_logit_biases = self.compute_attention_logit_biases(source_ids)
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
            input_ids=target_ids,
            input_segment_ids=target_segment_ids,
            positions=input_batch.get("target_positions"),
            cross_attention_data=encoder_output,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        return decoder_output

    def compute_attention_logit_biases(self, source_ids: Tensor) -> Tensor:
        """Produces cross-attention logit biases.

        Args:
            source_ids: A Tensor of shape [batch_size, source_len].

        Returns:
            Attention logit biases of shape [batch_size, num_heads, target_len, source_len].
        """
        # Compute padding logit biases: [batch_size, num_heads=1, target_len=1, source_len].
        cross_attention_biases = (source_ids == self.encoder.config.pad_token_id) * NEG_INF
        cross_attention_biases = cross_attention_biases[:, None, None, :]
        return cross_attention_biases

    # TODO(gyin): add score/inference function
    def _metrics(self, logits: Tensor, target_labels: Tensor) -> Dict[str, Tensor]:
        """Computes metrics from logits and target labels like loss and per token loss.

        Args:
            logits: A float Tensor of shape [batch_size, target_len, vocab_size].
            target_labels: An int Tensor of shape [batch_size, target_len].
                Values should be in the range [0, vocab_size). Out-of-class labels are ignored.

        Returns:
            A dict containing:
                loss: A scalar float Tensor.
                per_token_loss: A float Tensor of shape [batch_size, target_len].
        """
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
            mask=live_targets,
            z_loss_scale=cfg.z_loss_scale,
            label_smoothing=cfg.label_smoothing,
        )
        per_token_loss = loss_dict["pre_mask_loss"] * live_targets
        self.add_summary("loss", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))
        self.add_summary("token_accuracy", WeightedScalar(loss_dict["accuracy"], num_targets))
        return dict(loss=loss, per_token_loss=per_token_loss)

    def beam_search_decode(
        self,
        input_batch: Dict[str, Tensor],
        max_len: int,
        num_decodes: int,
        brevity_penalty: Optional[Callable[[jnp.array, Tensor], jnp.array]] = None,
    ) -> BeamSearchOutputs:
        """Performs beam search decoding given prefix prompt.

        Args:
            input_batch: A dict with the following entries:
                prefix: An int Tensor of shape [batch_size, prefix_len] where
                    prefix_len <= `max_len`.
                source_ids: An int Tensor of shape [batch_size, source_len].
                    Used as encoder input ids. Values should be in the range [0, vocab_size).
                target_ids: An int Tensor of shape [batch_size, target_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size).
                target_labels: An int Tensor of shape [batch_size, target_len].
                    Used as ground truth.
            max_len: The maximum sequence length of tokens to generate.
            num_decodes: The number of beams to decode. If set to 1, equivalent to greedy decoding.
            brevity_penalty: The length normalization function for beam search.

        Returns:
            Beam search outputs with sequences of shape [batch_size, num_decodes, max_len] and
            scores of shape [batch_size, num_decodes].
        """
        source_ids: Tensor = input_batch["source_ids"]
        prefix: Tensor = input_batch["prefix"]
        encoder_output = self.encoder(input_ids=source_ids)
        cross_attention_logit_biases = self.compute_attention_logit_biases(source_ids)

        with child_context("beam_search_decode", module=self.decoder):
            return self.decoder.beam_search_decode(
                prefix=prefix,
                max_sequence_length=max_len,
                num_decodes=num_decodes,
                cross_attention_data=encoder_output,
                cross_attention_logit_biases=cross_attention_logit_biases,
                brevity_penalty=brevity_penalty,
            )

    def sample_decode(
        self,
        input_batch: Dict[str, Tensor],
        max_len: int,
        num_decodes: int,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> SampleOutputs:
        """Perform sample-based decoding.

        Args:
            input_batch: A dict with the following entries:
                prefix: An int Tensor of shape [batch_size, prefix_len] where
                    prefix_len <= `max_len`.
                source_ids: An int Tensor of shape [batch_size, source_len].
                    Used as encoder input ids. Values should be in the range [0, vocab_size).
            max_len: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sequences to return.
                These are the number of hypotheses per batch example.
            logits_modifier: Function used to adjust the raw next-token logit distribution values,
                to e.g. implement top-k/top-p/etc sampling (see `logit_modifiers`).
                If None, do not modify the logits.

        Returns:
            The sample decoding outputs.
        """
        source_ids: Tensor = input_batch["source_ids"]
        prefix: Tensor = input_batch["prefix"]
        encoder_output = self.encoder(input_ids=source_ids)
        cross_attention_logit_biases = self.compute_attention_logit_biases(source_ids)

        with child_context("sample_decode", module=self.decoder):
            return self.decoder.sample_decode(
                prefix=prefix,
                max_sequence_length=max_len,
                num_decodes=num_decodes,
                cross_attention_data=encoder_output,
                cross_attention_logit_biases=cross_attention_logit_biases,
                logits_modifier=logits_modifier,
            )
