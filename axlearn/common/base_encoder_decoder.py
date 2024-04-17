# Copyright © 2023 Apple Inc.

"""Base Encoder-Decoder model interface."""

from typing import Dict, Optional, Sequence, Tuple

from axlearn.common import decoding
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class
from axlearn.common.decoding import BeamSearchOutputs, SampleOutputs
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.utils import Nested, Tensor, get_recursively


class BaseEncoderDecoderModel(BaseModel):
    """Defines the interface for Encoder-Decoder model implementations."""

    @config_class
    class Config(BaseModel.Config):
        """Configures BaseEncoderDecoderModel."""

        encoder: Required[BaseLayer.Config] = REQUIRED
        decoder: Required[BaseLayer.Config] = REQUIRED

    # We drop the kwargs from BaseModel, since they aren't used here.
    # pylint: disable-next=arguments-differ
    def forward(
        self,
        input_batch: Nested[Tensor],
        return_aux: bool = False,
    ) -> Tuple[Tensor, Nested[Tensor]]:
        """Produces Encoder-Decoder loss and predictions (such as logits and decoder hidden states)
        in auxiliary outputs.

        Args:
            input_batch: A dict with the following entries:
                source: A dict containing keyword arguments for the encoder.
                target: A dict containing keyword arguments for the decoder.
                target_labels: An int Tensor of shape [batch_size, target_len] for computing loss.
                    To represent paddings, use target_labels < 0.
            return_aux: Boolean to determine whether auxiliary outputs and metrics are returned.

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar float Tensor representing the cross-entropy loss.
                aux_outputs: A dict containing auxiliary outputs if `return_aux=True`; otherwise, an
                    empty dict.
        """
        self._validate_input_batch(input_batch, paths=["source", "target", "target_labels"])
        predict_outputs = self.predict(input_batch)
        loss, aux_metrics = self._metrics(input_batch, predict_outputs=predict_outputs)
        aux_output = dict(**predict_outputs, **aux_metrics)
        # N.B. Do not enable for large-scale training since auxiliary outputs are not partitioned.
        # TODO(rpang): support partitioning of auxiliary outputs.
        return loss, aux_output if return_aux else {}

    def predict(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Produces Encoder-Decoder logits and hidden states.

        Args:
            input_batch: See `forward` for details, except the input_batch need not contain
                target_labels.

        Returns:
            A dict containing prediction outputs, such as logits or decoder hidden states.
        """
        raise NotImplementedError(type(self))

    def _metrics(
        self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor]
    ) -> Tuple[Tensor, Nested[Tensor]]:
        """Computes metrics from logits and target labels like loss and per token loss.

        Args:
            input_batch: See `forward` for details.
            predict_outputs: Outputs from `predict(input_batch)`.

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar float Tensor.
                aux_outputs: A dict containing auxiliary metrics.
        """
        raise NotImplementedError(type(self))

    def _validate_input_batch(self, input_batch: Nested[Tensor], paths: Sequence[str]):
        """Raises ValueError if any of the given `paths` are not present in `input_batch`."""
        for path in paths:
            try:
                get_recursively(input_batch, path)
            except KeyError as e:
                raise ValueError(f"Input batch is expected to contain '{path}'.") from e

    def beam_search_decode(
        self,
        input_batch: Dict[str, Tensor],
        max_sequence_length: int,
        num_decodes: int,
        brevity_penalty: Optional[decoding.BrevityPenaltyFn] = None,
    ) -> BeamSearchOutputs:
        """Performs beam search decoding given prefix prompt.

        Args:
            input_batch: A dict with the following entries:
                prefix: An int Tensor of shape [batch_size, prefix_len] where
                    prefix_len <= `max_sequence_length`. See `decoding.beam_search_decode` for
                    details, such as how decoding is initialized from the `prefix`.
                source: A dict containing keyword arguments for the encoder.
            max_sequence_length: The maximum sequence length of tokens to generate, including the
                length of `prefix`.
            num_decodes: The number of beams to decode. If set to 1, equivalent to greedy decoding.
            brevity_penalty: Optional length normalization function for beam search. See
                `decoding.brevity_penalty_fn` for an example.

        Returns:
            Beam search outputs:
                sequences: A Tensor of shape [batch_size, num_decodes, max_sequence_length].
                scores: A Tensor of shape [batch_size, num_decodes].
        """
        raise NotImplementedError(type(self))

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
                    prefix_len <= `max_sequence_length`. See `decoding.sample_decode` for details,
                    such as how decoding is initialized from the `prefix`.
                source: A dict containing keyword arguments for the encoder.
            max_sequence_length: The maximum sequence length of tokens to generate, including the
                length of `prefix`.
            num_decodes: The number of decoded sequences to return.
                These are the number of hypotheses per batch example.
            logits_modifier: Function used to adjust the raw next-token logit distribution values,
                to e.g. implement top-k/top-p/etc sampling (see `logit_modifiers` for examples).
                If None, do not modify the logits.

        Returns:
            Sample decoding outputs:
                sequences: A Tensor of shape [batch_size, num_decodes, max_sequence_length].
                token_scores: A Tensor of shape [batch_size, num_decodes, max_sequence_length].
        """
        raise NotImplementedError(type(self))
