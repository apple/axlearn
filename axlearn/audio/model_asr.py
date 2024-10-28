# Copyright © 2024 Apple Inc.

"""ASR model layers."""

from typing import Optional

from axlearn.audio.decoder_asr import BaseASRDecoderModel, DecodeOutputs
from axlearn.audio.encoder_asr import ASREncoder
from axlearn.common.base_encoder_decoder import BaseEncoderDecoderModel
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor


class ASRModel(BaseEncoderDecoderModel):
    """ASR Encoder-Decoder model.

    The implementation closely follows the `BaseEncoderDecoderModel` API, although depending on the
    configured decoder, certain decoding methods (e.g. `beam_search_decode` or `sample_decode`) may
    expect different keyword arguments, which are exposed as `kwargs` in the API. Please refer to
    the configured decoder docstring for specifics.
    """

    @config_class
    class Config(BaseEncoderDecoderModel.Config):
        """Configures ASRModel."""

        encoder: Required[ASREncoder.Config] = REQUIRED
        decoder: Required[BaseASRDecoderModel.Config] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: ASRModel.Config = self.config
        self._add_child("encoder", cfg.encoder)
        self._add_child("decoder", cfg.decoder.set(input_dim=cfg.encoder.dim))

    def predict(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Produces encoder-decoder logits.

        Args:
            input_batch: A dict with the following entries:
                source: A dict containing keyword arguments for the encoder:
                    inputs: A Tensor of shape [batch_size, seq_len].
                    paddings: A 0/1 Tensor of shape [batch_size, seq_len]. 1's represent paddings.
                target: A dict containing keyword arguments for the decoder:
                    input_ids: An int Tensor of shape [batch_size, num_labels].
                        Values should be in the range [0, vocab_size). Out-of-range values
                        are paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels].

        Returns:
            A dict containing logits. The shape of logits depend on the decoder.
        """
        self._validate_input_batch(input_batch, ["source", "target", "target_labels"])
        # Encoder hidden states: [batch_size, source_len, dim].
        encoder_output = self.encoder(**input_batch["source"])
        logits = self.decoder.predict(
            input_batch=dict(inputs=encoder_output["outputs"], paddings=encoder_output["paddings"])
        )
        return dict(logits=logits)

    def forward(
        self, input_batch: Nested[Tensor], *, return_aux: bool = False
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Computes loss and predictions (such as logits) in auxiliary outputs.

        Args:
            input_batch: A dict with the following entries:
                source: A dict containing keyword arguments for the encoder:
                    inputs: A float Tensor of shape [batch_size, seq_len].
                        Values should not be normalized.
                    paddings: A 0/1 Tensor of shape [batch_size, seq_len].
                target: A dict containing keyword arguments for the decoder.
                    See corresponding decoder `forward` for details.
                target_labels: An int Tensor of shape [batch_size, num_labels] for computing loss.
                    To represent paddings, use out-of-range values.
            return_aux: Boolean to determine whether auxiliary outputs and metrics are returned.

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar float Tensor representing the loss.
                aux_outputs: A dict containing auxiliary outputs if `return_aux=True`, otherwise an
                    empty dict.
        """
        self._validate_input_batch(input_batch, ["source", "target", "target_labels"])
        # Encoder hidden states: [batch_size, source_len, dim].
        encoder_output = self.encoder(**input_batch["source"])
        loss, aux_outputs = self.decoder(
            input_batch=dict(
                inputs=encoder_output["outputs"],
                paddings=encoder_output["paddings"],
                target_labels=input_batch["target_labels"],
                target=input_batch["target"],
            )
        )
        return loss, aux_outputs if return_aux else {}

    def beam_search_decode(
        self, input_batch: Nested[Tensor], num_decodes: int, **kwargs
    ) -> DecodeOutputs:
        """Performs beam search decoding.

        Args:
            input_batch: A dict with the following entries:
                inputs: A Tensor of shape [batch_size, num_frames, dim] containing inputs to
                    decoder `beam_search_decode`.
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                source: A dict containing keyword arguments for the encoder:
                    inputs: A float Tensor of shape [batch_size, seq_len].
                        Values should not be normalized.
                    paddings: A 0/1 Tensor of shape [batch_size, seq_len].
            num_decodes: The number of beams to decode.
            kwargs: Additional kwargs for `decoder.beam_search_decode`.

        Returns:
            Beam search decode outputs.
        """
        self._validate_input_batch(input_batch, ["source/inputs", "source/paddings"])
        encoder_output = self.encoder(**input_batch["source"])
        return self.decoder.beam_search_decode(
            input_batch=dict(
                inputs=encoder_output["outputs"],
                paddings=encoder_output["paddings"],
                **input_batch,
            ),
            num_decodes=num_decodes,
            **kwargs,
        )
