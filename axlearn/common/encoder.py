# Copyright © 2023 Apple Inc.

"""Encoder layers."""

import math
from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common import param_init
from axlearn.common.attention import (
    AttentionLogitBiasLayer,
    BaseTransformerLayer,
    CausalAttentionLogitBiasLayer,
    FullAttentionLogitBiasLayer,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import BaseClassificationHead, set_dropout_rate_recursively
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.utils import NestedTensor, TensorSpec


class Encoder(BaseLayer):
    """Construct an encoder transformer to output hidden states based on input ids."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Encoder."""

        dim: Required[int] = REQUIRED  # Hidden dimension.
        vocab_size: Required[int] = REQUIRED  # Vocab size.
        pad_token_id: Required[int] = REQUIRED  # Value of input id corresponding to padding.
        dropout_rate: float = 0.0  # Dropout rate.
        attention_mask: AttentionLogitBiasLayer.Config = (
            FullAttentionLogitBiasLayer.default_config()
        )
        # Input embedding.
        emb: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config()
        # The transformer stack.
        transformer: Required[BaseTransformerLayer.Config] = REQUIRED
        # If not None, a layer to process transformer outputs. This can represent, e.g.,
        # the final layer norm + dropout in the T5 encoder (or other encoders that use the prenorm
        # structure): https://arxiv.org/abs/2002.04745.
        output: Optional[InstantiableConfig] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.vlog(3, "dim=%d, vocab_size=%d", cfg.dim, cfg.vocab_size)
        set_dropout_rate_recursively(cfg, dropout_rate=cfg.dropout_rate, set_only_if_none=True)
        self._add_child("emb", cfg.emb.set(dim=cfg.dim, vocab_size=cfg.vocab_size))
        self._add_child("transformer", cfg.transformer.set(input_dim=cfg.dim))
        if cfg.output is not None:
            self._add_child("output", cfg.output.set(input_dim=cfg.dim))
        self._add_child("attention_mask", cfg.attention_mask)

    # TODO(markblee): Generalize to support input_batch, similar to Decoder.
    def forward(
        self,
        input_ids: Tensor,
        input_segment_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes encoder hidden states from the input tokens and types.

        Args:
            input_ids: An int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, vocab_size).
            input_segment_ids: An optional Tensor of same shape as `input_ids` with values in
                [0, num_segments). Tokens are only allowed to attend to other tokens within the same
                segment.
            token_type_ids: An optional int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, type_vocab_size).
            positions: An optional int Tensor of shape [batch_size, seq_len].
                If None, assumed to be jnp.arange(seq_len) for each sequence.

        Returns:
            A Tensor of shape [batch_size, seq_len, hidden_dim].
        """
        # [batch_size, seq_len, hidden_dim].
        x = self.emb(
            input_batch=dict(inputs=input_ids, token_type_ids=token_type_ids, positions=positions)
        )
        # [batch_size, num_heads, seq_len, seq_len].
        attention_logit_biases = self.compute_attention_logit_biases(
            input_ids, segment_ids=input_segment_ids, positions=positions
        )
        # [batch_size, seq_len, hidden_dim].
        transformer_outputs = self.transformer(
            x, self_attention_logit_biases=attention_logit_biases
        )
        self.add_module_output("transformer_outputs", transformer_outputs)
        x = transformer_outputs.data
        if "output" in self.children:
            x = self.output(x)
        return x

    def compute_attention_logit_biases(
        self,
        input_ids: Tensor,
        *,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """Produces self-attention logit biases.

        Args:
            input_ids: A Tensor of shape [batch_size, seq_len].
            segment_ids: An optional integer Tensor of shape [batch_size, seq_len] with values in
                [0, num_segments). Tokens are only allowed to attend to other tokens within the same
                segment. segment_ids == 0 represents paddings. If provided, positions should also be
                provided.
            positions: An optional Tensor of same shape as `input_ids` with values in [0, seq_len).
                This can be used to produce biases for packed inputs. If provided, segment_ids
                should also be provided.

        Returns:
            Attention logit biases of shape [batch_size, seq_len, seq_len] or
                [batch_size, num_heads, seq_len, seq_len].

        Raises:
            ValueError: If segment_ids and positions are not both provided, or both omitted.
        """
        if (segment_ids is None) != (positions is None):
            raise ValueError("segment_ids and positions must be provided together")
        cfg = self.config
        if segment_ids is None or positions is None:
            segment_ids = input_ids != cfg.pad_token_id
            positions = jnp.arange(input_ids.shape[-1])[None, :]  # [batch=1, seq_len].
        return self.attention_mask(segment_ids=segment_ids, positions=positions)

    def attend(self, x: Tensor) -> Tensor:
        """Computes logits with token embedding.

        Args:
            x: an int Tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            A float Tensor of shape [batch_size, seq_len, vocab_size].
        """
        with child_context("emb_attend", module=self.emb):
            return self.emb.attend(x)


class CausalEncoder(Encoder):
    """The encoder uses a CausalAttentionLogitBiasLayer.

    With extend_step function, this CausalEncoder can be used in generative models for decoding.
    It is useful for multi-stage models like CoCa (https://arxiv.org/pdf/2205.01917.pdf).
    """

    @config_class
    class Config(Encoder.Config):
        attention_mask: AttentionLogitBiasLayer.Config = (
            CausalAttentionLogitBiasLayer.default_config()
        )
        num_cls_tokens: int = 0  # cls tokens to be appended at the end of seq.

    def init_states(self, *, batch_size: int, max_sequence_length: int) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            batch_size: the batch size of the target to be decoded.
            max_sequence_length: the sequence length of the target to be decoded.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        cfg: CausalEncoder.Config = self.config
        init_state, _ = self.transformer.init_states(
            time_step=None,
            data=TensorSpec([batch_size, max_sequence_length, cfg.dim], dtype=jnp.float32),
        )
        return dict(
            transformer_state=init_state,
            input_ids=jnp.full(
                (batch_size, max_sequence_length), cfg.pad_token_id, dtype=jnp.int32
            ),
            time_step=jnp.zeros(batch_size, dtype=jnp.int32),
        )

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        param_specs = {}
        if cfg.num_cls_tokens > 0:
            # Init consistent with token embedding
            # https://github.com/google-research/t5x/blob/f7978d63448c43bdb339ae73fa557ba472be30d6/t5x/examples/scalable_t5/layers.py#L535
            param_specs["cls_token"] = ParameterSpec(
                shape=(1, cfg.num_cls_tokens, cfg.dim),
                mesh_axes=(None, None, "model"),
                initializer=param_init.gaussian_initializer(std=1.0 / math.sqrt(cfg.dim)),
            )
        return param_specs

    def forward(
        self,
        input_ids: Tensor,
        input_segment_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> NestedTensor:
        """Computes encoder hidden states from the input tokens and types.

        Args:
            input_ids: An int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, vocab_size).
            input_segment_ids: An optional Tensor of same shape as `input_ids` with values in
                [0, num_segments). Tokens are only allowed to attend to other tokens within the same
                segment.
            token_type_ids: An optional int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, type_vocab_size).
            positions: An optional int Tensor of shape [batch_size, seq_len].
                If None, assumed to be jnp.arange(seq_len) for each sequence.

        Returns:
            A dictionary containing:
                *"hidden_states": A Tensor with shape
                    [batch_size, seq_len + num_cls_tokens, hidden_dim] for
                    unnormalized hidden states.
                *"normalized_states": A Tensor with shape
                    [batch_size, seq_len + num_cls_tokens, hidden_dim] for
                    normalized hidden states if output is set.

        Raises:
            ValueError: If pad token id is not set for cls token mode or if
                segment id is enabled for cls token mode.
        """
        cfg = self.config
        batch_size, max_seq_len = input_ids.shape

        # [batch_size, seq_len, hidden_dim].
        x = self.emb(
            input_batch=dict(inputs=input_ids, token_type_ids=token_type_ids, positions=positions)
        )

        # Append optional cls tokens as used in CoCa.
        if cfg.num_cls_tokens > 0:
            if cfg.pad_token_id is None:
                raise ValueError("Pad token id must be set for causal masking with cls tokens!")

            if input_segment_ids is not None:
                raise ValueError("Segment not supported for cls tokens yet!")

            if positions is not None:
                # Set dummy positions for cls tokens for causal masking.
                cls_positions = jnp.arange(cfg.num_cls_tokens)[None, :] + max_seq_len
                cls_positions = jnp.tile(cls_positions, (batch_size, 1))
                positions = jnp.concatenate([positions, cls_positions], axis=1)

            # Add dummy ids other than pad id for causal mask.
            dummy_ids = jnp.ones([batch_size, cfg.num_cls_tokens], dtype=input_ids.dtype) * (
                cfg.pad_token_id + 1
            )
            input_ids = jnp.concatenate([input_ids, dummy_ids], axis=1)

            cls_tokens = jnp.tile(self.parameters["cls_token"], (batch_size, 1, 1))
            x = jnp.concatenate([x, cls_tokens], axis=1)

        # [batch_size, num_heads, seq_len, seq_len].
        attention_logit_biases = self.compute_attention_logit_biases(
            input_ids, segment_ids=input_segment_ids, positions=positions
        )
        # [batch_size, seq_len, hidden_dim].
        transformer_outputs = self.transformer(
            x, self_attention_logit_biases=attention_logit_biases
        )
        self.add_module_output("transformer_outputs", transformer_outputs)
        x = transformer_outputs.data
        output_dict = {
            "hidden_states": x,
        }
        if "output" in self.children:
            output_dict["normalized_states"] = self.output(x)
        return output_dict

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        # Note: this follows `Decoder.prefill_states` closely. Refer to that method for details.
        # TODO(markblee): Possibly consolidate some of this with decoder.
        x = self.emb(
            input_batch=dict(inputs=input_ids, token_type_ids=token_type_ids, positions=None)
        )
        transformer_state, x = self.transformer.init_states(
            time_step=time_step,
            data=x,
            self_attention_logit_biases=self.compute_attention_logit_biases(input_ids),
        )
        x = x.data
        states = dict(time_step=time_step, input_ids=input_ids, transformer_state=transformer_state)
        return states, dict(hidden_states=x)

    def extend_step(
        self,
        *,
        cached_states: NestedTensor,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        cfg = self.config
        # Note: this follows `Decoder.extend_step` closely. Refer to that method for details.
        # TODO(markblee): Possibly consolidate some of this with decoder.
        time_step = cached_states["time_step"]
        assert time_step.ndim == 1

        # Update cache_input_ids.
        # Note: in the cases where `time_step` exceeds `target_len`, the update becomes a no-op.
        cached_input_ids = cached_states["input_ids"]
        source_len = cached_input_ids.shape[-1]
        oh_indices = jax.nn.one_hot(time_step, source_len, dtype=input_ids.dtype)
        cache_input_ids = cached_input_ids * (1 - oh_indices) + input_ids * oh_indices

        # Compute self-attention-mask logit biases.
        self_attention_biases = self.compute_attention_logit_biases(
            cache_input_ids,
            segment_ids=cache_input_ids != cfg.pad_token_id,
            positions=jnp.arange(source_len)[None, :],
        )
        # Select logit biases corresponding to time step [batch, num_heads, 1, source_length].
        # Note: if `time_step` exceeds `target_len`, e.g. in the case where one decode starts at a
        # later index than another, clip the indices instead of producing NaNs.
        # TODO(markblee): Update attention masks to support explicit positions, so we can skip this.
        self_attention_biases = jnp.take_along_axis(
            self_attention_biases, time_step[:, None, None, None], axis=2, mode="clip"
        )

        # [B, 1, D].
        x = self.emb(
            input_batch=dict(
                inputs=input_ids,
                positions=jnp.expand_dims(time_step, 1),
                token_type_ids=token_type_ids,
            )
        )
        updated_transformer_state, transformer_data = self.transformer.extend_step(
            cached_states=cached_states["transformer_state"],
            data=x,
            self_attention_logit_biases=self_attention_biases,
        )
        x = transformer_data.data
        # Notice that we do not apply layer_norm in extend_step as it is
        # applied for unimodal features only.

        updated_state = dict(
            transformer_state=updated_transformer_state,
            input_ids=cache_input_ids,
            time_step=cached_states["time_step"] + 1,
        )
        return updated_state, {"hidden_states": x}


class EncoderModel(BaseModel):
    """A general BaseModel for an encoder-only model such as BERT.

    Encoder-only models can produce a single prediction per input token,
    returned by the `sequence_output` of `predict()`.

    When a head config is specified, it can also return logits for an entire input sequence.
    """

    @config_class
    class Config(BaseModel.Config):
        dim: Required[int] = REQUIRED  # Hidden dim.
        vocab_size: Required[int] = REQUIRED  # Vocab size.
        encoder: InstantiableConfig = Encoder.default_config()  # Encoder.
        # Optional head.
        head: Optional[BaseClassificationHead.Config] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("encoder", cfg.encoder.set(dim=cfg.dim, vocab_size=cfg.vocab_size))
        if cfg.head is not None:
            self._add_child("head", cfg.head.set(input_dim=cfg.dim))

    def forward(
        self,
        input_batch: dict[str, Tensor],
        return_aux: bool = False,
    ) -> tuple[Tensor, NestedTensor]:
        """Produces prediction scores from the input tokens and types.

        Args:
            input_batch: A dict with the following entries:
                input_ids: An int Tensor of shape [..., seq_len]
                    representing token IDs in the vocabulary.
                    Values should be in the range [0, vocab_size).
                target_labels: An int Tensor. The semantics of this input depends on the type of
                    head being used. To run without `target_labels`, use `predict` instead.
                token_type_ids: An optional int Tensor of same shape as `input_ids`
                    representing different parts of the input.
                    Values should be in the range [0, type_vocab_size).
                    If unspecified and type embedding is used, defaults to 0 for all positions.
                soft_labels: An optional Tensor of labels that are already smoothed/in one-hot
                    form. If provided, target_labels are typically used for inferring the mask
                    during loss calculation.
            return_aux: whether to return auxiliary outputs.

        Returns:
            (loss, aux_outputs), where loss is a scalar output if head and target_labels are
            provided or None otherwise, and aux_outputs is a dict containing:
                sequence_output: A Tensor of shape [..., seq_len, hidden_dim].
                logits: A Tensor of shape [batch_size, num_classes], if head config is specified.
            if return_aux is True, or an empty dict otherwise.
        """
        target_labels: Tensor = input_batch["target_labels"]
        soft_labels = input_batch.get("soft_labels", None)
        predictions = self.predict(input_batch)
        loss = None
        if "head" in self.children:
            with child_context("head_loss", module=self.head):
                loss = self.head.loss(
                    logits=predictions["logits"],
                    target_labels=target_labels,
                    soft_labels=soft_labels,
                )
        return loss, predictions if return_aux else {}

    def predict(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Outputs predictions for the given inputs.

        Args:
            input_batch: A dict with the following entries:
                input_ids: An int Tensor of shape [..., seq_len]
                    representing token IDs in the vocabulary.
                    Values should be in the range [0, vocab_size).
                token_type_ids: An optional int Tensor of same shape as `input_ids`
                    representing different parts of the input.
                    Values should be in the range [0, type_vocab_size).
                    If unspecified and type embedding is used, defaults to 0 for all positions.

        Returns:
            A dict containing:
                sequence_output: A Tensor of shape [..., seq_len, hidden_dim].
                logits: If head config is specified, a Tensor of shape [..., num_classes].
        """
        input_ids: Tensor = input_batch["input_ids"]
        token_type_ids: Optional[Tensor] = input_batch.get("token_type_ids")
        batch_shape, seq_len = input_ids.shape[:-1], input_ids.shape[-1]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape((-1, seq_len))
        sequence_output = self.encoder(
            input_ids=input_ids.reshape((-1, seq_len)),
            token_type_ids=token_type_ids,
        )
        sequence_output = sequence_output.reshape(batch_shape + (seq_len, -1))
        predictions = dict(sequence_output=sequence_output)
        if self.config.head is not None:
            predictions["logits"] = self.head(
                input_batch=dict(hidden_states=sequence_output, **input_batch)
            )
        return predictions
