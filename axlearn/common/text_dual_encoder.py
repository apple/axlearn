# Copyright © 2023 Apple Inc.

"""Text-based dual-encoder module."""

from collections.abc import Sequence
from typing import Optional, Union

import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Linear
from axlearn.common.loss import (
    asymmetric_contrastive_loss_from_logits,
    contrastive_logits,
    flops_loss,
    ranking_pairwise_loss,
)
from axlearn.common.module import Module, child_context
from axlearn.common.multi_stream_model import FusionNetwork, MultiStreamModel, StreamEncoder
from axlearn.common.param_init import constant_initializer
from axlearn.common.schedule import ScheduleFn, as_schedule_fn
from axlearn.common.text_encoder import TEXT_EMBEDDINGS, TextEmbeddingEncoder
from axlearn.common.utils import NestedTensor, Tensor, get_recursively

POSITIVE_EMBEDDINGS = "positive_embeddings"
NEGATIVE_EMBEDDINGS = "negative_embeddings"
POSITIVE_PADDINGS = "positive_paddings"
NEGATIVE_PADDINGS = "negative_paddings"
POSITIVE_INPUT_IDS = "positive_input_ids"
NEGATIVE_INPUT_IDS = "negative_input_ids"
RIGHT_PADDINGS = "right_paddings"

FLATTENED_LEFT_EMBEDDINGS = "flattened_left_embeddings"
FLATTENED_RIGHT_EMBEDDINGS = "flattened_right_embeddings"

SIMILARITY_MATRIX = "similarity_matrix"

PAIRWISE_LOSS_INPUT_IDS = "pairwise_loss_input_ids"
PAIRWISE_LOSS_PADDINGS = "pairwise_loss_paddings"
PAIRWISE_LOSS_EMBEDDINGS = "pairwise_loss_embeddings"
RANKS = "ranks"

NUM_VALID_RANKING_PAIRS = "num_valid_ranking_pairs"

ENCODING_FIELD_MAP = {
    POSITIVE_EMBEDDINGS: POSITIVE_INPUT_IDS,
    NEGATIVE_EMBEDDINGS: NEGATIVE_INPUT_IDS,
    PAIRWISE_LOSS_EMBEDDINGS: PAIRWISE_LOSS_INPUT_IDS,
}

TEXT_DUAL_ENCODER_SHARED_MODULE_NAME = "shared_text_encoder"


class TextEmbeddingStreamEncoder(StreamEncoder):
    """A StreamEncoder that encodes inputs with a configured TextEmbeddingEncoder and outputs
    embeddings optionally applied with a linear projection and/or a normalization layer.
    """

    @config_class
    class Config(StreamEncoder.Config):
        """Configures TextEmbeddingStreamEncoder."""

        # Output dimension from this TextEmbeddingStreamEncoder.
        output_dim: Required[int] = REQUIRED
        # A map having output embedding name as key and input id name as value. All specified input
        # ids will be encoded by text_encoder and stored in input_batch with output embedding name
        # as the field name.
        encoding_field_map: dict[str, str] = ENCODING_FIELD_MAP
        # Text encoder that outputs a single embedding vector for each input sequence.
        text_encoder: TextEmbeddingEncoder.Config = TextEmbeddingEncoder.default_config()
        # Hidden dimension of base text_encoder. If None, it is assumed to be the same as
        # output_dim.
        hidden_dim: Optional[int] = None
        # Optional linear projection layer applied on embedding from text_encoder. If None,
        # embedding from text_encoder is taken as it is.
        output_proj: Optional[Linear.Config] = None
        # Optional normalization layer applied on embedding from text_encoder and after potential
        # projection layer. If None, no normalization will be applied.
        output_norm: Optional[BaseLayer.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.hidden_dim is not None:
            hidden_dim = cfg.hidden_dim
        else:
            hidden_dim = cfg.output_dim

        self._add_child("text_encoder", cfg.text_encoder.set(output_dim=hidden_dim))

        if hidden_dim != cfg.output_dim:
            assert (
                cfg.output_proj is not None
            ), "output_proj can't be None when hidden_dim != output_dim."

        if cfg.output_proj is not None:
            self._add_child(
                "output_proj", cfg.output_proj.set(input_dim=hidden_dim, output_dim=cfg.output_dim)
            )
        if cfg.output_norm is not None:
            self._add_child("output_norm", cfg.output_norm)

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function.

        Args:
            input_batch: A dictionary containing all values of cfg.encoding_field_map as keys.
                Each value of input_batch is a Tensor with shape
                [batch_size, num_inputs_per_example, max_seq_len].

        Returns:
            An updated input_batch dict where all keys of cfg.encoding_field_map are added. Each
            value of the new key is a Tensor with shape
            [batch_size, num_inputs_per_example, output_dim].
        """

        def encode(input_ids: Tensor) -> NestedTensor:
            embeddings = self.text_encoder(input_ids)[TEXT_EMBEDDINGS]
            if "output_proj" in self.children:
                embeddings = self.output_proj(embeddings)
            if "output_norm" in self.children:
                embeddings = self.output_norm(embeddings)
            return embeddings

        cfg = self.config

        for output_emb_field, input_id_field in cfg.encoding_field_map.items():
            if input_id_field in input_batch and input_batch[input_id_field].shape[1] > 0:
                input_ids = input_batch[input_id_field]
                batch_size, num_inputs_per_example, max_seq_len = input_ids.shape
                # Shape: [batch_size * num_inputs_per_example, max_seq_len].
                flattened_input_ids = input_ids.reshape(-1, max_seq_len)

                with child_context(f"encode_{input_id_field}", module=self):
                    embeddings = encode(flattened_input_ids)
                # Shape: [batch_size * num_inputs_per_example, output_dim].
                embeddings = embeddings.squeeze(axis=1)

                input_batch[output_emb_field] = embeddings.reshape(
                    batch_size, num_inputs_per_example, -1
                )

        return input_batch


def flatten_and_concat_embeddings(
    *,
    left_positive_embeddings: Tensor,
    right_positive_embeddings: Tensor,
    right_positive_paddings: Tensor,
    right_negative_embeddings: Optional[Tensor] = None,
    right_negative_paddings: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Flattens left and right embeddings and concatenates right encoder positive and negative
    embeddings.

    Args:
        left_positive_embeddings: A Tensor with shape [num_left_inputs, 1, dim].
        right_positive_embeddings: A Tensor with shape
            [num_left_inputs, max_right_positive_inputs, dim].
        right_positive_paddings: A 0/1 Tensor with shape
            [num_left_inputs, max_right_positive_inputs] where 1 means padded docs and 0 means
            effective docs.
        right_negative_embeddings: An optional Tensor with shape
            [num_left_inputs, max_right_negative_inputs, dim].
        right_negative_paddings: An optional 0/1 Tensor with shape
            [num_left_inputs, max_right_negative_inputs] where 1 means padded docs and
            0 means effective docs.

    Returns:
        A dict of Tensor:
            FLATTENED_LEFT_EMBEDDINGS: A Tensor with shape [num_left_inputs, dim].
            FLATTENED_RIGHT_EMBEDDINGS: A Tensor with shape
                [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs), dim].
                max_right_negative_inputs = 0 when there is no right_negative_embeddings.
            RIGHT_PADDINGS: A Tensor with shape
                [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs)].
                max_right_negative_inputs = 0 when there is no right_negative_embeddings.
    """
    embedding_dim = left_positive_embeddings.shape[-1]
    assert (
        left_positive_embeddings.shape[1] == 1
    ), "Expecting one positive embedding per example from left encoder."
    # Shape: [num_left_inputs, dim].
    flattened_left_embeddings = left_positive_embeddings.reshape(-1, embedding_dim)
    assert (
        right_positive_embeddings.shape[-1] == embedding_dim
    ), "right_positive_embeddings has a different dim than that of left_embeddings!"
    # Shape: [num_left_inputs * max_right_positive_inputs, dim].
    flattened_right_positive_embeddings = right_positive_embeddings.reshape(-1, embedding_dim)
    if right_negative_embeddings is not None:
        assert (
            right_negative_embeddings.shape[-1] == embedding_dim
        ), "right_negative_embeddings has a different dim than that of left_embeddings!"
        # Shape: [num_left_inputs * max_right_negative_inputs, dim].
        flattened_right_negative_embeddings = right_negative_embeddings.reshape(-1, embedding_dim)
        # Shape: [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs), dim].
        flattened_right_embeddings = jnp.concatenate(
            [flattened_right_positive_embeddings, flattened_right_negative_embeddings], axis=0
        )
        # Shape: [num_left_inputs * max_right_positive_inputs].
        flattened_right_positive_paddings = jnp.reshape(right_positive_paddings, -1)
        # Shape: [num_left_inputs * max_right_negative_inputs].
        flattened_right_negative_paddings = jnp.reshape(right_negative_paddings, -1)
        # Shape: [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs)].
        right_paddings = jnp.concatenate(
            [flattened_right_positive_paddings, flattened_right_negative_paddings]
        )
    else:
        # Shape: [num_left_inputs * max_right_positive_inputs, dim].
        flattened_right_embeddings = flattened_right_positive_embeddings
        # Shape: [num_left_inputs * max_right_positive_inputs].
        right_paddings = jnp.reshape(right_positive_paddings, -1)

    return {
        FLATTENED_LEFT_EMBEDDINGS: flattened_left_embeddings,
        FLATTENED_RIGHT_EMBEDDINGS: flattened_right_embeddings,
        RIGHT_PADDINGS: right_paddings,
    }


def flatten_and_concat_embeddings_from_input_batch(
    *,
    input_batch: NestedTensor,
    left_encoder_name: Union[str, Sequence[str]],
    right_encoder_name: Union[str, Sequence[str]],
    assert_one_positive_from_right_encoder: bool = True,
) -> NestedTensor:
    """Obtains embeddings from each encoder, flattening and concatenating them.

    Args:
        input_batch: A Nested Tensor contains all embeddings and paddings for each encoder.
        left_encoder_name: Left encoder name. A sequence path of names could be specified to get
            embeddings and paddings recursively.
        right_encoder_name: Right encoder name. A sequence path of names could be specified to get
            embeddings and paddings recursively.
        assert_one_positive_from_right_encoder: If True, assert whether there is only one positive
            embedding from right encoder.

    Returns:
        A dict of Tensor:
            FLATTENED_LEFT_EMBEDDINGS: A Tensor with shape [num_left_inputs, dim].
            FLATTENED_RIGHT_EMBEDDINGS: A Tensor with shape
                [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs), dim].
                max_right_negative_inputs = 0 when there is no right_negative_embeddings.
            RIGHT_PADDINGS: A Tensor with shape
                [num_left_inputs * (max_right_positive_inputs + max_right_negative_inputs)].
                max_right_negative_inputs = 0 when there is no right_negative_embeddings.
    """
    left_encoder_emb = get_recursively(input_batch, left_encoder_name)
    right_encoder_emb = get_recursively(input_batch, right_encoder_name)

    right_positive_embeddings = right_encoder_emb[POSITIVE_EMBEDDINGS]
    right_positive_paddings = right_encoder_emb[POSITIVE_PADDINGS]
    if assert_one_positive_from_right_encoder:
        assert (
            right_positive_embeddings.shape[1] == 1
        ), "Expecting one positive embedding per example from right encoder."
        assert (
            right_positive_paddings.shape[1] == 1
        ), "Expecting one positive embedding per example from right encoder."

    return flatten_and_concat_embeddings(
        left_positive_embeddings=left_encoder_emb[POSITIVE_EMBEDDINGS],
        right_positive_embeddings=right_positive_embeddings,
        right_positive_paddings=right_positive_paddings,
        right_negative_embeddings=right_encoder_emb.get(NEGATIVE_EMBEDDINGS, None),
        right_negative_paddings=right_encoder_emb.get(NEGATIVE_PADDINGS, None),
    )


class TextEmbeddingAsymmetricContrastiveLossLayer(FusionNetwork):
    """A FusionNetwork that computes asymmetric contrastive loss using text embeddings from
    left and right encoders.

    Asymmetric contrastive loss means the softmax cross-entropy loss will only be
    calculated for each query among all candidate keys, but not vice versa.

    This loss layer expects left encoder contributing one embedding per example as queries. Right
    encoder contributes one positive embedding and optionally some number of negative embeddings per
    example as keys.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures TextEmbeddingAsymmetricContrastiveLossLayer."""

        # Name of left encoder that gives embeddings as queries when computing asymmetric
        # contrastive loss. Could be a sequence path.
        left_encoder_name: Required[Union[str, Sequence[str]]] = REQUIRED
        # Name of right encoder that gives embeddings as keys when computing asymmetric
        # contrastive loss. Could be a sequence path.
        right_encoder_name: Required[Union[str, Sequence[str]]] = REQUIRED
        # A positive scalar float to be multiplied with logits. Default is 1.0.
        contrastive_loss_scale_factor: float = 1.0

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function.

        Args:
            input_batch: A dictionary containing:
                cfg.left_encoder_name (can be a sequence path):
                    POSITIVE_EMBEDDINGS: A Tensor with shape [batch_size, 1, dim].
                cfg.right_encoder_name (can be a sequence path):
                    POSITIVE_EMBEDDINGS: A Tensor with shape [batch_size, 1, dim].
                    POSITIVE_PADDINGS: A 0/1 Tensor with shape [batch_size, 1] where 1 means padded
                        inputs and 0 means effective inputs.
                    NEGATIVE_EMBEDDINGS: A Tensor with shape
                        [batch_size, num_negative_inputs_per_example, dim].
                    NEGATIVE_PADDINGS: A 0/1 Tensor with shape
                        [batch_size, num_negative_inputs_per_example] where 1 means padded inputs
                        and 0 means effective inputs.

        Returns:
            loss: A Tensor representing the loss.
            A dictionary containing:
                SIMILARITY_MATRIX: A Tensor representing the similarity between left encoder
                    embeddings and right encoder embeddings.
        """
        cfg = self.config

        inputs = flatten_and_concat_embeddings_from_input_batch(
            input_batch=input_batch,
            left_encoder_name=cfg.left_encoder_name,
            right_encoder_name=cfg.right_encoder_name,
        )

        similarity = contrastive_logits(
            inputs[FLATTENED_LEFT_EMBEDDINGS], inputs[FLATTENED_RIGHT_EMBEDDINGS]
        )
        contrastive_loss = asymmetric_contrastive_loss_from_logits(
            similarity,
            key_paddings=inputs[RIGHT_PADDINGS],
            temperature=1 / cfg.contrastive_loss_scale_factor,
        )

        return contrastive_loss, {SIMILARITY_MATRIX: similarity}


class RankingPairwiseLossLayer(FusionNetwork):
    """A FusionNetwork to compute pairwise loss among right encoder pairwise loss candidates based
    on relative ranks.

    The pairwise loss is defined as the binary cross-entropy loss between all possible candidates
    pair for each query. With this loss, model is trained to give higher score to candidate having
    higher rank. For example, (d1, d2, d3) having a rank of (1, 2, 3) could give three pairs for
    learning: (d1, d2), (d1, d3), (d2, d3), where we force model to give higher score to the former
    doc.
    """

    @config_class
    class Config(BaseLayer.Config):
        # Name of left encoder that gives embeddings as queries when computing pairwise loss.
        left_encoder_name: Required[str] = REQUIRED
        # Name of right encoder that gives embeddings as keys when computing pairwise loss.
        right_encoder_name: Required[str] = REQUIRED
        # A positive scalar float to be multiplied with logits. Default is 1.0.
        pairwise_loss_scale_factor: float = 1.0

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            input_batch: A dictionary containing:
                LEFT_ENCODER_NAME:
                    POSITIVE_EMBEDDINGS: A Tensor with shape [batch_size, 1, dim].
                RIGHT_ENCODER_NAME:
                    PAIRWISE_LOSS_EMBEDDINGS: A Tensor with shape
                        [batch_size, num_pairwise_loss_inputs_per_examples, dim].
                    PAIRWISE_LOSS_PADDINGS: A 0/1 Tensor with shape
                        [batch_size, num_pairwise_loss_inputs_per_examples] where 1 means padded
                        inputs and 0 means valid inputs.
                    RANKS: An int Tensor with shape
                        [batch_size, num_pairwise_loss_inputs_per_examples] which records ranks of
                        each candidate. Padded candidate will have a rank of 0.

        Returns:
            loss: A scalar Tensor representing the pairwise loss.
            A dictionary:
                NUM_VALID_RANKING_PAIRS: A scalar Tensor indicating the number of valid ranking
                    pairs to calculate pairwise loss.
        """
        cfg = self.config

        left_embeddings = input_batch[cfg.left_encoder_name][POSITIVE_EMBEDDINGS]
        assert (
            left_embeddings.shape[1] == 1
        ), "Expecting one positive embedding per example from left encoder."

        right_embeddings = input_batch[cfg.right_encoder_name][PAIRWISE_LOSS_EMBEDDINGS]
        right_paddings = input_batch[cfg.right_encoder_name][PAIRWISE_LOSS_PADDINGS]

        ranks = input_batch[cfg.right_encoder_name][RANKS]
        num_queries = left_embeddings.shape[0]

        # Shape: [batch_size, 1, num_pairwise_loss_inputs_per_examples].
        logits = jnp.einsum("b i d, b j d -> b i j", left_embeddings, right_embeddings)
        logits = logits * cfg.pairwise_loss_scale_factor
        # Shape: [batch_size, num_pairwise_loss_inputs_per_examples].
        logits = jnp.squeeze(logits, axis=1)

        # Shape: [batch_size, num_pairwise_loss_inputs_per_examples].
        ranks = ranks * (1 - right_paddings)
        loss, num_valid_pairs = ranking_pairwise_loss(
            logits=logits, ranks=ranks, loss_scale=jnp.ones(num_queries)
        )
        return loss, {NUM_VALID_RANKING_PAIRS: num_valid_pairs}


class FLOPsLossLayer(FusionNetwork):
    """A FusionNetwork to calculate the FLOPs loss."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures FLOPsLossLayer."""

        # Name of left encoder. Could be a sequence path.
        left_encoder_name: Required[Union[str, Sequence[str]]] = REQUIRED
        # Name of right encoder. Could be a sequence path.
        right_encoder_name: Required[Union[str, Sequence[str]]] = REQUIRED
        # A schedule to dynamically adjust the weight of FLOPs loss.
        flops_weight_schedule: Required[InstantiableConfig[ScheduleFn]] = REQUIRED
        # Constant weight of left encoder's flops loss, on top of which the weight schedule will
        # be applied.
        left_encoder_flops_loss_weight: float = 1.0
        # Constant weight of right encoder's flops loss, on top of which the weight schedule will
        # be applied.
        right_encoder_flops_loss_weight: float = 1.0
        # Embedding elements that are no greater than this threshold will be count as sparse.
        # The average number of sparse elements per query will be reported in summaries.
        sparsity_threshold: float = 0.0

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._flops_weight_schedule = as_schedule_fn(cfg.flops_weight_schedule)

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        param_specs = {
            "step": ParameterSpec(
                shape=[],
                dtype=jnp.int32,
                mesh_axes=None,
                initializer=constant_initializer(0),
                weight_decay_scale=0,
            )
        }
        return param_specs

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function.

        Args:
            input_batch: A dictionary containing:
                cfg.left_encoder_name (can be a sequence path):
                    POSITIVE_EMBEDDINGS: A Tensor with shape
                        [batch_size, num_left_positive_inputs, dim].
                cfg.right_encoder_name (can be a sequence path):
                    POSITIVE_EMBEDDINGS: A Tensor with shape
                        [batch_size, num_right_positive_inputs, dim].
                    POSITIVE_PADDINGS: A 0/1 Tensor with shape
                        [batch_size, num_left_positive_inputs] where 1 means padded inputs and 0
                        means effective inputs.
                    NEGATIVE_EMBEDDINGS: A Tensor with shape
                        [batch_size, num_right_negative_inputs, dim].
                    NEGATIVE_PADDINGS: A 0/1 Tensor with shape
                        [batch_size, num_right_negative_inputs] where 1 means padded inputs and 0
                        means effective inputs.

        Returns:
            loss: A Tensor representing the loss.
        """
        cfg = self.config

        inputs = flatten_and_concat_embeddings_from_input_batch(
            input_batch=input_batch,
            left_encoder_name=cfg.left_encoder_name,
            right_encoder_name=cfg.right_encoder_name,
            assert_one_positive_from_right_encoder=False,
        )
        left_encoder_flops_loss, left_encoder_avg_sparsity_count = flops_loss(
            embeddings=inputs[FLATTENED_LEFT_EMBEDDINGS],
            sparsity_threshold=cfg.sparsity_threshold,
        )
        right_encoder_flops_loss, right_encoder_avg_sparsity_count = flops_loss(
            embeddings=inputs[FLATTENED_RIGHT_EMBEDDINGS],
            paddings=inputs[RIGHT_PADDINGS],
            sparsity_threshold=cfg.sparsity_threshold,
        )
        self.add_summary(f"{cfg.left_encoder_name}_flops_loss", left_encoder_flops_loss)
        self.add_summary(f"{cfg.right_encoder_name}_flops_loss", right_encoder_flops_loss)
        self.add_summary(
            f"{cfg.left_encoder_name}_avg_sparsity_count", left_encoder_avg_sparsity_count
        )
        self.add_summary(
            f"{cfg.right_encoder_name}_avg_sparsity_count", right_encoder_avg_sparsity_count
        )
        scheduled_weight = self._flops_weight_schedule(self.parameters["step"])
        loss = (
            cfg.left_encoder_flops_loss_weight * left_encoder_flops_loss * scheduled_weight
            + cfg.right_encoder_flops_loss_weight * right_encoder_flops_loss * scheduled_weight
        )
        self.add_state_update("step", self.parameters["step"] + 1)
        return loss, {}


class TextEmbeddingDualEncoder(MultiStreamModel):
    """A basic dual-encoder model for text embedding based applications.

    This class inherits from MultiStreamModel as a two-stream MultiStreamModel.
    Each stream's encoder is expected to be a TextEmbeddingStreamEncoder instance.
    """

    @config_class
    class Config(MultiStreamModel.Config):
        # The name of encoder to be shared in the Siamese encoder case.
        shared_encoder_name: Optional[str] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        if cfg.shared_encoder_name is not None:
            self._share_with_descendants(
                self._stream_encoder[cfg.shared_encoder_name],
                shared_module_name=TEXT_DUAL_ENCODER_SHARED_MODULE_NAME,
            )

    def _forward_all_stream_encoder(self, input_batch: NestedTensor) -> NestedTensor:
        """Calls the forward function of all the stream encoders."""
        for encoder_name in self._stream_encoder:  # pylint: disable=consider-using-dict-items
            input_batch[encoder_name] = self._stream_encoder[encoder_name](
                input_batch[encoder_name]
            )

        return input_batch

    def forward_single_stream_encoder(
        self, input_batch: NestedTensor, encoder_name: str
    ) -> NestedTensor:
        """Calls the forward function of a specific stream encoders."""
        if encoder_name not in self._stream_encoder:
            raise ValueError(f"{encoder_name} has not been found in stream_encoder.")
        return self._stream_encoder[encoder_name](input_batch[encoder_name])
