# Copyright © 2023 Apple Inc.

"""Autoregressive decoder model, e.g. as seen in the GPT family."""

import math
import re
from typing import Callable, Optional, Union

import jax
from absl import logging
from jax import numpy as jnp
from jax._src.mesh import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common.attention import (
    LearnedPositionalEmbedding,
    PipelinedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class
from axlearn.common.decoder import Decoder
from axlearn.common.decoding import (
    BeamSearchOutputs,
    SampleOutputs,
    StopDecodingCondition,
    brevity_penalty_fn,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import LayerNorm
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.loss import cross_entropy
from axlearn.common.loss_metrics import BaseLossMetrics
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, NestedTensor, Tensor, child_context
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.utils import (
    Nested,
    flatten_items,
    validate_contains_paths,
    with_sharding_constraint,
)


def layer_norm_config(eps=1e-5):
    return LayerNorm.default_config().set(eps=eps)


# TODO(markblee): Move these to `axlearn.common.loss_metrics` and update golden configs.
class CrossEntropyLossMetrics(BaseLossMetrics):
    """Computes cross entropy loss and related training summaries."""

    @config_class
    class Config(BaseLossMetrics.Config):
        """Configures CrossEntropyLossMetrics.

        Attributes:
            z_loss_scale: An auxiliary z-loss scale. If not None and >0, encourages the softmax
                normalizer to be well behaved.
        """

        z_loss_scale: Optional[float] = None

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Computes cross entropy loss.

        Args:
            input_batch: A dict containing at minimum:
                target_labels: An int Tensor of shape [...]. Negative targets do not contribute to
                    the loss calculation.
            predict_outputs: A dict containing at minimum:
                logits: A float Tensor of shape [..., num_classes].
            module_outputs: Unused.

        Returns:
            A tuple (loss, metrics):
                loss: A scalar float Tensor corresponding to cross entropy loss, including auxiliary
                    z-loss if `cfg.z_loss_scale` is provided.
                metrics: A dict containing:
                    cross_entropy: Same as loss.
                    per_token_loss: A float Tensor of same shape as `target_labels`. Ignored targets
                        will be masked, i.e., have per-token loss of 0.
                    live_targets: A bool Tensor of same shape as `target_labels`. False indicates
                        ignored targets.
                    num_targets: A scalar int Tensor corresponding to number of live targets.
        """
        del module_outputs
        validate_contains_paths(input_batch, paths=["target_labels"])
        validate_contains_paths(predict_outputs, paths=["logits"])

        cfg: CrossEntropyLossMetrics.Config = self.config

        target_labels: Tensor = input_batch["target_labels"]
        target_num_bytes: Optional[Tensor] = input_batch.get("target_num_bytes")
        live_targets: Optional[Tensor] = input_batch.get("live_targets")
        logits = predict_outputs["logits"]

        if live_targets is None:
            live_targets = target_labels >= 0
        num_targets = live_targets.sum()
        accuracy = (
            jnp.equal(jnp.argmax(logits, axis=-1), target_labels) * live_targets
        ).sum() / jnp.maximum(1, num_targets)

        loss, loss_dict = cross_entropy(
            logits=logits,
            target_labels=target_labels,
            live_targets=live_targets,
            z_loss_scale=cfg.z_loss_scale if cfg.z_loss_scale is not None else 0.0,
        )
        per_token_loss = loss_dict["per_target_loss"] * live_targets
        self.add_summary("accuracy", WeightedScalar(accuracy, num_targets))
        self.add_summary("z_loss", WeightedScalar(loss_dict["z_loss"], num_targets))
        if target_num_bytes is not None:
            # N.B. we calculate bpb following Appendix D.2. of <https://arxiv.org/abs/2112.11446>,
            # (i.e. treat each token as an equal with the others in the batch).
            # This is also consistent with how we calculate the other metrics.
            total_bytes = target_num_bytes.sum()
            bits_per_byte = per_token_loss.sum() / jnp.maximum(1, total_bytes) / jnp.log(2)
            self.add_summary("bits_per_byte", WeightedScalar(bits_per_byte, total_bytes))
        metrics = {
            "cross_entropy": loss,
            "per_token_loss": per_token_loss,
            "live_targets": live_targets,
            "num_targets": num_targets,
        }
        self.add_summary("cross_entropy_loss", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))
        self.add_summary("loss", WeightedScalar(loss, num_targets))
        self.add_summary(
            "train_live_targets",
            WeightedScalar(num_targets / target_labels.shape[0], target_labels.shape[0]),
        )
        return loss, metrics


class AuxLossMetrics(BaseLossMetrics):
    """Computes aux loss by aggregating across layers.

    Aux loss metrics added via `add_module_output` are aggregated and returned.
    """

    @config_class
    class Config(BaseLossMetrics.Config):
        """Configures AuxLossMetrics.

        Attributes:
            aux_loss_regex: If not None, collect Tensors from `module_outputs` whose paths fully
                match the regular expression and compute the sum as the auxiliary loss, which will
                be added to the overall model loss and reported in the summary as `aux_loss`.
                This can be used to support regularization losses such as the load balancing loss in
                MoE routing.
        """

        aux_loss_regex: Optional[str] = None

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Computes aux loss by aggregating module outputs from all layers.

        Args:
            input_batch: A dict containing at minimum:
                target_labels: An int Tensor of any shape. Negative targets do not contribute to
                    the loss calculation.
            predict_outputs: Unused.
            module_outputs: A nested Tensor consisting of outputs added via `add_module_output`.
                Paths within `module_outputs` will be full-matched against `aux_loss_regex`.

        Returns:
            A tuple (loss, metrics):
                loss: A scalar float Tensor corresponding to the aux loss.
                metrics: A dict containing:
                    aux_loss: Same as loss.
        """
        del predict_outputs

        cfg: AuxLossMetrics.Config = self.config
        regex = cfg.aux_loss_regex

        if regex is None:
            return 0.0, {}

        validate_contains_paths(input_batch, paths=["target_labels"])
        target_labels: Tensor = input_batch["target_labels"]
        live_targets = target_labels >= 0
        num_targets = live_targets.sum()

        logging.info("Module outputs: %s", jax.tree_structure(module_outputs))
        accumulation = []
        for k, v in flatten_items(module_outputs):
            if re.fullmatch(regex, k):
                logging.info("Aux loss found at %s", k)
                accumulation.append(v.mean())
            else:
                logging.info("Aux loss not found at %s", k)

        if accumulation:
            aux_loss = sum(accumulation) / len(accumulation)
        else:
            logging.warning("Aux loss not found: %s", cfg.aux_loss_regex)
            aux_loss = 0.0

        self.add_summary("aux_loss", WeightedScalar(aux_loss, num_targets))
        return aux_loss, {"aux_loss": aux_loss}


def _update(x: dict, updates: dict):
    """Equivalent to `x.update(updates)` but raises upon key conflicts."""
    if not x.keys().isdisjoint(updates.keys()):
        raise KeyError(f"Key conflict: {set(x.keys()).intersection(updates)}")
    x.update(updates)


class CompositeLossWeights(Module):
    """Computes loss weights."""

    def forward(self, child_metrics: dict[str, tuple[Tensor, Nested[Tensor]]]) -> dict[str, Tensor]:
        """Computes per-child loss weights from child metrics.

        Args:
            child_metrics: A mapping from child name to (child_loss, child_metrics).

        Returns:
            A mapping from child name to loss weight.
        """
        raise NotImplementedError(type(self))


class CompositeLossMetrics(BaseLossMetrics):
    """Computes a composite loss from multiple child metrics."""

    @config_class
    class Config(BaseLossMetrics.Config):
        """Configures CompositeLossMetrics.

        Attributes:
            metrics: A mapping from child name to metrics config.
            loss_weights: A `CompositeLossWeights` implementation.
                If None, all weights are considered 1.
            flatten_metrics: Whether to flatten summaries and metrics from each child. If None,
                defaults to True.
        """

        metrics: Required[dict[str, BaseLossMetrics.Config]] = REQUIRED
        loss_weights: Optional[CompositeLossWeights.Config] = None
        flatten_metrics: Optional[bool] = None

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        cfg: CompositeLossMetrics.Config = self.config
        self._metrics: dict[str, BaseLossMetrics] = {}
        for name, child in cfg.metrics.items():
            self._metrics[name] = self._add_child(name, child)
        if cfg.loss_weights is not None:
            self.loss_weights: CompositeLossMetrics = self._add_child(
                "loss_weights", cfg.loss_weights
            )

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Combines losses and metrics from the configured children.

        By default, losses are summed and metrics/summaries are flattened, raising if any keys
        conflict.
        """
        cfg: CompositeLossMetrics.Config = self.config
        all_child_metrics = {}

        for name, child in self._metrics.items():
            all_child_metrics[name] = child.forward(
                input_batch=input_batch,
                predict_outputs=predict_outputs,
                module_outputs=module_outputs,
            )

        if "loss_weights" in self.children:
            loss_weights: dict[str, Tensor] = self.loss_weights(all_child_metrics)
        else:
            loss_weights = None

        loss = 0
        metrics = {}
        for name, (child_loss, child_metrics) in all_child_metrics.items():
            if loss_weights is not None and loss_weights.get(name, None) is not None:
                child_loss *= loss_weights[name]
            loss = loss + child_loss

            ctx = self.get_invocation_context()

            if cfg.flatten_metrics is False:
                _update(metrics, {name: child_metrics})
            else:
                _update(ctx.output_collection.summaries, ctx.output_collection.summaries.pop(name))
                _update(metrics, child_metrics)

        return loss, metrics


def metrics_config(
    *,
    z_loss_scale: Optional[float] = None,
    aux_loss_regex: Optional[str] = None,
) -> CompositeLossMetrics.Config:
    """Constructs a default causal-lm metrics config.

    Args:
        z_loss_scale: Auxiliary z-loss scale. See `CrossEntropyLossMetrics.Config`.
        aux_loss_regex: Aux loss regex. See `AuxLossMetrics.Config`.

    Returns:
        A composite of cross entropy and aux loss.
    """
    return CompositeLossMetrics.default_config().set(
        metrics={
            "lm": CrossEntropyLossMetrics.default_config().set(z_loss_scale=z_loss_scale),
            "aux": AuxLossMetrics.default_config().set(aux_loss_regex=aux_loss_regex),
        }
    )


class Model(BaseModel):
    """Autoregressive decoder-only transformer sequence model."""

    @config_class
    class Config(BaseModel.Config):
        """Configuration for a causal-lm."""

        # Decoder.
        decoder: Decoder.Config = Decoder.default_config()
        # TODO(markblee): Remove `batch_axis_names` and `seq_axis_names`. Input sharding should
        # happen at `Input.dispatch_global_batch` instead.
        # Batch mesh axis name(s). (Deprecated.)
        # These will be used to constrain the batch (first) axis of relevant inputs.
        # If None, no batch dim constraints are applied, rather than replicating across batch dim.
        batch_axis_names: Optional[tuple[str]] = ("data",)
        # Sequence-parallel mesh axis name(s). (Deprecated.)
        # These will be used to constrain the sequence axis of relevant inputs.
        # If None, no batch sequence dim constraints are applied.
        seq_axis_names: Optional[tuple[str]] = None
        # Configures training metrics.
        metrics: BaseLossMetrics.Config = metrics_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("decoder", cfg.decoder)
        self._add_child("metrics", cfg.metrics)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=0.02, distribution="normal"
                )
            }
        )
        return cfg

    # We drop the kwargs from BaseModel, since they aren't used here.
    # pylint: disable-next=arguments-differ
    def forward(
        self,
        input_batch: NestedTensor,
        return_aux: bool = False,
    ) -> tuple[Tensor, NestedTensor]:
        """Produce decoder-only loss and predictions including logits and decoder hidden states in
        auxiliary outputs.

        Args:
            input_batch: a dict with the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len].
                    Values should be in the range [0, type_vocab_size].
                target_labels: an optional int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].
                target_num_bytes: an optional int Tensor of shape [batch_size].
                    Used to provide the number of UTF-8 bytes represented by the target_labels.
                input_segment_ids: an optional int Tensor of shape [batch_size, seq_len] with
                    unique positive values for different input sequences.
                input_positions: An optional int Tensor of shape [batch_size, target_len] with
                    non-negative values representing token position indices.
            return_aux: boolean to determine whether logits and decoder hidden states are returned.

        Returns:
            loss: a scalar float Tensor.
            aux_outputs (a dict):
                logits: a float Tensor of shape [batch_size, seq_len, vocab_size].
                hidden_states: a float Tensor of shape [batch_size, seq_len, hidden_dim].
                metrics: a nested Tensor. See corresponding `metrics` implementation for details.
        """
        predictions = self.predict(input_batch)
        aux_outputs = {**predictions}
        loss = None
        target_labels: Tensor = input_batch.get("target_labels")
        if target_labels is not None:
            loss, metrics = self._metrics(input_batch=input_batch, predict_outputs=predictions)
            aux_outputs.update(loss=loss, metrics=metrics)
        # If return_aux, return the logits and output pre LM head (useful for downstream tasks),
        # as well as training metrics like the per-token-loss.
        #
        # N.B. Do not enable for large-scale training since auxiliary outputs are not partitioned.
        # TODO(rpang): support partitioning of auxiliary outputs.
        return loss, aux_outputs if return_aux else {}

    def beam_search_decode(
        self,
        input_batch: NestedTensor,
        num_decodes: int = 1,
        brevity_penalty: Optional[Callable[[jnp.array, Tensor], jnp.array]] = brevity_penalty_fn(
            alpha=0.0
        ),
    ) -> BeamSearchOutputs:
        """Perform beam search decoding given prefix prompt.

        Args:
            input_batch: a dict with a minimum of the following entries:
                prefix: Prompt IDs representing a Tensor of shape [batch, max_sequence_length].
            num_decodes: the number of beams to decode.
            brevity_penalty: brevity penalty function to add length normalization
                in the beam search.

        Returns:
            Beam search outputs.
        """
        self._constrain_input_batch(input_batch)
        with child_context("beam_search_decode", module=self.decoder):
            prefix = input_batch["prefix"]
            return self.decoder.beam_search_decode(
                input_batch=input_batch,
                max_sequence_length=prefix.shape[-1],
                num_decodes=num_decodes,
                brevity_penalty=brevity_penalty,
            )

    def sample_decode(
        self,
        input_batch: NestedTensor,
        *,
        num_decodes: int = 1,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
        stop_decoding_condition: Optional[StopDecodingCondition] = None,
    ) -> SampleOutputs:
        """Perform sample decoding given prefix prompt.

        Args:
            input_batch: a dict with a minimum of the following entries:
                prefix: Prompt IDs representing a Tensor of shape [batch, max_sequence_length].
            num_decodes: the number of paths to decode.
            logits_modifier: Function or function-config applied to the logits before sampling.
                If None, do not modify the logits.
            stop_decoding_condition: StopDecodingCondition callable indicating if generation should
                stop. If None, stop on EOS.

        Returns:
            Sample outputs.
        """
        self._constrain_input_batch(input_batch)
        with child_context("sample_decode", module=self.decoder):
            prefix = input_batch["prefix"]
            return self.decoder.sample_decode(
                input_batch=input_batch,
                max_sequence_length=prefix.shape[-1],
                num_decodes=num_decodes,
                logits_modifier=logits_modifier,
                stop_decoding_condition=stop_decoding_condition,
            )

    def extract_logits(self, input_batch: NestedTensor) -> Tensor:
        """Obtains logits from the language model.

        Args:
            input_batch: A dict containing:
                input_ids: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].
                target_labels: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].

        Returns:
           A float Tensor of shape [batch_size, target_len, hidden_dim].
        """
        self._constrain_input_batch(input_batch)
        predictions = self.predict(input_batch)
        return predictions["logits"]

    def score(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Produce decoder score like per_token_loss and live_targets.

        Args:
            input_batch: A dict containing:
                input_ids: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].
                target_labels: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].

        Returns:
            A dict containing:
                - "per_token_loss": a float Tensor of shape [batch_size, seq_len]
                - "live_targets": a float Tensor of shape [batch_size, seq_len]
        """
        self._constrain_input_batch(input_batch)
        predictions = self.predict(input_batch)
        _, results = self._metrics(input_batch=input_batch, predict_outputs=predictions)
        return {k: v for k, v in results.items() if k in ("per_token_loss", "live_targets")}

    def predict(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Produce decoder logits and hidden states.

        Args:
            input_batch: a dict with the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len].
                    Used as decoder input ids. Values should be in the range [0, vocab_size].
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len].
                    Values should be in the range [0, type_vocab_size].
                input_segment_ids: an optional int Tensor of shape [batch_size, seq_len].
                    Denotes the segments within the sequence.
                input_positions: an optional int Tensor of shape [batch_size, seq_len].
                    Values should be in the range [0, seq_len].

        Returns:
            A dict containing:
                logits: a float Tensor of shape [batch_size, seq_len, vocab_size]
                hidden_states: a float Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        self._constrain_input_batch(input_batch)
        # TODO(markblee): Simplify by using consistent naming between `input_positions` and
        # `positions`, `input_segment_ids` and `segment_ids`.
        # Decoder hidden states: [batch_size, target_len, hidden_dim].
        decoder_batch = {**input_batch}
        decoder_batch["positions"] = input_batch.get("input_positions")
        return self.decoder(input_batch=decoder_batch)

    def _metrics(
        self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor]
    ) -> tuple[Tensor, Nested[Tensor]]:
        cfg: Model.Config = self.config
        target_labels: Tensor = input_batch["target_labels"]
        self.vlog(3, "targets=%s(%s)", target_labels.dtype, target_labels.shape)
        # Map padding targets to out-of-class label for metrics calculation.
        target_labels = jnp.where(target_labels == cfg.decoder.pad_token_id, -1, target_labels)

        ctx = self.get_invocation_context()
        loss, metrics = self.metrics.forward(
            input_batch={**input_batch, "target_labels": target_labels},
            predict_outputs=predict_outputs,
            module_outputs=ctx.get_module_outputs(),
        )
        # Flatten summaries for backwards compatibility.
        _update(ctx.output_collection.summaries, ctx.output_collection.summaries.pop("metrics"))

        return loss, metrics

    def _constrain_input_batch(self, input_batch: NestedTensor):
        """Applies sharding constraints in-place for relevant named tensors in the input batch."""
        mesh = thread_resources.env.physical_mesh  # type: ignore
        if mesh.empty or mesh.size == 1:
            return
        cfg: Model.Config = self.config
        if cfg.batch_axis_names is None and cfg.seq_axis_names is None:
            return

        logging.log_first_n(
            logging.WARNING,
            "cfg.batch_axis_names and cfg.seq_axis_names are deprecated. "
            "Dispatch inputs using `Input.dispatch_global_batch` instead. "
            "See `input_base.Input.input_partitioner` for more details.",
            1,
        )

        for k, v in input_batch.items():
            if k in [
                "input_ids",
                "target_labels",
                "token_type_ids",
                "prefix",
                "input_segment_ids",
                "input_positions",
            ]:
                assert v.ndim == 2
                input_batch[k] = with_sharding_constraint(
                    v, PartitionSpec(cfg.batch_axis_names, cfg.seq_axis_names)
                )
            elif k == "target_num_bytes":
                assert v.ndim == 1
                input_batch[k] = with_sharding_constraint(v, PartitionSpec(cfg.batch_axis_names))
            else:
                # We warn as not-constraining may be an oversight.
                logging.log_first_n(
                    logging.WARNING, "Not constraining input_batch[%s].", len(input_batch), k
                )


TransformerStackConfig = Union[
    StackedTransformerLayer.Config,
    RepeatedTransformerLayer.Config,
    PipelinedTransformerLayer.Config,
]


def residual_initializer_cfg(num_layers, scale=0.02):
    # GPT decoder: "Scale weights on residual path by 1/sqrt(num_layers)".
    scale = scale / math.sqrt(2 * num_layers)  # 2 x residuals per layer.
    init_cfg = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, distribution="normal", scale=scale
            )
        }
    )
    return init_cfg


def gpt_decoder_config(
    stack_cfg: TransformerStackConfig,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    max_position_embeddings: int,
    activation_function: str = "nn.relu",
    layer_norm_epsilon: float = 1e-08,
    dropout_rate: float = 0.0,
    layer_remat: Optional[RematSpec] = None,
) -> Decoder.Config:
    """Build a decoder transformer config in the style of GPT.

    Reference: https://github.com/openai/gpt-2.

    Args:
        stack_cfg: A config of StackedTransformerLayer, RepeatedTransformerLayer, or
            PipelinedTransformerLayer.
        num_layers: Number of transformer decoder layers.
        hidden_dim: Dimension of embeddings and input/output of each transformer layer.
        num_heads: Number of attention heads per transformer layer.
        vocab_size: Size of vocabulary.
        max_position_embeddings: Number of positional embeddings.
        activation_function: Type of activation function.
        layer_norm_epsilon: Epsilon for layer normalization. Defaults to LayerNorm.config.eps.
        dropout_rate: Dropout rate applied throughout model, including output_dropout.
        layer_remat: If not None, use as transformer.layer.remat_spec.

    Returns:
        A Decoder config.
    """
    stack_cfg = stack_cfg.clone()

    assert stack_cfg.klass in [
        StackedTransformerLayer,
        RepeatedTransformerLayer,
        PipelinedTransformerLayer,
    ]

    # TransformerLayer.
    layer_cfg = TransformerLayer.default_config()
    # Feed-forward transformer layer config.
    layer_cfg.feed_forward.activation = activation_function
    layer_cfg.feed_forward.norm = LayerNorm.default_config().set(eps=layer_norm_epsilon)
    layer_cfg.feed_forward.hidden_dim = 4 * hidden_dim
    # Self attention transformer layer config.
    layer_cfg.self_attention.norm = LayerNorm.default_config().set(eps=layer_norm_epsilon)
    layer_cfg.self_attention.attention.causal = True
    layer_cfg.self_attention.attention.num_heads = num_heads
    # Use residual initialization for output linear layer
    layer_cfg.self_attention.attention.output_linear.param_init = residual_initializer_cfg(
        num_layers=num_layers
    )
    layer_cfg.feed_forward.linear2.param_init = residual_initializer_cfg(num_layers=num_layers)
    layer_cfg.remat_spec = layer_remat
    transformer_cls = stack_cfg.set(num_layers=num_layers, layer=layer_cfg)
    decoder = Decoder.default_config().set(
        transformer=transformer_cls,
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=TransformerTextEmbeddings.default_config().set(
            pos_emb=LearnedPositionalEmbedding.default_config().set(
                shape=(max_position_embeddings,)
            )
        ),
        output_norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
        dropout_rate=dropout_rate,
        attention_mask=None,
    )
    return decoder
