# Copyright © 2025 Apple Inc.

"""Layers for computing training time metrics."""

import re
from typing import Any, Optional

import jax
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.embedding import ModalityVocabInfo
from axlearn.common.metrics import MetricSummary, WeightedSummary
from axlearn.common.utils import (
    Nested,
    Tensor,
    flatten_items,
    get_recursively,
    set_recursively,
    validate_contains_paths,
)


class BaseLossMetrics(BaseLayer):
    """A module for computing training time metrics.

    See `causal_lm.Model` for an example usage.
    """

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[WeightedSummary, dict[str, MetricSummary | Tensor]]:
        """Computes metrics from inputs and predictions.

        Args:
            input_batch: A mapping from input keys to Tensors.
            predict_outputs: Model predictions for computing metrics.
            module_outputs: Outputs from the model's invocation context.

        Returns:
            A tuple (loss, metrics).
                loss: A WeightedSummary loss. Callers should call loss.value() for gradient.
                metrics: A dict containing auxiliary losses and metrics.
        """
        raise NotImplementedError(type(self))


def filter_module_outputs(
    module_outputs: Nested[Tensor], *, path_regex: str, default: Any = REQUIRED
) -> Nested[Tensor]:
    """Retrieves the leaf value(s) corresponding to `path_regex` from `module_outputs`.

    If `path_regex` matches multiple subpaths, the subtree will be returned.

    Raises if no paths match and no default is specified.
    """
    path_regex = re.compile(path_regex)
    matched_subpaths = {}
    for path, value in flatten_items(module_outputs):
        if m := re.fullmatch(path_regex, path):
            match_groups = m.groups()
            if len(match_groups) != 1:
                raise ValueError(
                    f"{path_regex=} should define exactly one matching group; "
                    f"instead, found: {match_groups}"
                )
            path = match_groups[0]
            try:
                get_recursively(matched_subpaths, path=path)
                raise ValueError(f"Multiple paths matched {path=}.")
            except KeyError:
                set_recursively(matched_subpaths, path=path, value=value)
    if not matched_subpaths:
        if default is REQUIRED:
            raise ValueError(
                f"No paths matched '{path_regex}': {jax.tree_util.tree_structure(module_outputs)}"
            )
        matched_subpaths = default
    return matched_subpaths


class ModalityLossMetrics(BaseLossMetrics):
    """Wraps a metrics implementation with per-modality masking.

    Specifically, it uses `cfg.modality_vocab_info` to construct "live_targets" prior to
    `forward()`, such that loss is only computed on tokens from the modality corresponding to
    `cfg.modality_vocab_info`.
    """

    @config_class
    class Config(BaseLossMetrics.Config):
        """Configures ModalityLossMetrics.

        Args:
            inner: The inner BaseLossMetrics to compute loss metrics.
            modality_vocab_info: An option ModalityVocabInfo that contains vocab.
            target_labels_pattern: The regex pattern to search for target_labels in
                module_outputs.
        """

        inner: Required[BaseLossMetrics.Config] = REQUIRED
        modality_vocab_info: Optional[ModalityVocabInfo] = None
        target_labels_pattern: str = "(?:.*/?)(target_labels)"

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        self._add_child("inner", cfg.inner)

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[WeightedSummary, dict[str, WeightedSummary | Tensor]]:
        """Computes loss and metrics.

        Args:
            input_batch: A nested Tensor of inputs to `model.forward()`.
                target_labels: A Tensor of shape [batch_size, seq_len]. It is assumed to contain -1
                    values for positions corresponding to padding (see e.g.
                    `causal_lm.Model._metrics`).
            predict_outputs: A nested Tensor of predictions from `model.predict()`. Forwarded to
                inner layer.
            module_outputs: Module outputs from `model.predict()`, containing at minimum:
                target_labels: Target labels (after replacing placeholder tokens with real tokens).

        Returns:
            A tuple (loss, metrics). Summaries and metrics will be namespaced under its name.
        """
        cfg: ModalityLossMetrics.Config = self.config

        # Original target_labels contains placeholders and -1 values for paddings.
        validate_contains_paths(input_batch, paths=["target_labels"])
        target_labels_with_placeholders = input_batch["target_labels"]
        assert target_labels_with_placeholders is not None

        # Gets vocab targets w.r.t. current modality.
        vocab_targets = jnp.greater_equal(target_labels_with_placeholders, 0)
        if cfg.modality_vocab_info is not None:
            vocab_targets = jnp.logical_and(
                vocab_targets,
                jnp.logical_and(
                    cfg.modality_vocab_info.placeholder_start <= target_labels_with_placeholders,
                    target_labels_with_placeholders < cfg.modality_vocab_info.placeholder_end,
                ),
            )
        # Note that the `live_targets` in the input_batch could be float.
        live_targets: Optional[Tensor] = input_batch.get("live_targets", None)
        if live_targets is not None:
            live_targets = live_targets.astype(jnp.float32) * vocab_targets.astype(jnp.float32)
        else:
            live_targets = vocab_targets

        # The target_labels in module_outputs will have placeholders replaced
        # by the embedding layer with real token ids.
        target_labels = filter_module_outputs(module_outputs, path_regex=cfg.target_labels_pattern)[
            "target_labels"
        ]
        assert target_labels is not None

        # Forwards processed input_patch to inner.
        inner_batch = {**input_batch, "target_labels": target_labels, "live_targets": live_targets}
        inner_outputs = self.inner.forward(
            input_batch=inner_batch, predict_outputs=predict_outputs, module_outputs=module_outputs
        )

        return inner_outputs
