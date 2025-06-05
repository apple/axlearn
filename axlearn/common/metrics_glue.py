# Copyright © 2023 Apple Inc.

"""GLUE metric calculators.

https://arxiv.org/abs/1804.07461
"""
from typing import Optional, Union

import jax
import jax.numpy as jnp

from axlearn.common.config import REQUIRED, Configurable, Required, config_class
from axlearn.common.evaler import ModelSummaryAccumulator
from axlearn.common.metrics import WeightedScalar
from axlearn.common.metrics_classification import f_score
from axlearn.common.metrics_correlation import (
    matthews_corrcoef,
    pearson_corrcoef,
    spearman_corrcoef,
)
from axlearn.common.utils import NestedTensor, Tensor


class GLUEMetricAccumulator(Configurable):
    """A GLUEMetricAccumulator is used during GLUE evaluation to accumulate metrics across batches.

    This differs from MetricAccumulator because we accumulate all preds and gt across batches
    before calculating metrics. This is because certain GLUE metrics (such as Spearman corrcoef)
    cannot be computed online. T5X also computes metrics in this manner as well by using the Seqio
    Evaluator.
    """

    def __init__(self, cfg: Configurable.Config, task: str):
        super().__init__(cfg)
        self._preds = []
        self._gt = []
        self._mask = []

        self._task = task

    def update(self, *, target_labels: Tensor, preds: Tensor, mask: Optional[Tensor] = None):
        if mask is None:
            mask = jnp.ones_like(target_labels)

        if not target_labels.ndim == preds.ndim == mask.ndim == 1:
            raise ValueError(
                f"Pred labels, gt labels, and mask should have exactly one dimension. "
                f"Got pred: {preds.ndim}, gt: {target_labels.ndim}, mask: {mask.ndim}."
            )

        if not target_labels.shape == preds.shape == mask.shape:
            raise ValueError(
                f"Pred labels, gt labels, and mask should have the same shape. "
                f"Got pred: {preds.shape}, gt: {target_labels.shape}, mask: {mask.shape}."
            )

        self._preds.append(preds)
        self._gt.append(target_labels)
        self._mask.append(mask)

    def summaries(self) -> dict[str, WeightedScalar]:
        preds = jnp.hstack(self._preds)
        gt = jnp.hstack(self._gt)
        mask = jnp.hstack(self._mask)

        return glue_metrics(task=self._task, target_labels=gt, preds=preds, mask=mask)


class GLUEMetricCalculator(ModelSummaryAccumulator):
    """Computes GLUE evaluation metrics."""

    @config_class
    class Config(ModelSummaryAccumulator.Config):
        task: Required[str] = REQUIRED
        ignore_target_label: Required[Union[int, jnp.float32]] = REQUIRED

    def _forward_in_pjit(
        self,
        model_params: NestedTensor,
        prng_key: Tensor,
        input_batch: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Calls `self._model` and returns summaries."""
        cfg = self.config
        next_key, forward_prng = jax.random.split(prng_key)
        model_outputs, model_output_collection = self._call_model(
            method=cfg.model_method,
            prng_key=forward_prng,
            model_params=model_params,
            input_batch=input_batch,
            **cfg.model_method_kwargs,
        )
        loss, aux = model_outputs
        logits: Tensor = aux["logits"]
        target_labels: Tensor = input_batch["target_labels"]
        live_targets = (target_labels != cfg.ignore_target_label).astype(logits.dtype)

        task = cfg.task
        if task == "stsb":
            preds = logits.squeeze(axis=-1)
        else:
            preds = logits.argmax(axis=-1)

        summaries = model_output_collection.summaries
        summaries.update(
            glue_metrics(
                task=task,
                preds=preds,
                target_labels=target_labels,
                mask=live_targets,
            )
        )

        num_targets = live_targets.sum()
        summaries["loss"] = WeightedScalar(loss, num_targets)
        return dict(
            replicated=dict(
                prng_key=next_key,
                summaries=summaries,
            ),
            per_example={},
        )


def glue_metrics(
    *,
    task: str,
    target_labels: Tensor,
    preds: Tensor,
    mask: Tensor,
) -> dict[str, WeightedScalar]:
    """Computes the corresponding metrics for tasks.

    Args:
        task: The name of the task to compute metrics for.
        target_labels: Tensor of shape [batch_size] and values [0, num_classes)
            representing ground truth.
        preds: Tensor of shape [batch_size] and values representing predictions.
            While predictions are typically in [0, num_classes), they are not guaranteed to be.
        mask: A bool tensor with the same shape of `target_labels`. True values are
            included in metrics calculation.

    Returns:
        A dict with metric names as keys and corresponding metric values as values.
        Each metric value is a WeightedScalar.
    """
    summaries_dict = {}
    num_targets = mask.sum()

    if task == "stsb":
        summaries_dict["pearson_corr"] = WeightedScalar(
            pearson_corrcoef(preds, target_labels, weight=mask), num_targets
        )
        summaries_dict["spearman_corr"] = WeightedScalar(
            spearman_corrcoef(preds, target_labels, mask=mask), num_targets
        )
    else:
        accuracy = jnp.sum(mask * (preds == target_labels)) / jnp.maximum(num_targets, 1)
        summaries_dict["accuracy"] = WeightedScalar(accuracy, num_targets)

    # Add additional metrics for some tasks.
    if task == "cola":
        # Compute metrics on the normalized values.
        summaries_dict["matthews_corr"] = WeightedScalar(
            matthews_corrcoef(
                target_labels,
                preds,
                weight=mask,
            ),
            num_targets,
        )
    elif task in {"mrpc", "qqp", "record"}:
        # Map out-of-class predictions to the wrong label (to mark them as incorrect).
        invalid_mask = jnp.logical_and(preds != 0, preds != 1)
        preds = jnp.where(invalid_mask, 1 - target_labels, preds)

        summaries_dict["f1"] = WeightedScalar(
            f_score(target_labels, preds, beta=1, weight=mask), num_targets
        )
    return summaries_dict
