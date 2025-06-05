# Copyright © 2023 Apple Inc.

"""Classification evaluation pipeline."""
from collections.abc import Sequence
from typing import NamedTuple, Optional

import jax
from jax import numpy as jnp

from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.evaler import GlobalMetricCalculator
from axlearn.common.metrics_classification import precision_recall_curve
from axlearn.common.utils import NestedTensor, Tensor


class PrecisionRecallMetricCalculator(GlobalMetricCalculator):
    """A metric calculator designed to calculate precision recall curve on the entire dataset."""

    @config_class
    class Config(GlobalMetricCalculator.Config):
        """Configures PrecisionRecallMetricCalculator."""

        # The name of the labels corresponding to the order of the outputs.
        label_names: Required[Sequence[str]] = REQUIRED
        # The name of the field corresponds to the target in input_batch.
        label_field: str = "target_labels"
        # The name of the field corresponds to the prediction in predict_outputs.
        prediction_field: str = "logits"
        # The precision levels we use to calculate recall, in the range (0, 1].
        precision_levels: Required[Sequence[float]] = REQUIRED
        # The recall levels we use to calculate precision, in the range (0, 1].
        recall_levels: Required[Sequence[float]] = REQUIRED
        # Boolean to indicate whether we need to convert label from integer to one hot setting.
        expand_label_to_onehot: Optional[bool] = False

    class Output(NamedTuple):
        # A NestedTensor contains the ground truth labels 0 or 1.
        # [batch_size, num_labels] or [batch_size]
        y_true: NestedTensor
        # A NestedTensor contains the prediction scores in the range (-inf, inf).
        # [batch_size, num_labels]
        y_score: NestedTensor

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Calls predict method of the model and returns the state, PredictionOutputs.

        Args:
            input_batch: The evaluation input batch.
            model_params: The model parameters.
            state: As returned by `init_state` or by the previous invocation of `forward`.

        Returns:
            A dict containing:
            - "state": A dict containing prng_key.
            - "output": A NamedTuple containing per-batch model outputs and corresponding targets.
        """
        outputs = super().forward(input_batch=input_batch, model_params=model_params, state=state)
        state = outputs["state"]
        output = outputs["output"]
        y_scores = output.predict_outputs[self.config.prediction_field]

        # We assume negative labels as padding example.
        return dict(
            state=state,
            output=self.Output(input_batch[self.config.label_field], y_scores),
        )

    def _calculate_metrics(self, outputs: Output) -> dict[str, Tensor]:
        """Calculates metrics from concatenated_outputs of the whole evaluation set.

           If no threshold can satisfy the target level, the precision / recall will be -1
           and the threshold will be inf.

        Args:
            outputs: A PredictionOutputs with input field name as key and a tensor of shape
                [num_examples, ...] representing the concatenated input across the whole evaluation
                set for metrics calculation.

        Returns:
            A dict containing all metrics for current task.
        """
        cfg = self.config
        # [num_examples, num_labels].
        labels = outputs.y_true
        if cfg.expand_label_to_onehot:
            labels = jax.nn.one_hot(labels, len(cfg.label_names))
        if len(labels.shape) == 1:
            labels = jnp.expand_dims(labels, axis=1)
        # [num_examples].
        is_valid_input = jnp.all(jnp.greater_equal(labels, 0), axis=1)
        # [num_examples, num_labels].
        scores = outputs.y_score

        all_metrics = {}
        for i in range(labels.shape[-1]):
            output = precision_recall_curve(labels[:, i], scores[:, i], weight=is_valid_input)
            precisions, recalls, thresholds = (
                output["precisions"],
                output["recalls"],
                output["thresholds"],
            )

            # Mask out unqualified precisions (with inf) and recall (with -1).
            # We use -1 to mask out recall so that user won't be confused with
            # recall actually being 0 vs. no recall can satisfy precision level.
            # [num_examples].
            for precision_level in cfg.precision_levels:
                qualified_cond = jnp.logical_and(
                    precisions >= precision_level, thresholds != jnp.finfo(jnp.float32).max
                )
                qualified_precisions = jnp.where(
                    qualified_cond, precisions, jnp.finfo(jnp.float32).max
                )
                qualified_recalls = jnp.where(qualified_cond, recalls, -1)
                idx_closest_to_threshold = jnp.argmin(qualified_precisions)

                all_metrics[
                    cfg.label_names[i] + f"_recall@p{precision_level * 100}"
                ] = qualified_recalls[idx_closest_to_threshold]
                all_metrics[
                    cfg.label_names[i] + f"_recall@p{precision_level * 100}_threshold"
                ] = thresholds[idx_closest_to_threshold]

            # Mask out unqualified recalls (with inf) and precision (with -1).
            # [num_examples].
            for recall_level in cfg.recall_levels:
                qualified_cond = jnp.logical_and(
                    recalls >= recall_level, thresholds != jnp.finfo(jnp.float32).max
                )
                qualified_recalls = jnp.where(qualified_cond, recalls, jnp.finfo(jnp.float32).max)
                qualified_precisions = jnp.where(qualified_cond, precisions, -1)
                idx_closest_to_threshold = jnp.argmin(qualified_recalls)

                all_metrics[
                    cfg.label_names[i] + f"_precision@r{recall_level * 100}"
                ] = qualified_precisions[idx_closest_to_threshold]
                all_metrics[
                    cfg.label_names[i] + f"_precision@r{recall_level * 100}_threshold"
                ] = thresholds[idx_closest_to_threshold]

        return all_metrics
