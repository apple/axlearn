# Copyright © 2023 Apple Inc.

"""Tests classification eval pipeline."""
# pylint: disable=no-self-use
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.experimental.pjit import pjit

from axlearn.common.eval_classification import PrecisionRecallMetricCalculator
from axlearn.common.evaler_test import DummyModel
from axlearn.common.test_utils import assert_allclose
from axlearn.common.utils import NestedTensor, Tensor


def _dummy_data_generator(
    input_ids: Tensor,
    target_labels: Tensor,
    predictions: Tensor,
    batch_size: int = 2,
):
    """A dummy data generator spites out input data with specific batch size."""
    for batch_idx in range(int(jnp.ceil(len(input_ids) / batch_size))):
        start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
        yield {
            "input_ids": jnp.asarray(input_ids[start:end]),
            "target_labels": jnp.asarray(target_labels[start:end]),
            "logits": jnp.array(predictions[start:end]),
        }


# pylint: disable-next=abstract-method
class DummyClassificationModel(DummyModel):
    """A dummy model which spites out input logits."""

    # pylint: disable-next=no-self-use
    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        return {"logits": input_batch["logits"]}


def _compute_metrics(
    *,
    data_generator: Iterable,
    calculator_cfg: PrecisionRecallMetricCalculator.Config,
) -> dict:
    """Computes classification metrics on the entire provided evaluation set.

    Args:
        data_generator: An Iterable that yield batch dummy samples.
        calculator_cfg: A subclass of ClassificationMetricCalculator.Config used to
            instantiate the metric calculator.
    Returns:
        "summaries": a Dict of WeightedScalar values computed by
            A subclass of ClassificationMetricCalculator.
    """
    with jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
    ):
        model = DummyClassificationModel.default_config().set(name="model").instantiate(parent=None)
        model_param_partition_specs = jax.tree.map(
            lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
        )
        calculator = calculator_cfg.set(name="calculator").instantiate(
            parent=None, model=model, model_param_partition_specs=model_param_partition_specs
        )
        model_params = pjit(
            model.initialize_parameters_recursively,
            in_shardings=(None,),
            out_shardings=model_param_partition_specs,
        )(jax.random.PRNGKey(0))

        state = calculator.init_state(prng_key=jax.random.PRNGKey(0), model_params=model_params)
        all_forward_outputs = []
        for input_batch in data_generator:
            forward_outputs = calculator.forward(
                input_batch, model_params=model_params, state=state
            )
            state = forward_outputs["state"]
            all_forward_outputs.append(forward_outputs["output"])

        summaries = calculator.get_summaries(
            model_params=model_params, state=state, all_forward_outputs=all_forward_outputs
        )
        return summaries


class ClassificationMetricCalculatorTest(parameterized.TestCase):
    @parameterized.parameters(
        # Test case for multiple precision, recall level
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[1], [1], [1], [1], [0], [1]]),
            "predictions": jnp.array([[0.4], [1], [0.8], [0.9], [0], [0.6]]),
            "batch_size": 6,
            "precision_levels": [0.9, 0.85],
            "recall_levels": [0.8, 0.75],
            "expected_metrics": {
                "0_recall@p90.0": 1.0,
                "0_recall@p90.0_threshold": 0.4,
                "0_recall@p85.0": 1.0,
                "0_recall@p85.0_threshold": 0.4,
                "0_precision@r80.0": 1.0,
                "0_precision@r80.0_threshold": 0.6,
                "0_precision@r75.0": 1.0,
                "0_precision@r75.0_threshold": 0.6,
            },
        },
        # Test case for multi-batch metric aggregation.
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[1], [1], [1], [1], [0], [1]]),
            "predictions": jnp.array([[0.4], [1], [1], [1], [0], [1]]),
            "batch_size": 2,
            "precision_levels": [0.9],
            "recall_levels": [0.8],
            "expected_metrics": {
                "0_recall@p90.0": 1.0,
                "0_recall@p90.0_threshold": 0.4,
                "0_precision@r80.0": 1.0,
                "0_precision@r80.0_threshold": 1.0,
            },
        },
        # Test case for single batch metric aggregation.
        # Notice this is a noisy case where precision is not necessarily negatively correlated to
        # recall.
        # Precision, recall with threshold 0 are 0.83 and 1. With threshold 0.3, they are 0.8 and
        # 0.8.
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[1], [1], [1], [1], [0], [1]]),
            "predictions": jnp.array([[0], [0.5], [0.6], [0.4], [0.3], [1]]),
            "batch_size": 6,
            "precision_levels": [0.9],
            "recall_levels": [0.8],
            "expected_metrics": {
                "0_recall@p90.0": 0.8,
                "0_recall@p90.0_threshold": 0.4,
                "0_precision@r80.0": 0.8,
                "0_precision@r80.0_threshold": 0.3,
            },
        },
        # Test case for boundary case. Precision level 1 and Recall level 1.
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[1], [1], [1], [1], [0], [1]]),
            "predictions": jnp.array([[0], [0.5], [0.6], [0.4], [0.3], [1]]),
            "batch_size": 3,
            "precision_levels": [1.0],
            "recall_levels": [1.0],
            "expected_metrics": {
                "0_recall@p100.0": 0.8,
                "0_recall@p100.0_threshold": 0.4,
                "0_precision@r100.0": 0.8333333,
                "0_precision@r100.0_threshold": 0,
            },
        },
        # Test case where the target labels have length larger than 1.
        # In this case, multiple thresholds has recall 1. We picked the first one (i.e.
        # the one with the lowest precision to report due to sorting).
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[1, 0], [1, 1], [0, 0], [0, 1], [1, 1], [0, 0]]),
            "predictions": jnp.array(
                [[0.9, 0.4], [0.6, 0.8], [0.3, 0.4], [0.4, 0.6], [0.8, 0.1], [0.2, 0.1]]
            ),
            "batch_size": 3,
            "precision_levels": [0.85],
            "recall_levels": [0.8],
            "expected_metrics": {
                "0_recall@p85.0": 1.0,
                "0_recall@p85.0_threshold": 0.6,
                "0_precision@r80.0": 0.5,
                "0_precision@r80.0_threshold": 0.2,
                "1_recall@p85.0": 0.66666667,
                "1_recall@p85.0_threshold": 0.6,
                "1_precision@r80.0": 0.5,
                "1_precision@r80.0_threshold": 0.1,
            },
        },
        # Test case where metric target can't be achieved.
        # In this case, precision recall will be both -1s since we can't achieve this target.
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[0], [0], [0], [0], [0], [0]]),
            "predictions": jnp.array([[0], [0], [0], [0], [0], [0]]),
            "batch_size": 2,
            "precision_levels": [1.0],
            "recall_levels": [1.0],
            "expected_metrics": {
                "0_recall@p100.0": -1.0,
                "0_recall@p100.0_threshold": 0.0,
                "0_precision@r100.0": -1.0,
                "0_precision@r100.0_threshold": 0.0,
            },
        },
        # Test case where we have padding samples.
        {
            "input_ids": jnp.zeros(shape=(6, 5)),
            "target_labels": jnp.array([[-1], [1], [-1], [1], [0], [1]]),
            "predictions": jnp.array([[0], [0.5], [0.6], [0.4], [0.3], [1]]),
            "batch_size": 3,
            "precision_levels": [1.0],
            "recall_levels": [1.0],
            "expected_metrics": {
                "0_recall@p100.0": 1.0,
                "0_recall@p100.0_threshold": 0.4,
                "0_precision@r100.0": 0.75,
                "0_precision@r100.0_threshold": 0.3,
            },
        },
    )
    def test_classification_metrics(
        self,
        input_ids: NestedTensor,
        target_labels: NestedTensor,
        predictions: NestedTensor,
        batch_size: int,
        precision_levels: float,
        recall_levels: float,
        expected_metrics: dict[str, float],
    ):
        summaries = _compute_metrics(
            calculator_cfg=PrecisionRecallMetricCalculator.default_config().set(
                label_names=["0", "1", "2"],
                precision_levels=precision_levels,
                recall_levels=recall_levels,
            ),
            data_generator=_dummy_data_generator(
                input_ids=input_ids,
                target_labels=target_labels,
                predictions=predictions,
                batch_size=batch_size,
            ),
        )
        for key in summaries.keys():
            assert_allclose(summaries[key], expected_metrics[key])
