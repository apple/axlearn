# Copyright © 2023 Apple Inc.

"""Tests GLUE metrics."""
# pylint: disable=no-self-use
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from axlearn.common.metrics_glue import GLUEMetricAccumulator, GLUEMetricCalculator, glue_metrics
from axlearn.common.test_utils import DummyForwardModel, TestCase, assert_allclose
from axlearn.common.utils import NestedTensor, Tensor

# Run additional tests against T5, if installed.
try:
    import t5  # pytype: disable=import-error
except ModuleNotFoundError:
    t5 = None


@pytest.mark.skipif(t5 is None, reason="T5 is not installed.")
class T5GLUEMetricsTest(TestCase):
    def test_against_t5(self):
        # We throw in some random (possibly invalid) test cases, as though they were predicted by a
        # dummy model.
        for task, metrics in t5.data.glue_utils.GLUE_METRICS.items():
            batch_size = 100

            # Not currently supported (only test-set available).
            if task == "ax":
                continue

            # Regression task with scores typically in [0,5].
            if task == "stsb":
                preds = jax.random.uniform(
                    jax.random.PRNGKey(123), shape=[batch_size], minval=-10, maxval=10
                )
                label = jax.random.uniform(
                    jax.random.PRNGKey(321), shape=[batch_size], minval=0, maxval=5
                )
            # Classification tasks with num_classes=3.
            elif task.startswith("mnli"):
                preds = jax.random.randint(
                    jax.random.PRNGKey(123), shape=[batch_size], minval=-5, maxval=5
                )
                label = jax.random.randint(
                    jax.random.PRNGKey(321), shape=[batch_size], minval=0, maxval=3
                )
            # Binary classification tasks.
            else:
                preds = jax.random.randint(
                    jax.random.PRNGKey(123), shape=[batch_size], minval=-5, maxval=5
                )
                label = jax.random.randint(
                    jax.random.PRNGKey(321), shape=[batch_size], minval=0, maxval=2
                )

            # Compare assuming that no predictions are masked.
            ref = {}
            for metric in metrics:
                ref.update(metric(targets=np.array(label), predictions=np.array(preds)))
            test = glue_metrics(
                task=task, target_labels=label, preds=preds, mask=jnp.ones([batch_size])
            )

            # Mapping from T5 metric names to ours.
            mapping = {
                "matthews_corrcoef": "matthews_corr",
                "pearson_corrcoef": "pearson_corr",
                "spearman_corrcoef": "spearman_corr",
            }

            # Compare.
            for metric, ref_value in ref.items():
                test_value = test[mapping.get(metric, metric)]
                assert_allclose(ref_value, test_value.mean * test_value.weight)


class GLUEMetricCalculatorTest(TestCase):
    def _compute_summaries(
        self,
        task: str,
        ignore_target_label: Union[int, jnp.float32],
        input_batch: NestedTensor,
        logits: Tensor,
    ):
        with jax.sharding.Mesh(
            jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
        ):
            model = DummyForwardModel.default_config().set(name="model").instantiate(parent=None)
            cfg = GLUEMetricCalculator.default_config().set(
                name="test", task=task, ignore_target_label=ignore_target_label
            )
            calculator: GLUEMetricCalculator = cfg.instantiate(
                parent=None, model=model, model_param_partition_specs={}
            )
            state = calculator.init_state(prng_key=jax.random.PRNGKey(123), model_params={})
            input_batch["aux"] = dict(logits=logits)
            forward_outputs = calculator.forward(input_batch, model_params={}, state=state)
            state = forward_outputs["state"]
            all_forward_outputs = [forward_outputs["output"]]
            return calculator.get_summaries(
                model_params={}, state=state, all_forward_outputs=all_forward_outputs
            )

    def test_calculate_matthews_corr(self):
        task = "cola"
        num_classes = 2

        # Match everything.
        input_batch = dict(target_labels=jnp.arange(num_classes + 1))
        logits = jnp.eye(num_classes + 1, num_classes)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=2, input_batch=input_batch, logits=logits
        )

        for summary in "accuracy", "matthews_corr":
            self.assertEqual(summaries[summary].weight, num_classes)
            self.assertEqual(summaries[summary].mean, 1)

        # Match nothing.
        input_batch = dict(target_labels=jnp.ones(num_classes, dtype=jnp.int32))
        logits = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=2, input_batch=input_batch, logits=logits
        )

        for summary in "accuracy", "matthews_corr":
            self.assertEqual(summaries[summary].weight, num_classes)
            self.assertEqual(summaries[summary].mean, 0)

        # Mask everything.
        input_batch = dict(target_labels=jnp.ones(num_classes, dtype=jnp.int32) * num_classes)
        logits = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=2, input_batch=input_batch, logits=logits
        )

        for summary in "accuracy", "matthews_corr":
            self.assertEqual(summaries[summary].weight, 0)
            self.assertEqual(summaries[summary].mean, 0)

    def test_calculate_pearson_corr(self):
        task = "stsb"
        batch_size = 5

        # Perfect positive correlation.
        target_labels = jnp.arange(batch_size)
        input_batch = dict(target_labels=target_labels)
        logits = jnp.expand_dims(target_labels, axis=-1)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=jnp.nan, input_batch=input_batch, logits=logits
        )

        self.assertEqual(summaries["pearson_corr"].weight, batch_size)
        self.assertEqual(summaries["pearson_corr"].mean, 1.0)

        # Perfect negative correlation.
        target_labels = jnp.arange(batch_size)
        input_batch = dict(target_labels=target_labels)
        logits = jnp.expand_dims(target_labels[::-1], axis=-1)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=jnp.nan, input_batch=input_batch, logits=logits
        )

        self.assertEqual(summaries["pearson_corr"].weight, batch_size)
        self.assertEqual(summaries["pearson_corr"].mean, -1.0)

        # No correlation.
        input_batch = dict(
            target_labels=jax.random.randint(jax.random.PRNGKey(111), (batch_size,), 0, 100)
        )
        logits = jax.random.randint(jax.random.PRNGKey(222), (batch_size, 1), 0, 100)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=jnp.nan, input_batch=input_batch, logits=logits
        )

        self.assertEqual(summaries["pearson_corr"].weight, batch_size)
        self.assertLess(summaries["pearson_corr"].mean, 1.0)
        self.assertGreater(summaries["pearson_corr"].mean, -1.0)

        # Mask everything.
        input_batch = dict(target_labels=jnp.full(batch_size, jnp.nan))
        logits = jnp.expand_dims(jnp.arange(batch_size), axis=-1)
        summaries = self._compute_summaries(
            task=task, ignore_target_label=jnp.nan, input_batch=input_batch, logits=logits
        )

        self.assertEqual(summaries["pearson_corr"].weight, batch_size)
        self.assertTrue(jnp.isnan(summaries["pearson_corr"].mean))


class GLUEMetricAccumulatorTest(TestCase):
    @parameterized.parameters(
        dict(task_name="cola", num_classes=2),
        dict(task_name="sst2", num_classes=2),
        dict(task_name="mrpc", num_classes=2),
        dict(task_name="qqp", num_classes=2),
        dict(task_name="stsb", num_classes=1),
        dict(task_name="mnli", num_classes=3),
        dict(task_name="mnli_mismatched", num_classes=3),
        dict(task_name="qnli", num_classes=2),
        dict(task_name="rte", num_classes=2),
    )
    def test_glue_metric_accumulator(self, task_name: str, num_classes: int):
        num_batches = 10
        batch_size = 128

        gt_prng_key = jax.random.PRNGKey(111)
        pred_prng_key = jax.random.PRNGKey(222)

        if task_name == "stsb":
            y_true = jax.random.uniform(gt_prng_key, (num_batches, batch_size), jnp.float32, 1, 5)
            y_pred = jax.random.uniform(pred_prng_key, (num_batches, batch_size), jnp.float32, 1, 5)
        else:
            y_true = jax.random.randint(gt_prng_key, (num_batches, batch_size), 0, num_classes)
            y_pred = jax.random.randint(pred_prng_key, (num_batches, batch_size), 0, num_classes)
        mask = jnp.ones_like(y_true)

        metric_accumulator = GLUEMetricAccumulator.default_config().instantiate(task=task_name)
        for pred_batch, gt_batch in zip(y_pred, y_true):
            metric_accumulator.update(target_labels=gt_batch, preds=pred_batch)

        expected = glue_metrics(
            task=task_name,
            target_labels=y_true.flatten(),
            preds=y_pred.flatten(),
            mask=mask.flatten(),
        )
        actual = metric_accumulator.summaries()

        self.assertNestedAllClose(expected, actual)
