# Copyright © 2023 Apple Inc.

"""Tests correlation metrics."""

# pylint: disable=no-self-use
import evaluate
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from scipy.stats import pearsonr as scipy_pearson_corrcoef
from scipy.stats import rankdata as scipy_rankdata
from scipy.stats import spearmanr as scipy_spearman_corrcoef
from sklearn.metrics import matthews_corrcoef as sklearn_matthews_corrcoef

from axlearn.common.metrics_correlation import (
    _rankdata,
    matthews_corrcoef,
    pearson_corrcoef,
    spearman_corrcoef,
)
from axlearn.common.test_utils import TestWithTemporaryCWD, assert_allclose


# Note: Hugging Face evaluate.load depends on an existence of a cwd.
class TestMetrics(TestWithTemporaryCWD):
    """Tests metric utils."""

    @parameterized.product(batch_size=[100, 500, 1000, 5000, 10000], num_classes=[2, 10, 20, 30])
    def test_matthews_corrcoef(self, batch_size, num_classes):
        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)
        weight = jax.random.uniform(jax.random.PRNGKey(111), [batch_size], jnp.float32, 1, 10)

        jit_matthews_corrcoef = jax.jit(matthews_corrcoef)

        # Test equivalence with sklearn.
        actual = jit_matthews_corrcoef(pred, label, weight=weight)
        expected = sklearn_matthews_corrcoef(pred, label, sample_weight=weight)
        assert_allclose(expected, actual)

        # Test a case where number of unique labels is smaller than batch size.
        # In this case, we pad uniques with nans.
        pred = jnp.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        label = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        actual = jit_matthews_corrcoef(pred, label)
        expected = sklearn_matthews_corrcoef(pred, label)
        assert_allclose(expected, actual)

    @parameterized.product(batch_size=[100, 500, 1000, 5000, 10000], num_classes=[2, 10, 20, 30])
    @pytest.mark.skip(reason="Intended to be run manually as it requires `evaluate.load`.")
    def test_matthews_corrcoef_hf(self, batch_size, num_classes):
        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)

        # Test equivalence with hf.
        jit_matthews_corrcoef = jax.jit(matthews_corrcoef)
        actual = jit_matthews_corrcoef(pred, label)
        metric = evaluate.load("glue", config_name="cola")
        expected = metric.compute(predictions=pred, references=label)["matthews_correlation"]
        assert_allclose(expected, actual)

    def test_matthews_corrcoef_validation(self):
        with self.assertRaisesRegex(ValueError, "rank 1"):
            matthews_corrcoef(jnp.ones((1, 2)), jnp.ones((1, 2)))
        with self.assertRaisesRegex(ValueError, "shapes should be equal"):
            matthews_corrcoef(jnp.ones(5), jnp.ones(3))

    def test_pearson_corrcoef(self):
        batch_size = 100
        num_classes = 30

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)

        jit_pearson_corrcoef = jax.jit(pearson_corrcoef)
        actual = jit_pearson_corrcoef(pred, label)

        # Test equivalence with scipy.
        expected = scipy_pearson_corrcoef(pred, label)[0]
        assert_allclose(expected, actual)

        # Test equivalence with hf and scipy with mask.
        # We do so by adding (ignored) padding and verifying the result is the same.
        mask = jnp.concatenate([jnp.ones_like(pred), jnp.zeros(10)])
        pred = jnp.concatenate(
            [pred, jax.random.randint(jax.random.PRNGKey(111), [10], 0, num_classes)]
        )
        label = jnp.concatenate(
            [label, jax.random.randint(jax.random.PRNGKey(222), [10], 0, num_classes)]
        )
        actual = jit_pearson_corrcoef(pred, label, weight=mask)
        assert_allclose(expected, actual)

        # Test when everything is masked.
        self.assertEqual(pearson_corrcoef(pred, label, weight=jnp.zeros_like(pred)), 0)

    @pytest.mark.skip(reason="Intended to be run manually as it requires `evaluate.load`.")
    def test_pearson_corrcoef_hf(self):
        batch_size = 100
        num_classes = 30

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)

        jit_pearson_corrcoef = jax.jit(pearson_corrcoef)
        actual = jit_pearson_corrcoef(pred, label)

        # Test equivalence with hf.
        hf_metric = evaluate.load("glue", config_name="stsb")
        expected = hf_metric.compute(predictions=pred, references=label)["pearson"]
        assert_allclose(expected, actual)

    def test_rankdata(self):
        batch_size = 100

        arr = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, 100)

        jit_rank_data = jax.jit(_rankdata)
        actual = jit_rank_data(arr)

        # Test equivalence with scipy.
        expected = scipy_rankdata(arr)
        assert_allclose(expected, actual)

    def test_spearman_corrcoef(self):
        batch_size = 100

        pred = jax.random.uniform(jax.random.PRNGKey(123), [batch_size], minval=0, maxval=5)
        label = jax.random.uniform(jax.random.PRNGKey(321), [batch_size], minval=0, maxval=5)

        jit_spearman_corrcoef = jax.jit(spearman_corrcoef)
        actual = jit_spearman_corrcoef(pred, label)

        # Test equivalence with scipy.
        expected = scipy_spearman_corrcoef(pred, label).correlation
        assert_allclose(expected, actual)

        # Test equivalence with scipy with mask
        mask = jnp.concatenate([jnp.ones_like(pred), jnp.zeros(10)])
        pred = jnp.concatenate(
            [pred, jax.random.uniform(jax.random.PRNGKey(111), [10], minval=0, maxval=5)]
        )
        label = jnp.concatenate(
            [label, jax.random.uniform(jax.random.PRNGKey(222), [10], minval=0, maxval=5)]
        )
        actual = jit_spearman_corrcoef(pred, label, mask=mask)
        assert_allclose(expected, actual)

    @pytest.mark.skip(reason="Intended to be run manually as it requires `evaluate.load`.")
    def test_spearman_corrcoef_hf(self):
        batch_size = 100

        pred = jax.random.uniform(jax.random.PRNGKey(123), [batch_size], minval=0, maxval=5)
        label = jax.random.uniform(jax.random.PRNGKey(321), [batch_size], minval=0, maxval=5)

        jit_spearman_corrcoef = jax.jit(spearman_corrcoef)
        actual = jit_spearman_corrcoef(pred, label)

        # Test equivalence with hf.
        hf_metric = evaluate.load("glue", config_name="stsb")
        expected = hf_metric.compute(predictions=pred, references=label)["spearmanr"]
        assert_allclose(expected, actual)

    def test_pearson_corrcoef_validation(self):
        with self.assertRaisesRegex(ValueError, "shapes should be equal"):
            pearson_corrcoef(jnp.ones(5), jnp.ones(3))
