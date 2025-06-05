# Copyright © 2023 Apple Inc.

"""Tests samplers."""
import jax
import numpy as np
from absl.testing import absltest

from axlearn.common.test_utils import set_threefry_partitionable
from axlearn.vision import samplers


# pylint: disable=no-self-use
class SampleTest(absltest.TestCase):
    """Tests sample."""

    def test_sample_elements_are_valid(self):
        is_candidate = np.array(
            [
                [True, True, False, True, False, True, True, False],
                [True, False, False, True, True, False, True, True],
            ]
        )
        prng_key = jax.random.PRNGKey(123)
        sample_size = np.array([3])
        sample = samplers.sample(is_candidate=is_candidate, size=sample_size, prng_key=prng_key)
        np.testing.assert_array_equal(sample & is_candidate, sample)

    def test_sample_all_valid_elements(self):
        is_candidate = np.array(
            [
                [True, True, False, True, False, True, True, False],
                [True, False, False, True, True, False, True, True],
            ]
        )
        prng_key = jax.random.PRNGKey(123)
        sample_size = np.array([8])
        sample = samplers.sample(is_candidate=is_candidate, size=sample_size, prng_key=prng_key)
        np.testing.assert_array_equal(sample, is_candidate)

    def test_sample_candidates_uniformly(self):
        is_candidate = np.array(
            [
                [True, True, False, True, False, True, True, False],
            ]
        )
        sample_size = np.array([3])
        runs = 1000
        prng_keys = jax.random.split(jax.random.PRNGKey(123), num=runs)
        num_samples = np.zeros_like(is_candidate)
        expected_prob = sample_size / np.sum(is_candidate)
        for prng_key in prng_keys:
            num_samples += samplers.sample(
                is_candidate=is_candidate, size=sample_size, prng_key=prng_key
            ).astype(float)
        np.testing.assert_allclose(
            expected_prob * is_candidate, num_samples / runs, rtol=0.2, atol=0.0
        )

    def test_sample_size(self):
        is_candidate1 = np.zeros(1024, dtype=bool)
        is_candidate1[np.random.randint(low=0, high=1024, size=128)] = True
        is_candidate2 = np.zeros(1024, dtype=bool)
        is_candidate2[np.random.randint(low=0, high=1024, size=128)] = True
        is_candidate = np.stack([is_candidate1, is_candidate2])

        size = np.array([32, 32])
        prng_key = jax.random.PRNGKey(123)
        samples = samplers.sample(is_candidate=is_candidate, size=size, prng_key=prng_key)
        np.testing.assert_array_equal(np.sum(samples, axis=-1), [32, 32])

    def test_sample_with_insufficient_candidates(self):
        is_candidate1 = np.zeros(1024, dtype=bool)
        is_candidate1[np.random.choice(np.arange(1024), size=128, replace=False)] = 1
        is_candidate2 = np.zeros(1024, dtype=bool)
        is_candidate2[np.random.choice(np.arange(1024), size=64, replace=False)] = 1
        is_candidate = np.stack([is_candidate1, is_candidate2])

        size = np.array([256, 256])
        prng_key = jax.random.PRNGKey(123)
        samples = samplers.sample(is_candidate=is_candidate, size=size, prng_key=prng_key)
        np.testing.assert_array_equal(np.sum(samples, axis=-1), [128, 64])


class LabelSamplerTest(absltest.TestCase):
    """Tests LabelSampler."""

    def test_sample_foreground_only(self):
        labels = np.array([[1, 1, 0, -1, 0, 1]])
        paddings = np.zeros_like(labels, dtype=bool)
        prng_key = jax.random.PRNGKey(123)
        size = 2
        sampler = samplers.LabelSampler(
            size=size, foreground_fraction=1.0, background_label=0, ignore_label=-1
        )
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(size, samples.indices.shape[-1])
        np.testing.assert_array_equal(1, labels[..., samples.indices])
        np.testing.assert_array_equal(False, samples.paddings)

    def test_sample_background_only(self):
        labels = np.array([[1, 1, 0, -1, 0, 0, 1]])
        paddings = np.zeros_like(labels, dtype=bool)
        prng_key = jax.random.PRNGKey(123)
        size = 2
        sampler = samplers.LabelSampler(
            size=size, foreground_fraction=0.0, background_label=0, ignore_label=-1
        )
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(size, samples.indices.shape[-1])
        np.testing.assert_array_equal(0, labels[..., samples.indices])
        np.testing.assert_array_equal(False, samples.paddings)

    def test_sample_foreground_background_rates(self):
        labels1 = np.zeros(1024, dtype=np.int32)
        labels1[np.random.choice(np.arange(512), size=128, replace=False)] = 1
        labels1[np.random.choice(np.arange(512, 1024), size=64, replace=False)] = -1
        labels2 = np.zeros(1024, dtype=np.int32)
        labels2[np.random.choice(np.arange(512), size=128, replace=False)] = 1
        labels2[np.random.choice(np.arange(512, 1024), size=64, replace=False)] = -1
        labels = np.stack([labels1, labels2])
        paddings = np.zeros_like(labels, dtype=bool)
        sampler = samplers.LabelSampler(
            size=64, foreground_fraction=0.25, background_label=0, ignore_label=-1
        )
        prng_key = jax.random.PRNGKey(123)
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(64, samples.indices.shape[-1])
        np.testing.assert_array_equal(
            [16, 16], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 1, axis=-1)
        )
        np.testing.assert_array_equal(
            [48, 48], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 0, axis=-1)
        )
        np.testing.assert_array_equal(False, samples.paddings)

    def test_sample_with_insufficient_foreground_candidates(self):
        labels1 = np.zeros(1024, dtype=np.int32)
        labels1[np.random.choice(np.arange(512), size=10, replace=False)] = 1
        labels1[np.random.choice(np.arange(512, 1024), size=64, replace=False)] = -1
        labels2 = np.zeros(1024, dtype=np.int32)
        labels2[np.random.choice(np.arange(16), size=16, replace=False)] = 1
        labels2[np.random.choice(np.arange(16, 1024), size=900, replace=False)] = -1
        labels = np.stack([labels1, labels2])
        paddings = np.zeros_like(labels, dtype=bool)
        sampler = samplers.LabelSampler(
            size=64, foreground_fraction=0.25, background_label=0, ignore_label=-1
        )
        prng_key = jax.random.PRNGKey(123)
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(64, samples.indices.shape[-1])
        np.testing.assert_array_equal(
            [10, 16], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 1, axis=-1)
        )
        np.testing.assert_array_equal(
            [54, 48], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 0, axis=-1)
        )
        np.testing.assert_array_equal(False, samples.paddings)

    def test_sample_with_insufficient_total_candidates(self):
        labels1 = -1 * np.ones(1024, dtype=np.int32)
        labels1[np.random.choice(np.arange(512), size=10, replace=False)] = 1
        labels1[np.random.choice(np.arange(512, 1024), size=40, replace=False)] = 0
        labels2 = -1 * np.ones(1024, dtype=np.int32)
        labels2[np.random.choice(np.arange(16), size=16, replace=False)] = 1
        labels2[np.random.choice(np.arange(16, 1024), size=42, replace=False)] = 0
        labels = np.stack([labels1, labels2])
        paddings = np.zeros_like(labels, dtype=bool)
        sampler = samplers.LabelSampler(
            size=64, foreground_fraction=0.25, background_label=0, ignore_label=-1
        )
        prng_key = jax.random.PRNGKey(123)
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(64, samples.indices.shape[-1])
        np.testing.assert_array_equal(
            [10, 16], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 1, axis=-1)
        )
        np.testing.assert_array_equal(
            [40, 42], np.sum(np.take_along_axis(labels, samples.indices, axis=-1) == 0, axis=-1)
        )
        np.testing.assert_array_equal([14, 6], np.sum(samples.paddings, axis=-1))

    @set_threefry_partitionable(False)  # TODO(markblee): update for threefry_partitionable True
    def test_exclude_ignore_and_paddings(self):
        labels = np.array([[1, 1, 0, 0, -1, 0, 1]])
        paddings = np.array([[True, False, False, False, False, True, False]])
        prng_key = jax.random.PRNGKey(123)
        sampler = samplers.LabelSampler(
            size=2, foreground_fraction=0.5, background_label=0, ignore_label=-1
        )
        samples = sampler(labels=labels, paddings=paddings, prng_key=prng_key)
        np.testing.assert_array_equal(2, samples.indices.shape[-1])
        out_labels = np.take_along_axis(labels, samples.indices, axis=-1)
        np.testing.assert_array_equal(0, np.sum(out_labels == -1))
        # Jax newer versions have a strong enforcement of shape for == operator.
        np.testing.assert_array_equal(0, np.sum(out_labels == paddings[:, :2]))
        np.testing.assert_array_equal(False, samples.paddings)
