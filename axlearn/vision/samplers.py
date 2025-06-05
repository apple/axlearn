# Copyright © 2023 Apple Inc.

"""Sampling Ops."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common.utils import Tensor, safe_not


def sample(*, is_candidate: Tensor, size: Tensor, prng_key: Tensor) -> Tensor:
    """Samples candidate elements.

    Samples `size` elements along the last axis and returns a boolean tensor indicating their
    locations. If number of candidates is smaller than `size` returns all candidates.

    Args:
        is_candidate: A bool tensor of shape [..., N] indicating candidate elements that can be
            sampled.
        size: A tensor of shape [...] indicating sample size along the last dimension of
            `is_candidate`.
        prng_key: PRNG key.

    Returns:
        A bool tensor of shape [...., N] indicating the locations of the sample.
    """
    indices = np.broadcast_to(np.arange(is_candidate.shape[-1]), is_candidate.shape)
    permutation = jax.random.permutation(prng_key, indices, axis=-1, independent=True)
    inverse_permutation = jnp.argsort(permutation, axis=-1)
    is_candidate = jnp.take_along_axis(is_candidate, permutation, axis=-1)
    labeled_candidates = jnp.cumsum(is_candidate, axis=-1) * is_candidate
    sampled_candidates = (labeled_candidates > 0) & (labeled_candidates <= size[..., None])
    reordered_sample = jnp.take_along_axis(sampled_candidates, inverse_permutation, axis=-1)
    return reordered_sample


@dataclass
class LabelSamples:
    """A container for results from LabelSampler.

    indices: An int32 tensor of shape [..., sample_size] containing the indices of the sampled
        labels. The values are in [0, N) where N is the length of label tensor.
    paddings: A bool tensor of shape [..., sample_size] indicating whether the elements of
        `indices` are paddings.
    """

    indices: Tensor
    paddings: Tensor


class LabelSampler:
    """Sampler for foreground and background labels.

    Samples foreground and background labels at the specified rate.
    """

    def __init__(
        self, size: int, foreground_fraction: float, background_label: int, ignore_label: int
    ):
        """Constructs LabelSampler.

        Args:
            size: Total sample size.
            foreground_fraction: Fraction of subsample with foreground labels. Rest are background
                labels. If the total number of foreground samples are insufficient, samples
                additional background for the deficit.
            background_label: Value indicating background label.
            ignore_label: Value indicating labels to ignore.
        """
        self.num_foreground = int(size * foreground_fraction)
        self.size = size
        self.background_label = background_label
        self.ignore_label = ignore_label

    def __call__(self, *, labels: Tensor, paddings: Tensor, prng_key: Tensor) -> LabelSamples:
        """Samples foreground and background labels at the specified rate.

        Avoids sampling `ignore` labels and padding.

        Args:
            labels: A int32 tensor of shape [..., N] containing labels.
            paddings: A bool tensor of shape [..., N] indicating paddings.
            prng_key: PRNG key

        Returns:
            A LabelSamples object containing the sample indices.
        """
        prng_key1, prng_key2 = jax.random.split(prng_key, num=2)
        foreground_candidates = (
            safe_not(paddings) & (labels != self.ignore_label) & (labels != self.background_label)
        )
        num_foreground = jnp.minimum(jnp.sum(foreground_candidates, axis=-1), self.num_foreground)
        foreground_samples = sample(
            is_candidate=foreground_candidates,
            size=num_foreground,
            prng_key=prng_key1,
        )
        background_candidates = safe_not(paddings) & (labels == self.background_label)
        num_background = self.size - num_foreground
        background_samples = sample(
            is_candidate=background_candidates,
            size=num_background,
            prng_key=prng_key2,
        )
        samples = foreground_samples | background_samples
        _, indices = jax.lax.top_k(samples, k=self.size)
        paddings = jnp.bitwise_not(jnp.take_along_axis(samples, indices, axis=-1))
        return LabelSamples(indices=indices, paddings=paddings)
