# Copyright © 2023 Apple Inc.

"""Tests normalization utils."""
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf

from axlearn.common.normalize import l2_normalize


@pytest.mark.parametrize("shape, axis", [([1, 4], -1), ([1, 3], 1), ([2, 5, 4], 2), ([1, 3, 4], 0)])
def test_l2_normalize(shape, axis):
    x = np.random.rand(*shape)
    ref = tf.math.l2_normalize(x, axis)
    assert jnp.allclose(l2_normalize(x, eps=1e-12, axis=axis), ref.numpy())
