# Copyright © 2023 Apple Inc.

"""Tests normalization utils."""

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common.normalize import l2_normalize


class L2NormalizeTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(shape=[1, 4], axis=-1),
        dict(shape=[1, 3], axis=1),
        dict(shape=[2, 5, 4], axis=2),
        dict(shape=[1, 3, 4], axis=0),
    )
    def test_l2_normalize(self, shape, axis):
        x = np.random.rand(*shape)
        ref = tf.math.l2_normalize(x, axis)
        assert jnp.allclose(l2_normalize(x, eps=1e-12, axis=axis), ref.numpy())


if __name__ == "__main__":
    absltest.main()
