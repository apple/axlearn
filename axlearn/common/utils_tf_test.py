# Copyright © 2023 Apple Inc.

"""Tests Tensorflow utils."""
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common.utils_tf import masked_fill


class MaskedFillTest(parameterized.TestCase):
    @parameterized.parameters(tf.string, tf.int32, tf.float32)
    def test_basic(self, dtype):
        if dtype == tf.string:
            orig = ["a", "b", "c"]
            fill_value = "x"
        else:
            orig = [1, 2, 3]
            fill_value = tf.constant(-1, dtype=dtype)
        result = masked_fill(
            tf.convert_to_tensor(orig, dtype=dtype),
            mask=tf.convert_to_tensor([True, False, True]),
            fill_value=fill_value,
        )
        if dtype == tf.string:
            self.assertSequenceEqual(result.numpy().tolist(), [b"x", b"b", b"x"])
        else:
            self.assertSequenceEqual(result.numpy().tolist(), [-1, 2, -1])


if __name__ == "__main__":
    absltest.main()
