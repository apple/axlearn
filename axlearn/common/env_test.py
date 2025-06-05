"""Tests for AXLearn environment."""

# pylint: disable=no-self-use

import tensorflow_io  # noqa: F401 # pylint: disable=unused-import
from absl.testing import absltest
from tensorflow import io as tf_io

from axlearn.common import test_utils


class EnvTest(test_utils.TestCase):
    def test_tf_io_s3_support(self):
        self.assertIn("s3", tf_io.gfile.get_registered_schemes())


if __name__ == "__main__":
    absltest.main()
