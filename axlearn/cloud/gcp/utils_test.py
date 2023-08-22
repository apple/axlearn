# Copyright © 2023 Apple Inc.

"""Tests general GCP utils."""

from absl.testing import parameterized

from axlearn.cloud.gcp import utils


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    @parameterized.parameters(
        dict(name="test--01-exp123", expected=True),
        dict(name="123-test", expected=False),  # Must begin with letter.
        dict(name="test+123", expected=False),  # No other special characters allowed.
    )
    def test_is_valid_resource_name(self, name: str, expected: bool):
        self.assertEqual(expected, utils.is_valid_resource_name(name))
