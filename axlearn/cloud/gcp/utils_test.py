# Copyright © 2023 Apple Inc.

"""Tests general GCP utils."""

from absl.testing import parameterized

from axlearn.cloud.gcp import utils


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    @parameterized.parameters(
        dict(name="test--01-exp123", should_raise=False),
        dict(name="123-test", should_raise=True),  # Must begin with letter.
        dict(name="test+123", should_raise=True),  # No other special characters allowed.
        dict(name="a" * 64, should_raise=True),  # Too long.
    )
    def test_validate_resource_name(self, name: str, should_raise: bool):
        if should_raise:
            with self.assertRaises(ValueError):
                utils.validate_resource_name(name)
        else:
            utils.validate_resource_name(name)
