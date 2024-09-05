# Copyright © 2024 Apple Inc.

"""Tests cloud_build module."""
from absl.testing import parameterized

from axlearn.cloud.gcp.cloud_build import _get_build_request_filter
from axlearn.common.test_utils import TestCase


@parameterized.parameters(
    dict(
        image_name="",
        tags=["tag1"],
        expected_filter='(tags = tag1) OR results.images.name=""',
    ),
    dict(
        image_name="image",
        tags=[],
        expected_filter='results.images.name="image"',
    ),
    dict(
        image_name="image",
        tags=["tag1", "tag2"],
        expected_filter='(tags = tag1 AND tags = tag2) OR results.images.name="image"',
    ),
    dict(
        image_name="image",
        tags=["tag1"],
        expected_filter='(tags = tag1) OR results.images.name="image"',
    ),
)
class CloudBuildTest(TestCase):
    def test_get_cloud_build_status(self, image_name, tags, expected_filter):
        self.assertEqual(
            _get_build_request_filter(image_name=image_name, tags=tags), expected_filter
        )
