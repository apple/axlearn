# Copyright © 2024 Apple Inc.

"""Tests cloud_build module."""
from axlearn.cloud.gcp.cloud_build import _get_build_request_filter
from axlearn.common.test_utils import TestCase


class CloudBuildTest(TestCase):
    def test_get_cloud_build_status(self):
        self.assertEqual(
            _get_build_request_filter(image_name="image", tags=["tag1", "tag2"]),
            'tags = tag1 AND tags = tag2 OR results.images.name="image"',
        )
