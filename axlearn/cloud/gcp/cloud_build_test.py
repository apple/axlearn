# Copyright © 2024 Apple Inc.

"""Tests cloud_build module."""
from unittest import mock

from absl.testing import parameterized
from google.cloud.compute_v1 import ListRegionsRequest, RegionsClient
from google.cloud.devtools.cloudbuild_v1 import Build, ListBuildsRequest

from axlearn.cloud.gcp.cloud_build import (
    CloudBuildStatus,
    _get_build_request_filter,
    _get_cloud_build_status_for_region,
    _list_available_regions,
    get_cloud_build_status,
)
from axlearn.common.test_utils import TestCase


class CloudBuildTest(TestCase):
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
    def test_get_cloud_build_status(self, image_name, tags, expected_filter):
        self.assertEqual(
            _get_build_request_filter(image_name=image_name, tags=tags), expected_filter
        )

    @mock.patch("axlearn.cloud.gcp.cloud_build.RegionsClient")
    def test_list_available_regions_success(self, mock_regions_client):
        project_id = "test-project"
        mock_client = mock.Mock(spec=RegionsClient)
        mock_region_1 = mock.Mock()
        mock_region_1.name = "us-central1"
        mock_region_2 = mock.Mock()
        mock_region_2.name = "europe-west1"

        mock_client.list.return_value = [mock_region_1, mock_region_2]
        mock_regions_client.return_value = mock_client

        regions = _list_available_regions(project_id)

        self.assertEqual(regions, ["us-central1", "europe-west1"])
        mock_client.list.assert_called_once_with(request=ListRegionsRequest(project=project_id))

    @mock.patch("axlearn.cloud.gcp.cloud_build.logging.error")
    @mock.patch("axlearn.cloud.gcp.cloud_build.RegionsClient")
    def test_list_available_regions_raises_api_error(self, mock_regions_client, mock_log_error):
        project_id = "test-project"
        mock_client = mock.Mock(spec=RegionsClient)
        mock_client.list.side_effect = Exception("API error")
        mock_regions_client.return_value = mock_client

        with self.assertRaises(Exception) as context:
            _list_available_regions(project_id)

        self.assertIn("API error", str(context.exception))
        mock_log_error.assert_called_once_with(
            "Failed to look up regions for project: %s", mock_client.list.side_effect
        )
        mock_client.list.assert_called_once_with(request=ListRegionsRequest(project=project_id))

    @mock.patch("google.cloud.devtools.cloudbuild_v1.CloudBuildClient")
    def test_get_cloud_build_status_for_region_success(self, mock_cloudbuild_client):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag-1", "tag-2"]
        region = "us-central1"
        mock_build_1 = mock.Mock()
        mock_build_1.status = Build.Status.WORKING
        mock_build_1.create_time = (
            "2023-07-30T00:00:00Z"  # Mock the creation time of an older build.
        )
        mock_build_2 = mock.Mock()
        mock_build_2.status = Build.Status.SUCCESS
        mock_build_2.create_time = (
            "2023-07-31T00:00:00Z"  # Mock the creation time of a newer build.
        )

        mock_client = mock.Mock()
        mock_client.list_builds.return_value = [mock_build_1, mock_build_2]
        mock_cloudbuild_client.return_value = mock_client
        expected_status = CloudBuildStatus.SUCCESS

        # Call the function with the mocked client.
        status = _get_cloud_build_status_for_region(
            project_id=project_id, image_name=image_name, tags=tags, region=region
        )

        self.assertEqual(status, expected_status)
        mock_client.list_builds.assert_called_once_with(
            request=ListBuildsRequest(
                parent=f"projects/{project_id}/locations/{region}",
                project_id=project_id,
                filter=_get_build_request_filter(image_name=image_name, tags=tags),
            )
        )

    @mock.patch("google.cloud.devtools.cloudbuild_v1.CloudBuildClient")
    def test_get_cloud_build_status_for_region_failure(self, mock_cloudbuild_client):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag-1", "tag-2"]
        region = "us-central1"
        mock_build_1 = mock.Mock()
        mock_build_1.status = Build.Status.WORKING
        mock_build_1.create_time = (
            "2023-07-30T00:00:00Z"  # Mock the creation time of an older build.
        )
        mock_build_2 = mock.Mock()
        mock_build_2.status = Build.Status.FAILURE
        mock_build_2.create_time = (
            "2023-07-31T00:00:00Z"  # Mock the creation time of a newer build.
        )

        mock_client = mock.Mock()
        mock_client.list_builds.return_value = [mock_build_1, mock_build_2]
        mock_cloudbuild_client.return_value = mock_client
        expected_status = CloudBuildStatus.FAILURE

        # Call the function with the mocked client.
        status = _get_cloud_build_status_for_region(
            project_id=project_id, image_name=image_name, tags=tags, region=region
        )

        self.assertEqual(status, expected_status)
        mock_client.list_builds.assert_called_once_with(
            request=ListBuildsRequest(
                parent=f"projects/{project_id}/locations/{region}",
                project_id=project_id,
                filter=_get_build_request_filter(image_name=image_name, tags=tags),
            )
        )

    @mock.patch("google.cloud.devtools.cloudbuild_v1.CloudBuildClient")
    def test_get_cloud_build_status_for_region_status_unknown(self, mock_cloudbuild_client):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag-1", "tag-2"]
        region = "us-central1"
        mock_build_1 = mock.Mock()
        mock_build_1.status = Build.Status.WORKING
        mock_build_1.create_time = (
            "2023-07-30T00:00:00Z"  # Mock the creation time of an older build.
        )
        mock_build_2 = mock.Mock()
        mock_build_2.status = Build.Status.STATUS_UNKNOWN
        mock_build_2.create_time = (
            "2023-07-31T00:00:00Z"  # Mock the creation time of a newer build.
        )

        mock_client = mock.Mock()
        mock_client.list_builds.return_value = [mock_build_1, mock_build_2]
        mock_cloudbuild_client.return_value = mock_client
        expected_status = CloudBuildStatus.STATUS_UNKNOWN

        # Call the function with the mocked client.
        status = _get_cloud_build_status_for_region(
            project_id=project_id, image_name=image_name, tags=tags, region=region
        )

        self.assertEqual(status, expected_status)
        mock_client.list_builds.assert_called_once_with(
            request=ListBuildsRequest(
                parent=f"projects/{project_id}/locations/{region}",
                project_id=project_id,
                filter=_get_build_request_filter(image_name=image_name, tags=tags),
            )
        )

    @mock.patch("axlearn.cloud.gcp.cloud_build.cloudbuild_v1.CloudBuildClient")
    def test_get_cloud_build_status_for_region_raises_exception(self, mock_cloud_build_client):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag1", "tag2"]
        region = "us-central1"
        mock_client = mock.Mock()
        mock_cloud_build_client.return_value = mock_client
        mock_client.list_builds.side_effect = Exception("Failed to find image")

        with self.assertRaises(Exception) as cm:
            _get_cloud_build_status_for_region(
                project_id=project_id, image_name=image_name, tags=tags, region=region
            )
            self.assertEqual(str(cm.exception), "Failed to find image")

        mock_client.list_builds.assert_called_once_with(
            request=ListBuildsRequest(
                parent=f"projects/{project_id}/locations/{region}",
                project_id=project_id,
                filter=_get_build_request_filter(image_name=image_name, tags=tags),
            )
        )

    @mock.patch("axlearn.cloud.gcp.cloud_build._get_cloud_build_status_for_region")
    @mock.patch("axlearn.cloud.gcp.cloud_build._list_available_regions")
    def test_get_cloud_build_status_success_with_two_regions(
        self, mock_list_available_regions, mock_get_cloud_build_status_for_region
    ):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag1", "tag2"]

        mock_list_available_regions.return_value = ["us-central1", "europe-west1"]
        # Mock the return value for each region to simulate a build found in us-central1.
        mock_get_cloud_build_status_for_region.side_effect = [
            None,  # No builds found in 'global' region.
            CloudBuildStatus.SUCCESS,  # Successful build found in us-central1 region.
            None,  # No builds found in 'europe-west1' region.
        ]
        expected_status = CloudBuildStatus.SUCCESS
        build_status = get_cloud_build_status(
            project_id=project_id, image_name=image_name, tags=tags
        )
        self.assertEqual(build_status, expected_status)
        mock_list_available_regions.assert_called_once_with(project_id)

        # Assert that the function was called twice, once for global and once for us-central1.
        # Tests that the last region was short-circuited given that a build was found earlier
        calls = [
            mock.call(project_id=project_id, image_name=image_name, tags=tags, region="global"),
            mock.call(
                project_id=project_id, image_name=image_name, tags=tags, region="us-central1"
            ),
        ]
        mock_get_cloud_build_status_for_region.assert_has_calls(calls, any_order=False)

    @mock.patch("axlearn.cloud.gcp.cloud_build._get_cloud_build_status_for_region")
    @mock.patch("axlearn.cloud.gcp.cloud_build._list_available_regions")
    def test_get_cloud_build_status_failure_with_two_regions(
        self, mock_list_available_regions, mock_get_cloud_build_status_for_region
    ):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag1", "tag2"]

        mock_list_available_regions.return_value = ["us-central1", "europe-west1"]
        # Mock the return value for each region to simulate a build found in "europe-west1".
        mock_get_cloud_build_status_for_region.side_effect = [
            None,  # No builds found in 'global' region.
            None,  # No builds found in 'us-central1' region.
            CloudBuildStatus.FAILURE,  # Failed build found in 'europe-west1' region.
        ]
        expected_status = CloudBuildStatus.FAILURE
        build_status = get_cloud_build_status(
            project_id=project_id, image_name=image_name, tags=tags
        )
        self.assertEqual(build_status, expected_status)
        mock_list_available_regions.assert_called_once_with(project_id)

        # Assert that the function was called three times: once for global and once for each region.
        calls = [
            mock.call(project_id=project_id, image_name=image_name, tags=tags, region="global"),
            mock.call(
                project_id=project_id, image_name=image_name, tags=tags, region="us-central1"
            ),
            mock.call(
                project_id=project_id, image_name=image_name, tags=tags, region="europe-west1"
            ),
        ]
        mock_get_cloud_build_status_for_region.assert_has_calls(calls, any_order=False)

    @mock.patch("axlearn.cloud.gcp.cloud_build._get_cloud_build_status_for_region")
    @mock.patch("axlearn.cloud.gcp.cloud_build._list_available_regions")
    def test_get_cloud_build_status_success_with_no_regions(
        self, mock_list_available_regions, mock_get_cloud_build_status_for_region
    ):
        project_id = "test-project"
        image_name = "test-image"
        tags = ["tag1", "tag2"]

        mock_list_available_regions.return_value = []
        # Mock the return value for each region to simulate a build found in 'global' region.
        mock_get_cloud_build_status_for_region.side_effect = [
            CloudBuildStatus.SUCCESS,  # Successful build found in 'global' region.
        ]
        expected_status = CloudBuildStatus.SUCCESS
        build_status = get_cloud_build_status(
            project_id=project_id, image_name=image_name, tags=tags
        )
        self.assertEqual(build_status, expected_status)
        mock_list_available_regions.assert_called_once_with(project_id)

        # Assert that the function was called once for 'global' region.
        calls = [
            mock.call(project_id=project_id, image_name=image_name, tags=tags, region="global"),
        ]
        mock_get_cloud_build_status_for_region.assert_has_calls(calls)
