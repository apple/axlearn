# Copyright © 2024 Apple Inc.

"""CloudBuild utilities"""

import enum
from typing import Optional

from absl import logging
from google.cloud.compute_v1 import ListRegionsRequest, RegionsClient
from google.cloud.devtools import cloudbuild_v1
from google.cloud.devtools.cloudbuild_v1.types import Build


class CloudBuildStatus(enum.Enum):
    """CloudBuild Status.

    See also:
    https://cloud.google.com/python/docs/reference/cloudbuild/latest/google.cloud.devtools.cloudbuild_v1.types.Build.Status

    Attributes:
        STATUS_UNKNOWN (0): Status of the build is unknown.
        QUEUED (1): Build or step is queued; work has not yet begun.
        WORKING (2): Build or step is being executed.
        SUCCESS (3): Build or step finished successfully.
        FAILURE (4): Build or step failed to complete successfully.
        INTERNAL_ERROR (5): Build or step failed due to an internal cause.
        TIMEOUT (6): Build or step took longer than was allowed.
        CANCELLED (7): Build or step was canceled by a user.
        EXPIRED (9): Build was enqueued for longer than the value of queue_ttl.
        PENDING (10): Build has been created and is pending execution and queuing.
                      It has not been queued.
    """

    STATUS_UNKNOWN = 0
    QUEUED = 1
    WORKING = 2
    SUCCESS = 3
    FAILURE = 4
    INTERNAL_ERROR = 5
    TIMEOUT = 6
    CANCELLED = 7
    EXPIRED = 9
    PENDING = 10

    @classmethod
    def from_build_status(cls, build_status: Build.Status) -> "CloudBuildStatus":
        return cls(build_status)

    def is_success(self) -> bool:
        return self == CloudBuildStatus.SUCCESS

    def is_failed(self) -> bool:
        return self in {
            CloudBuildStatus.FAILURE,
            CloudBuildStatus.INTERNAL_ERROR,
            CloudBuildStatus.TIMEOUT,
            CloudBuildStatus.CANCELLED,
            CloudBuildStatus.EXPIRED,
            CloudBuildStatus.STATUS_UNKNOWN,
        }

    def is_pending(self) -> bool:
        return self in {CloudBuildStatus.PENDING, CloudBuildStatus.QUEUED, CloudBuildStatus.WORKING}


def _get_build_request_filter(*, image_name: str, tags: list[str]) -> str:
    # To filter builds by multiple tags, use "AND", "OR", or "NOT" to list tags.
    # Example: '(tags = tag1 AND tags = tag2) OR results.images.name="image"'.
    filter_by_tag = ""
    if tags:
        filter_by_tag = "(" + " AND ".join(f"tags = {tag}" for tag in tags) + ")" + " OR "
    filter_by_image = f'results.images.name="{image_name}"'
    return filter_by_tag + filter_by_image


def _list_available_regions(project_id: str) -> list[str]:
    """Retrieves all available regions/locations for the given project using the Compute Engine API.

    Args:
        project_id: The GCP project ID.

    Returns:
        A list of regions/locations for the given project as strings.

    Raises:
        Exception: If an error occurs when retrieving regions from the Compute Engine API.
    """
    try:
        # Initialize the Compute Engine client.
        client = RegionsClient()

        # List all regions for the given project.
        request = ListRegionsRequest(project=project_id)
        regions = client.list(request=request)

        # Extract and return region names as list.
        return [region.name for region in regions]
    except Exception as e:
        logging.error("Failed to look up regions for project: %s", e)
        raise


def _get_cloud_build_status_for_region(
    *, project_id: str, image_name: str, tags: list[str], region: str = "global"
) -> Optional[CloudBuildStatus]:
    """Gets the status of the latest build by filtering on the build tags, image name, and region.

    Args:
        project_id: The GCP project ID.
        region: The GCP region. Defaults to 'global' if no region is given.
        image_name: The image name including the image path of the Artifact Registry.
        tags: A list of the CloudBuild build tags. Note that these are not docker image tags.

    Returns:
        CloudBuild status for the latest build in this region.
        None if no build found for the image name in the given region.

    Raises:
        Exception: On failure to get the latest build status of a given image in a GCP project.
    """
    try:
        client = cloudbuild_v1.CloudBuildClient()
        request = cloudbuild_v1.ListBuildsRequest(
            # CloudBuild lookups are region-specific.
            parent=f"projects/{project_id}/locations/{region}",
            project_id=project_id,
            filter=_get_build_request_filter(image_name=image_name, tags=tags),
        )
        builds = list(client.list_builds(request=request))

        if not builds:
            logging.warning("No builds found in region '%s' for image '%s'", image_name, region)
            return None

        builds.sort(key=lambda build: build.create_time)
        logging.info("Build found in region '%s' for image '%s': %s", region, image_name, builds)

        latest_build = builds[-1]
        return CloudBuildStatus.from_build_status(latest_build.status)

    # TODO(liang-he): Distinguish retryable and non-retryable google.api_core.exceptions
    except Exception as e:
        logging.warning(
            "Failed to find the build for image '%s' in region '%s', exception: %s",
            image_name,
            region,
            e,
        )
        raise


def get_cloud_build_status(
    *, project_id: str, image_name: str, tags: list[str]
) -> Optional[CloudBuildStatus]:
    """Gets the status of the latest CloudBuild by filtering on the build tags and image name.

    Performs a request for each available region, including 'global' first.

    Args:
        project_id: The GCP project ID.
        image_name: The image name including the image path of the Artifact Registry.
        tags: A list of the CloudBuild build tags. Note that these are not docker image tags.

    Returns:
        CloudBuild status for the latest build found in the first available region.
        None if no build found for the image name and tag across all available regions.
    """
    build_status = None
    # Unfortunately the CloudBuild API does not support wildcard region lookup.
    # Workaround: Check each region for the latest build, stopping when the first is found.
    # Try global (default) region first before other regions.
    all_regions = ["global"] + _list_available_regions(project_id)
    for region in all_regions:
        logging.info("Looking for CloudBuild with image '%s' in region '%s'", image_name, region)
        build_status = _get_cloud_build_status_for_region(
            project_id=project_id, image_name=image_name, tags=tags, region=region
        )
        if build_status is not None:
            # Short-circuit so there are no extraneous queries after the first build is found.
            break
    return build_status
