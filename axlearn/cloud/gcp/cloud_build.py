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
    """Constructs a filter string to query build requests based on image name and tags.

    To filter builds by multiple tags, use "AND", "OR", or "NOT" to list tags.
    Example: '(tags = tag1 AND tags = tag2) OR results.images.name="image"'.

    Args:
        image_name: The name of the image to filter build requests by.
        tags: A list of tags to filter build requests by.

    Returns:
        str: A filter string suitable for use in build request queries.
    """
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
        client = RegionsClient()
        request = ListRegionsRequest(project=project_id)
        regions = client.list(request=request)
        return [region.name for region in regions]
    except Exception as e:
        logging.error("Failed to look up regions for project '%s': %s", project_id, e)
        raise


def _list_builds_in_region(
    project_id: str, image_name: str, tags: tuple[str, ...], region: str
) -> list[Build]:
    """Lists all builds for a given combination of region, project, image name, and tags.

    Args:
        project_id: The GCP project ID.
        image_name: The name of the Docker image.
        tags: A tuple of build tags to filter the builds.
        region: The region to query for builds.

    Returns:
        A list of CloudBuild Builds matching the criteria.
    """
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ListBuildsRequest(
        parent=f"projects/{project_id}/locations/{region}",
        project_id=project_id,
        filter=_get_build_request_filter(image_name=image_name, tags=list(tags)),
    )
    return list(client.list_builds(request=request))


def _get_latest_build_status_in_region(
    project_id: str, image_name: str, tags: tuple[str, ...], region: str
) -> Optional[CloudBuildStatus]:
    """Gets the CloudBuild status for the latest build in a given region (no caching).

    Args:
        project_id: The GCP project ID.
        image_name: The name of the Docker image.
        tags: A tuple of build tags to filter the builds.
        region: The region to query for the latest build.

    Returns:
        The CloudBuildStatus of the latest build, or None if no build is found.
    """
    try:
        builds = _list_builds_in_region(
            project_id=project_id, image_name=image_name, tags=tags, region=region
        )
        if not builds:
            logging.info("No builds found in region '%s' for image '%s'.", region, image_name)
            return None

        # Sort builds by creation time and pick the latest.
        builds.sort(key=lambda build: build.create_time)
        latest_build = builds[-1]
        logging.info(
            "Latest build found in region '%s' for image '%s': %s", region, image_name, latest_build
        )
        return CloudBuildStatus.from_build_status(latest_build.status)

    except Exception as e:
        logging.warning(
            "Failed to find the build for image '%s' in region '%s', exception: %s",
            image_name,
            region,
            e,
        )
        raise


# In-memory memo to store the last known region for a given (project_id, image_name, tags).
_last_known_region_for_build = {}


def get_cloud_build_status(
    *, project_id: str, image_name: str, tags: list[str]
) -> Optional[CloudBuildStatus]:
    """Gets the status of the latest CloudBuild by filtering on the build tags and image name.

    In order:
    1. Queries the last known region where a build was previously found (if any).
    2. Queries all regions if not found above.

    The build results are not cached to ensure the latest build status is always retrieved.

    Args:
        project_id: The GCP project ID.
        image_name: The name of the image.
        tags: A list of tags used to filter the builds.

    Returns:
        The CloudBuildStatus of the latest build, or None if no build is found.
    """
    tags_tuple = tuple(sorted(tags))

    # If there is a last known region where a build was found previously, use it
    last_region = _last_known_region_for_build.get((project_id, image_name, tags_tuple))
    if last_region:
        logging.info("Checking last known region '%s' for image '%s'.", last_region, image_name)
        status = _get_latest_build_status_in_region(
            project_id=project_id, image_name=image_name, tags=tags_tuple, region=last_region
        )
        if status is not None:
            return status

    # If not found yet, iterate over all available regions.
    all_regions = ["global"] + _list_available_regions(project_id)
    for region in all_regions:
        logging.info(
            "Checking region '%s' for image '%s' in project '%s'.", region, image_name, project_id
        )
        status = _get_latest_build_status_in_region(
            project_id=project_id, image_name=image_name, tags=tags_tuple, region=region
        )
        if status is not None:
            _last_known_region_for_build[(project_id, image_name, tags_tuple)] = region
            return status

    # No build found in any region.
    return None
