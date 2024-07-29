# Copyright © 2024 Apple Inc.

"""Cloud build utilities"""
import enum
from typing import Optional

from absl import logging
from google.cloud.devtools import cloudbuild_v1


class CloudBuildStatus(enum.Enum):
    """Cloud Build Status.

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
    def from_build_status(cls, build_status):
        return cls(build_status)

    @classmethod
    def is_success(cls, build_status):
        return build_status == cls.SUCCESS

    @classmethod
    def is_failed(cls, build_status):
        return build_status in {
            cls.FAILURE,
            cls.INTERNAL_ERROR,
            cls.TIMEOUT,
            cls.CANCELLED,
            cls.EXPIRED,
            cls.STATUS_UNKNOWN,
        }

    @classmethod
    def is_pending(cls, build_status):
        return build_status in {cls.PENDING, cls.QUEUED, cls.WORKING}


def get_cloud_build_status(project_id: str, image_name: str) -> Optional[CloudBuildStatus]:
    """Get the status of the latest build filter on the image_name.

    Returns:
        Cloud build status for the latest build if exist.
        None if no build found for the image name.

    Raises:
        Error: If fails to get the latest build status of a given image in a GCP project.
    """
    try:
        client = cloudbuild_v1.CloudBuildClient()
        request = cloudbuild_v1.ListBuildsRequest(
            project_id=project_id,
            filter=f'results.images.name="{image_name}"',
        )

        builds = list(client.list_builds(request=request))
        if not builds:
            logging.error("No builds found for image name: %s.", image_name)
            return None

        builds.sort(key=lambda build: build.create_time)
        logging.debug("Builds for image %s: %s", image_name, builds)

        latest_build = builds[-1]
        return CloudBuildStatus.from_build_status(latest_build.status)

    # TODO(liang-he): Distinguish retryable and non-retryable google.api_core.exceptions
    except Exception as e:
        logging.warning("Failed to find the build for image %s, exception: %s", image_name, e)
        raise
