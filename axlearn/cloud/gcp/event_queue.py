# Copyright © 2024 Apple Inc.

"""Tools to publish events in GCP."""

from typing import Optional

from absl import flags

from axlearn.cloud.common.event_queue import (
    CONFIGURED_KEY_EVENT_QUEUE_HOST,
    CONFIGURED_KEY_EVENT_QUEUE_NUM_TRIES,
    CONFIGURED_KEY_EVENT_QUEUE_PORT,
    CONFIGURED_KEY_JOB_EVENT_QUEUE_ID,
    DEFAULT_EVENT_QUEUE_HOST,
    DEFAULT_EVENT_QUEUE_PORT,
    BaseQueueClient,
    RabbitMQClient,
    is_publish_job_event_configured,
)
from axlearn.cloud.gcp.config import gcp_settings


def event_queue_from_config(
    flag_values: flags.FlagValues = flags.FLAGS,
) -> Optional[BaseQueueClient.Config]:
    """Create config for EventQueue.

    Args:
        flag_values: Flag configurations defined in gcp_settings.

    Returns:
        A configured `RabbitMQClient.Config` object if the event queue is configured.
        Returns `None` if the event queue is not configured.
    """
    if not is_publish_job_event_configured(
        gcp_settings(CONFIGURED_KEY_JOB_EVENT_QUEUE_ID, required=False, fv=flag_values)
    ):
        return None

    return RabbitMQClient.default_config().set(
        host=gcp_settings(
            CONFIGURED_KEY_EVENT_QUEUE_HOST,
            required=False,
            default=DEFAULT_EVENT_QUEUE_HOST,
            fv=flag_values,
        ),
        port=gcp_settings(
            CONFIGURED_KEY_EVENT_QUEUE_PORT,
            required=False,
            default=DEFAULT_EVENT_QUEUE_PORT,
            fv=flag_values,
        ),
        queue_id=gcp_settings(CONFIGURED_KEY_JOB_EVENT_QUEUE_ID, required=False, fv=flag_values),
        num_tries=gcp_settings(
            CONFIGURED_KEY_EVENT_QUEUE_NUM_TRIES, required=False, default=1, fv=flag_values
        ),
    )
