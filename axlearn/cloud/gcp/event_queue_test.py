# Copyright © 2024 Apple Inc.

"""Test event queue publishing in GCP."""

import os
from unittest import mock

from absl import flags

from axlearn.cloud.common import config
from axlearn.cloud.common.config_test import _setup_fake_repo, create_default_config
from axlearn.cloud.gcp import config as gcp_config
from axlearn.cloud.gcp.event_queue import event_queue_from_config
from axlearn.common.test_utils import TestWithTemporaryCWD


class EventQueueTest(TestWithTemporaryCWD):
    """Test Event Queue functions."""

    def run(self, result=None):
        # Run tests under mock env.
        with mock.patch("os.environ", {"RABBITMQ_USER": "test", "RABBITMQ_PASSWORD": "test"}):
            return super().run(result)

    def test_event_queue_from_config(self):
        temp_dir = os.path.realpath(self._temp_root.name)
        _setup_fake_repo(temp_dir)

        flag_values = flags.FlagValues()
        flags.DEFINE_string("project", None, "The project name.", flag_values=flag_values)
        flags.DEFINE_string("zone", None, "The zone name.", flag_values=flag_values)
        flag_values.project = "test"
        flag_values.zone = "test"
        flag_values.mark_as_parsed()

        # No value is configured for job_event_queue_id.
        base_queue_client_config = event_queue_from_config(flag_values)
        self.assertIsNone(base_queue_client_config)

        # Create a default config, which should get picked up.
        default_config = create_default_config(temp_dir)

        # Write job_event_queue_id to the config.
        config.write_configs_with_header(
            str(default_config),
            {gcp_config.CONFIG_NAMESPACE: {"test:test": {"job_event_queue_id": "test-queue"}}},
        )

        base_queue_client_config = event_queue_from_config(flag_values)
        self.assertIsNotNone(base_queue_client_config)
        self.assertEqual(base_queue_client_config.queue_id, "test-queue")
        # These fields are set as default values.
        self.assertEqual(base_queue_client_config.host, "rabbitmq")
        self.assertEqual(base_queue_client_config.port, 5672)
        self.assertEqual(base_queue_client_config.num_tries, 1)

        # Write job_event_queue_id to the config.
        config.write_configs_with_header(
            str(default_config),
            {
                gcp_config.CONFIG_NAMESPACE: {
                    "test:test": {
                        "job_event_queue_id": "test-queue",
                        "job_event_queue_host": "test-host",
                        "job_event_queue_port": 8000,
                        "job_event_queue_num_tries": 2,
                    }
                }
            },
        )

        base_queue_client_config = event_queue_from_config(flag_values)
        self.assertIsNotNone(base_queue_client_config)
        self.assertEqual(base_queue_client_config.queue_id, "test-queue")
        self.assertEqual(base_queue_client_config.host, "test-host")
        self.assertEqual(base_queue_client_config.port, 8000)
        self.assertEqual(base_queue_client_config.num_tries, 2)
