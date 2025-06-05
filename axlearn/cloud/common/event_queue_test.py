# Copyright © 2024 Apple Inc.

"""Tests event queue."""
# pylint: disable=redefined-outer-name, protected-access

import os
import uuid
from unittest import TestCase, mock
from unittest.mock import MagicMock

import pika

from axlearn.cloud.common.event_queue import (
    Event,
    EventQueueConnectionError,
    EventQueueInvalidCredentialsError,
    RabbitMQClient,
)


class TestRabbitMQClient(TestCase):
    """Tests for the RabbitMQClient class."""

    @mock.patch("pika.BlockingConnection")
    @mock.patch("pika.BasicProperties")
    @mock.patch("os.getenv")
    def test_publish_success(self, mock_getenv, mock_basic_properties, mock_block_connection):
        """Test successful message publishing."""
        # Mock environment variables for RabbitMQ credentials
        mock_getenv.side_effect = lambda key, default=None: {
            "RABBITMQ_USER": "test_user",
            "RABBITMQ_PASSWORD": "test_password",
        }.get(key, default)

        # Mock the PlainCredentials and ConnectionParameters
        mock_uuid = str(uuid.uuid4())
        mock_basic_properties.return_value = mock.Mock(delivery_mode=2, correlation_id=mock_uuid)
        mock_connection = mock_block_connection.return_value
        mock_channel = mock_connection.channel.return_value
        mock_event = MagicMock(spec=Event)
        mock_event.serialize.return_value = "serialized_event"
        rabbitmq_client = (
            RabbitMQClient.default_config()
            .set(host="rabbitmq", port=5672, queue_id="test_queue")
            .instantiate()
        )
        rabbitmq_client.publish(mock_event)
        mock_channel.queue_declare.assert_called_once_with(queue="test_queue", durable=True)
        mock_channel.basic_publish.assert_called_once_with(
            exchange="",
            routing_key="test_queue",
            body="serialized_event",
            properties=mock_basic_properties.return_value,
        )

    @mock.patch.dict(os.environ, {"RABBITMQ_USER": "test_user", "RABBITMQ_PASSWORD": "test_pass"})
    @mock.patch("pika.BlockingConnection")
    def test_connect_success(self, mock_blocking_connection):
        # Arrange
        mock_channel = MagicMock()
        mock_blocking_connection.return_value.channel.return_value = mock_channel
        rabbitmq_client = (
            RabbitMQClient.default_config()
            .set(host="rabbitmq", port=5672, queue_id="test_queue")
            .instantiate()
        )

        rabbitmq_client.connect()
        mock_blocking_connection.assert_called_once()
        self.assertIsNotNone(rabbitmq_client._connection)
        self.assertIsNotNone(rabbitmq_client._channel)

    @mock.patch.dict(os.environ, {"RABBITMQ_USER": "", "RABBITMQ_PASSWORD": ""})
    def test_connect_missing_credentials(self):
        rabbitmq_client = (
            RabbitMQClient.default_config()
            .set(host="rabbitmq", port=5672, queue_id="test_queue")
            .instantiate()
        )

        with self.assertRaises(EventQueueInvalidCredentialsError):
            rabbitmq_client.connect()

    @mock.patch.dict(os.environ, {"RABBITMQ_USER": "test_user", "RABBITMQ_PASSWORD": "test_pass"})
    @mock.patch("pika.BlockingConnection", side_effect=pika.exceptions.AMQPConnectionError)
    def test_connect_amqp_connection_error(self, mock_blocking_connection):
        rabbitmq_client = (
            RabbitMQClient.default_config()
            .set(host="rabbitmq", port=5672, queue_id="test_queue")
            .instantiate()
        )
        mock_blocking_connection.assert_not_called()

        with self.assertRaises(EventQueueConnectionError):
            rabbitmq_client.connect()
