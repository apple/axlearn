# Copyright © 2024 Apple Inc.

"""Tools to publish events."""

import logging
import os
import time
import uuid
from typing import Optional, Protocol

from absl import flags

try:
    import pika

    _PIKA_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    _PIKA_INSTALLED = False

from axlearn.common.config import REQUIRED, Configurable, Required, config_class

# Suppress Pika logs.
logging.getLogger("pika").setLevel(logging.WARNING)

# Configured key name of queue id.
CONFIGURED_KEY_JOB_EVENT_QUEUE_ID = "job_event_queue_id"
# Configured key name of queue connection host.
CONFIGURED_KEY_EVENT_QUEUE_HOST = "job_event_queue_host"
# Configured key name of queue connection port.
CONFIGURED_KEY_EVENT_QUEUE_PORT = "job_event_queue_port"
# Configured num of tries.
CONFIGURED_KEY_EVENT_QUEUE_NUM_TRIES = "job_event_queue_num_tries"
# Default RabbitMQ connection host.
DEFAULT_EVENT_QUEUE_HOST = "rabbitmq"
# Default RabbitMQ connection port.
DEFAULT_EVENT_QUEUE_PORT = 5672


class EventQueueInvalidCredentialsError(RuntimeError):
    """A non-recoverable error in EventQueue connection creation."""


class EventQueueConnectionError(RuntimeError):
    """A recoverable connection error in EventQueue connection creation."""


def is_publish_job_event_configured(job_queue_id: Optional[str]) -> bool:
    """Checks the config to see whether Event Publishing should be enabled."""
    return _PIKA_INSTALLED and job_queue_id is not None


class Event(Protocol):
    """An event that can be published via `BaseQueueClient.publish`."""

    def serialize(self) -> str:
        """Serializes the job lifecycle event into a JSON string."""
        raise NotImplementedError(type(self))


class BaseQueueClient(Configurable):
    """Interface for event queue management with retry and error handling logic."""

    @config_class
    class Config(Configurable.Config):
        """Configuration for BaseQueueClient."""

        # Queue id.
        queue_id: Required[str] = REQUIRED
        # Queue client host name. Default connection to use RabbitMQ solution.
        host: str = DEFAULT_EVENT_QUEUE_HOST
        # Queue client port name. Default connection to RabbitMQ port 5672.
        port: int = DEFAULT_EVENT_QUEUE_PORT
        # The number of attempts to publish an event.
        num_tries: int = 1

    def __init__(self, cfg: Config):
        """Init client instance."""
        super().__init__(cfg)
        cfg = self.config
        self._queue_id = cfg.queue_id
        self._host = cfg.host
        self._port = cfg.port
        self._num_tries = cfg.num_tries
        self._channel = None
        self._connection = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines absl flags to be read by `from_flags()`."""
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string(
            CONFIGURED_KEY_JOB_EVENT_QUEUE_ID, None, "Event Queue Id.", **common_kwargs
        )
        flags.DEFINE_string(
            CONFIGURED_KEY_EVENT_QUEUE_HOST,
            DEFAULT_EVENT_QUEUE_HOST,
            "Event queue connection host.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            CONFIGURED_KEY_EVENT_QUEUE_PORT,
            DEFAULT_EVENT_QUEUE_PORT,
            "Event queue connection port.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            CONFIGURED_KEY_EVENT_QUEUE_NUM_TRIES,
            1,
            "Event publish number of tries.",
            **common_kwargs,
        )

    def connect(self):
        """Connect to the message broker."""
        raise NotImplementedError(type(self))

    def close(self):
        """Close the connection to the message broker."""
        raise NotImplementedError(type(self))

    def publish(self, event: Event):
        """Publish an event to the specified queue with retry logic.

        Args:
            event: The event content to be published.
        """
        raise NotImplementedError(type(self))


class RabbitMQClient(BaseQueueClient):
    """Implementation of BaseQueueClient using RabbitMQ.

    This class provides methods to connect to a RabbitMQ server, publish messages to queues,
    and manage the RabbitMQ connection and channel. It is a concrete implementation of the
    `BaseQueueClient` interface.
    """

    def connect(self):
        """Establishes a connection to the RabbitMQ server."""
        # Secret is stored in the Env variable.
        user = os.getenv("RABBITMQ_USER", None)
        password = os.getenv("RABBITMQ_PASSWORD", None)
        if not user or not password:
            raise EventQueueInvalidCredentialsError(
                "The RabbitMQ connection username and password "
                "environment secrets are not configured."
            )

        try:
            credentials = pika.PlainCredentials(user, password)
            parameters = pika.ConnectionParameters(
                host=self._host,
                port=self._port,
                credentials=credentials,
                # Heartbeat interval to keep the connection alive.
                heartbeat=600,
                # Timeout for blocked connection.
                blocked_connection_timeout=300,
            )
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            logging.info(
                "Connected RabbitMQ at host: %s, port: %s to publish events.",
                self._host,
                self._port,
            )
        except pika.exceptions.ProbableAuthenticationError as e:
            raise EventQueueInvalidCredentialsError(
                "Authentication with RabbitMQ failed. Please check your username and password."
            ) from e
        except pika.exceptions.AMQPConnectionError as e:
            raise EventQueueConnectionError(
                "An unexpected error occurred while connecting to RabbitMQ."
            ) from e
        except ConnectionResetError as e:
            raise e

    def close(self):
        """Closes the RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            self._connection.close()
            logging.info("Closed RabbitMQ connection.")
        if self._channel and not self._channel.is_closed:
            self._channel.close()
            logging.info("Closed RabbitMQ channel.")
        self._connection = None
        self._channel = None

    def publish(self, event: Event):
        """Publishes an event to the queue."""
        logging.info("Publishing event: %s", event)
        message = event.serialize()
        attempt = 0
        while attempt <= self._num_tries:
            try:
                # Ensure connection is established before publishing.
                if not self._channel or not self._connection:
                    logging.error("RabbitMQ publisher channel is closed, reconnecting...")
                    self.connect()

                # Setting durable=True ensures that the queue will survive.
                # a RabbitMQ server restart.
                self._channel.queue_declare(queue=self._queue_id, durable=True)
                self._channel.basic_publish(
                    # Set exchange empty refers to the default exchange,
                    # which directly routes messages to the queue specified by the routing_key
                    exchange="",
                    routing_key=self._queue_id,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        # Generate a unique correlation ID.
                        correlation_id=str(uuid.uuid4()),
                    ),
                )
                logging.debug("Published event in queue: %s. message: %s", self._queue_id, message)
                return
            except EventQueueInvalidCredentialsError as e:
                # Throws for un-recoverable errors.
                raise e
            except (EventQueueConnectionError, ConnectionResetError) as e:
                # Only retry on recoverable exceptions.
                # AMQPConnectionError is assumed to be related to network issues,
                # or temporary unavailable host.
                logging.error(
                    "Failed to publish event: %s. Error: %s. Attempt: %d",
                    message,
                    str(e),
                    attempt,
                )
                self._handle_publish_error()
                attempt += 1
                if attempt <= self._num_tries:
                    time.sleep(2**attempt)
            except Exception as e:  # pylint: disable=broad-except
                # Unknown errors. Don't retry. Log to avoid crashing clients.
                logging.error(
                    "Unknown error. Failed to publish event: %s. Error: %s.", message, str(e)
                )
                self._handle_publish_error()
                attempt += 1
                if attempt <= self._num_tries:
                    time.sleep(2**attempt)

    def _handle_publish_error(self):
        """Handle publish errors with retrying on connection issue."""
        try:
            self.close()
        # pylint: disable=broad-exception-caught.
        except Exception as handling_error:
            logging.error(
                "RabbitMQ event client failed in _handle_publish_error: %s", str(handling_error)
            )
