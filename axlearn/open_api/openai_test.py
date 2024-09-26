# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for openai.py."""
import asyncio
import json
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

_module_root = "axlearn"


# pylint: disable=wrong-import-position
from axlearn.open_api.common import ClientRateLimitError, Generator, ValidationError
from axlearn.open_api.openai import OpenAIClient

# pylint: enable=wrong-import-position


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    """Unit tests for class OpenAIClient."""

    def setUp(self):
        self.client: OpenAIClient = (
            OpenAIClient.default_config().set(model="gpt-3.5-turbo").instantiate()
        )
        self.client._client = AsyncMock()

    async def test_async_generate_raises_validation_error(self):
        request = {}
        with self.assertRaises(ValidationError) as context:
            await self.client.async_generate(request=request)

        self.assertEqual(str(context.exception), "Both prompt and messages are None.")

    async def test_generate_with_messages_tools(self):
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": ["tool_1", "tool_2"],
        }

        mock_chat_completion = MagicMock()
        mock_response = '{"id": "chat_test", "choices": []}'
        mock_chat_completion.model_dump_json.return_value = mock_response
        self.client._client.chat.completions.create.return_value = mock_chat_completion

        result = await self.client.async_generate(request=request)

        self.client._client.chat.completions.create.assert_awaited_once_with(
            messages=request["messages"],
            extra_body=self.client.config.extra_body,
            tools=["tool_1", "tool_2"],
        )
        self.assertEqual(result, mock_response)

    async def test_generate_with_messages(self):
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        mock_chat_completion = MagicMock()
        mock_chat_completion.model_dump_json.return_value = '{"id": "chat_test", "choices": []}'
        self.client._client.chat.completions.create.return_value = mock_chat_completion

        await self.client.async_generate(request=request)

        self.client._client.chat.completions.create.assert_awaited_once_with(
            messages=request["messages"],
            extra_body=self.client.config.extra_body,
        )


class TestOpenAIAsyncGenerateFromRequests(unittest.IsolatedAsyncioTestCase):
    """Unit test for async_generate_from_requests."""

    def setUp(self):
        cfg: Generator.Config = Generator.default_config().set(
            concurrency=2,
            max_non_rate_limit_retries=3,
            client=OpenAIClient.default_config().set(model="gpt-3.5-turbo"),
        )
        open_api: Generator = cfg.instantiate()
        self.open_api = open_api
        # Mock clients for concurrency.
        self.open_api._clients = [MagicMock(), MagicMock()]

    @patch(
        f"{_module_root}.open_api.common.Generator._async_generate_from_request",
        new_callable=AsyncMock,
    )
    async def test_async_generate_from_requests(self, mock_async_generate_from_request):
        # Mock async_generate_from_request to return a predefined response.
        mock_response = {"response": "test response", "async_index": 1}
        mock_async_generate_from_request.return_value = mock_response

        # Create mock requests.
        mock_requests = [
            {"prompt": "How are you?"},
            {"prompt": "Tell me a joke."},
            {"prompt": "What is the weather today?"},
        ]

        # Call the method under test.
        responses = await self.open_api.async_generate_from_requests(mock_requests)

        # Assertions to verify behavior.
        self.assertEqual(len(responses), len(mock_requests))  # Ensure all requests are processed
        for response in responses:
            self.assertIn("response", response)  # Check structure of the response
            self.assertEqual(response["response"], "test response")  # Validate response content

        # Check if async_generate_from_request is called correctly.
        calls = [
            unittest.mock.call(client=ANY, request=ANY, **self.open_api.config.decode_parameters)
            for _ in range(len(mock_requests))
        ]
        mock_async_generate_from_request.assert_has_calls(
            calls, any_order=True
        )  # Ensure each request is processed


class TestOpenAI(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the OpenAIClient class focusing
    on the async_generate_from_request method."""

    async def asyncSetUp(self):
        """Set up resources prior to each test."""
        cfg: Generator.Config = Generator.default_config().set(
            concurrency=1,
            max_non_rate_limit_retries=3,
            client=OpenAIClient.default_config().set(model="gpt-3.5-turbo"),
        )
        open_api: Generator = cfg.instantiate()
        # Mock clients for concurrency.
        open_api._clients = [AsyncMock()]
        open_api._semaphore = asyncio.Semaphore(1)
        self.open_api = open_api
        self.request = {"messages": [{"text": "Hello"}], "prompt": "Test prompt"}

    async def test_async_generate_from_request_success(self):
        """Test that async_generate_from_request returns a successful response."""
        self.open_api._clients[0].async_generate = AsyncMock(return_value="success response")
        result = await self.open_api._async_generate_from_request(
            self.open_api._clients[0], request=self.request
        )
        self.assertEqual(result["response"], "success response")

    async def test_async_generate_from_request_failure_with_retries(self):
        """Test retries on failure that is not due to rate limiting."""
        self.open_api._clients[0].async_generate = AsyncMock(
            side_effect=[Exception("Error"), "success response"]
        )
        result = await self.open_api._async_generate_from_request(
            self.open_api._clients[0], request=self.request
        )
        self.assertEqual(result["response"], "success response")
        self.assertEqual(self.open_api._clients[0].async_generate.call_count, 2)

    async def test_async_generate_from_request_handles_rate_limiting(self):
        """Test handling of rate limiting by pausing and retrying."""
        self.open_api._clients[0].async_generate = AsyncMock(
            side_effect=[ClientRateLimitError("Rate limiting"), "success response"]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await self.open_api._async_generate_from_request(
                self.open_api._clients[0], request=self.request
            )
            mock_sleep.assert_awaited()
            self.assertEqual(result["response"], "success response")

    async def test_async_generate_from_request_exceeds_max_retries(self):
        """Test the behavior when the number of retries exceeds the max not allowed."""
        cfg: Generator.Config = Generator.default_config().set(
            concurrency=1,
            max_non_rate_limit_retries=3,
            allow_non_rate_limit_error=False,
            client=OpenAIClient.default_config().set(model="gpt-3.5-turbo"),
        )
        open_api: Generator = cfg.instantiate()
        open_api = cfg.instantiate()
        # Mock clients for concurrency.
        open_api._clients = [MagicMock()]
        open_api._clients[0].async_generate = AsyncMock(side_effect=[Exception("Error")])
        open_api._semaphore = asyncio.Semaphore(1)
        with self.assertRaises(ValueError):
            await open_api._async_generate_from_request(open_api._clients[0], request=self.request)
            self.assertEqual(open_api._clients[0].async_generate.call_count, 1)

    async def test_async_generate_from_request_exceeds_allowed_max_retries(self):
        """Test the behavior when the number of retries exceeds the max allowed."""
        self.open_api._clients[0].async_generate = AsyncMock(side_effect=Exception("Error"))
        result = await self.open_api._async_generate_from_request(
            self.open_api._clients[0], request=self.request
        )
        self.assertEqual(
            result["response"],
            json.dumps({"error": "Exceed max retries of non rate limiting error."}),
        )
        self.assertEqual(self.open_api._clients[0].async_generate.call_count, 3)
