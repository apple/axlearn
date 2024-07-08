# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for generator.py."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from absl import flags

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

_module_root = "axlearn"


# pylint: disable=wrong-import-position
from axlearn.open_api.common import Generator
from axlearn.open_api.generator import generate_from_requests
from axlearn.open_api.openai import OpenAIClient

# pylint: enable=wrong-import-position


class TestGenerateFromRequests(unittest.IsolatedAsyncioTestCase):
    """Unit test for generate_from_requests."""

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
            {"messages": [{"role": "user", "content": "How are you?"}]},
            {"messages": [{"role": "user", "content": "Tell me a joke."}]},
            {"messages": [{"role": "user", "content": "What is the weather today?"}]},
        ]

        fv = flags.FlagValues()
        Generator.define_flags(fv)
        fv.set_default("model", "test")
        fv.mark_as_parsed()

        # Call the method under test.
        responses = await generate_from_requests(gen_requests=mock_requests, fv=fv)

        # Assertions to verify behavior.
        self.assertEqual(len(responses), len(mock_requests))
        for response in responses:
            self.assertIn("response", response)
            self.assertEqual(response["response"], "test response")
