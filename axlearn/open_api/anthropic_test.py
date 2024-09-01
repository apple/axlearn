# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for anthropic.py."""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from axlearn.open_api.mock_utils import mock_anthropic_package, mock_openai_package

mock_openai_package()
mock_anthropic_package()

_module_root = "axlearn"


# pylint: disable=wrong-import-position
from axlearn.open_api.anthropic import (
    AnthropicClient,
    _convert_openai_messages_to_anthropic,
    _convert_openai_tools_to_anthropic,
    _system_parallel_tools_prompt,
)
from axlearn.open_api.common import ValidationError

# pylint: enable=wrong-import-position


class TestAnthropicClient(unittest.IsolatedAsyncioTestCase):
    """Unit test for class AnthropicClient."""

    def _create_anthropic_client(self) -> AnthropicClient:
        client = (
            AnthropicClient.default_config()
            .set(model="claude", extra_body={"add_system_parallel_tools": True})
            .instantiate()
        )
        client._client = AsyncMock()
        return client

    @patch(f"{_module_root}.open_api.anthropic._convert_openai_messages_to_anthropic")
    @patch(f"{_module_root}.open_api.anthropic._convert_openai_tools_to_anthropic")
    async def test_async_generate(self, mock_convert_tools, mock_convert_messages):
        mock_convert_messages.return_value = "converted_messages"
        mock_convert_tools.return_value = "converted_tools"
        client = self._create_anthropic_client()
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = json.dumps({"response": "test_response"})
        client._client.messages.create = AsyncMock(return_value=mock_response)
        result = await client.async_generate(
            request={
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"name": "tool1"}],
            },
        )
        # Assert the expected result.
        self.assertEqual(result, json.dumps({"response": "test_response"}))

        # Assert the expected calls.
        mock_convert_messages.assert_called_with(messages=[{"role": "user", "content": "Hello"}])
        mock_convert_tools.assert_called_with(tools=[{"name": "tool1"}])
        client._client.messages.create.assert_called_with(
            messages="converted_messages",
            tools="converted_tools",
            extra_body={},
            system=_system_parallel_tools_prompt,
        )


class TestConvertOpenAIMessagesToAnthropic(unittest.TestCase):
    """Unit tests for _convert_openai_messages_to_anthropic."""

    def test_convert_user_text_message(self):
        messages = [{"role": "user", "content": "Hello, how can I help you?"}]

        result = _convert_openai_messages_to_anthropic(messages)
        self.assertEqual(result, [{"role": "user", "content": "Hello, how can I help you?"}])

    def test_convert_user_multimodal_message(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            }
        ]

        expected_result = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    },
                ],
            }
        ]

        result = _convert_openai_messages_to_anthropic(messages)
        self.assertEqual(result, expected_result)

    def test_convert_assistant_message_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "Sure, I can help with that.",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "tool1", "arguments": '{"arg1": "value1"}'},
                    }
                ],
            }
        ]

        expected_result = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure, I can help with that."},
                    {
                        "type": "tool_use",
                        "id": "1",
                        "name": "tool1",
                        "input": {"arg1": "value1"},
                    },
                ],
            }
        ]

        result = _convert_openai_messages_to_anthropic(messages)
        self.assertEqual(result, expected_result)

    def test_convert_tool_message(self):
        messages = [
            {
                "role": "tool",
                "content": "tool response",
                "tool_call_id": "1",
            }
        ]

        expected_result = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": "tool response",
                    }
                ],
            }
        ]

        result = _convert_openai_messages_to_anthropic(messages)
        self.assertEqual(result, expected_result)

    def test_invalid_content_type(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "unknown_type", "text": "This should raise an error."}],
            }
        ]

        with self.assertRaises(ValidationError):
            _convert_openai_messages_to_anthropic(messages)

    def test_no_tool_calls_in_assistant_message(self):
        messages = [{"role": "assistant", "content": "Here is your answer."}]

        result = _convert_openai_messages_to_anthropic(messages)
        self.assertEqual(result, [{"role": "assistant", "content": "Here is your answer."}])


class TestConvertOpenAIToolsToAnthropic(unittest.TestCase):
    """Unit tests for _convert_openai_tools_to_anthropic."""

    def test_convert_single_tool(self):
        tools = [
            {
                "function": {
                    "name": "tool1",
                    "description": "A test tool",
                    "parameters": {"param1": "value1"},
                }
            }
        ]

        expected_result = [
            {
                "name": "tool1",
                "description": "A test tool",
                "input_schema": {"param1": "value1"},
            }
        ]

        result = _convert_openai_tools_to_anthropic(tools)
        self.assertEqual(result, expected_result)

    def test_convert_multiple_tools(self):
        tools = [
            {
                "function": {
                    "name": "tool1",
                    "description": "First test tool",
                    "parameters": {"param1": "value1"},
                }
            },
            {
                "function": {
                    "name": "tool2",
                    "description": "Second test tool",
                    "parameters": {"param2": "value2"},
                }
            },
        ]

        expected_result = [
            {
                "name": "tool1",
                "description": "First test tool",
                "input_schema": {"param1": "value1"},
            },
            {
                "name": "tool2",
                "description": "Second test tool",
                "input_schema": {"param2": "value2"},
            },
        ]

        result = _convert_openai_tools_to_anthropic(tools)
        self.assertEqual(result, expected_result)

    def test_convert_tool_with_no_parameters(self):
        tools = [
            {
                "function": {
                    "name": "tool1",
                    "description": "A test tool with no parameters",
                    "parameters": {},
                }
            }
        ]

        expected_result = [
            {
                "name": "tool1",
                "description": "A test tool with no parameters",
                "input_schema": {},
            }
        ]

        result = _convert_openai_tools_to_anthropic(tools)
        self.assertEqual(result, expected_result)

    def test_convert_empty_tools_list(self):
        tools = []

        expected_result = []

        result = _convert_openai_tools_to_anthropic(tools)
        self.assertEqual(result, expected_result)
