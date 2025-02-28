# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for gemini.py"""
import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from axlearn.open_api.mock_utils import mock_openai_package, mock_vertexai_package

mock_openai_package()
mock_vertexai_package()


_module_root = "axlearn"


# pylint: disable=wrong-import-position
from axlearn.open_api.common import ValidationError
from axlearn.open_api.gemini import (
    GeminiClient,
    _convert_openai_messages_to_gemini,
    _format_tool_message,
)

# pylint: enable=wrong-import-position


class TestGeminiClient(unittest.IsolatedAsyncioTestCase):
    """Unit test for class GeminiClient."""

    @patch(f"{_module_root}.open_api.gemini._init_vertexai")
    def _create_gemini_client(
        self,
        mock_init_vertexai=None,
    ) -> GeminiClient:
        mock_init_vertexai.return_value = AsyncMock()
        client: GeminiClient = (
            GeminiClient.default_config().set(model="gemini-1.0-pro").instantiate()
        )
        client._client = AsyncMock()
        return client

    @patch(f"{_module_root}.open_api.gemini._convert_openai_messages_to_gemini")
    @patch(f"{_module_root}.open_api.gemini._convert_openai_tools_to_gemini")
    async def test_async_generate(self, mock_convert_tools, mock_convert_messages):
        mock_convert_messages.return_value = "converted_messages"
        mock_convert_tools.return_value = "converted_tools"
        client = self._create_gemini_client()
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"response": "test_response"}
        client._client.generate_content_async = AsyncMock(return_value=mock_response)

        result = await client.async_generate(
            request={
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": {"name": "func1"}}],
            },
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            max_tokens=100,
            stop_sequences=["\n"],
        )
        # Assert the expected result.
        self.assertEqual(result, json.dumps({"response": "test_response"}))

    def test_format_tool_message(self):
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "long_name" * 32}}]}
        processed_message = _format_tool_message(message)
        self.assertTrue(len(processed_message["tool_calls"][0]["function"]["name"]) <= 32)


class TestConvertOpenAIMessagesToGemini(unittest.TestCase):
    """Unit tests for _convert_openai_messages_to_gemini."""

    def setUp(self):
        # Reset mocks before each test.
        self.mock_part = sys.modules["vertexai.generative_models"].Part
        self.mock_part.reset_mock()

        self.mock_content = sys.modules["vertexai.generative_models"].Content
        self.mock_content.reset_mock()

    def test_convert_user_text_message(self):
        messages = [{"role": "user", "content": "Hello, how can I help you?"}]

        part_instance = MagicMock()
        self.mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        self.mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)
        # mock_part
        self.mock_part.from_text.assert_called_with("Hello, how can I help you?")
        self.mock_content.assert_called_once()
        self.mock_content.assert_called_with(role="user", parts=[part_instance])
        self.assertEqual(result, [content_instance])

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

        text_part_instance = MagicMock()
        image_part_instance = MagicMock()
        self.mock_part.from_text.return_value = text_part_instance
        self.mock_part.from_data.return_value = image_part_instance

        content_instance = MagicMock()
        self.mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        self.mock_part.from_text.assert_called_with("Look at this image:")
        self.mock_part.from_data.assert_called_with(data="abc123", mime_type="image/png")
        self.mock_content.assert_called_with(
            role="user", parts=[text_part_instance, image_part_instance]
        )
        self.assertEqual(result, [content_instance])

    def test_convert_system_message(self):
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ]

        part_instance = MagicMock()
        self.mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        self.mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)
        # mock_part
        self.mock_part.from_text.assert_has_calls([call("System message."), call("User message.")])
        self.mock_content.assert_has_calls(
            [
                call(role="user", parts=[part_instance]),
                call(role="user", parts=[part_instance]),
            ]
        )
        self.assertEqual(result, [content_instance, content_instance])

    def test_convert_assistant_message(self):
        messages = [{"role": "assistant", "content": "Sure, I can help with that."}]

        part_instance = MagicMock()
        self.mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        self.mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        self.mock_part.from_text.assert_called_with("Sure, I can help with that.")
        self.mock_content.assert_called_with(role="model", parts=[part_instance])
        self.assertEqual(result, [content_instance])

    def test_convert_tool_message(self):
        messages = [
            {
                "role": "tool",
                "content": "tool response",
                "name": "tool1",
            }
        ]

        part_instance = MagicMock()
        self.mock_part.from_function_response.return_value = part_instance

        content_instance = MagicMock()
        self.mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        self.mock_part.from_function_response.assert_called_with(
            name="tool1", response={"content": "tool response"}
        )
        self.mock_content.assert_called_with(parts=[part_instance])
        self.assertEqual(result, [content_instance])

    def test_invalid_role(self):
        messages = [{"role": "invalid_role", "content": "This should raise an error."}]

        with self.assertRaises(ValidationError):
            _convert_openai_messages_to_gemini(messages)
