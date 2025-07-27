# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for gemini.py"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from absl.testing import parameterized

from axlearn.common.test_utils import TestCase
from axlearn.open_api import mock_utils

with mock_utils.mock_google_genai_package(), mock_utils.mock_openai_package():
    # pylint: disable=wrong-import-position
    from axlearn.open_api.common import ValidationError
    from axlearn.open_api.gemini import (
        GeminiClient,
        _convert_openai_messages_to_gemini,
        _format_tool_message,
    )

# pylint: enable=wrong-import-position

_MODULE_ROOT = "axlearn"


@mock_utils.safe_mocks(mock_utils.mock_openai_package, mock_utils.mock_google_genai_package)
class TestGeminiClient(unittest.IsolatedAsyncioTestCase):
    """Unit test for class GeminiClient."""

    def _create_gemini_client(self) -> GeminiClient:
        client: GeminiClient = (
            GeminiClient.default_config().set(model="gemini-1.0-pro", extra_body={}).instantiate()
        )
        client._client = AsyncMock()
        return client

    @patch(f"{_MODULE_ROOT}.open_api.gemini._convert_openai_messages_to_gemini")
    @patch(f"{_MODULE_ROOT}.open_api.gemini._convert_openai_tools_to_gemini")
    async def test_async_generate(self, mock_convert_tools, mock_convert_messages):
        mock_convert_messages.return_value = "converted_messages"
        mock_convert_tools.return_value = "converted_tools"
        client = self._create_gemini_client()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"response": "test_response"}
        client._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

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


@mock_utils.safe_mocks(mock_utils.mock_openai_package, mock_utils.mock_google_genai_package)
class TestConvertOpenAIMessagesToGemini(TestCase):
    """Unit tests for _convert_openai_messages_to_gemini."""

    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Content")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Part")
    def test_convert_user_text_message(self, mock_part, mock_content):
        messages = [{"role": "user", "content": "Hello, how can I help you?"}]

        part_instance = MagicMock()
        mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)
        # mock_part
        mock_part.from_text.assert_called_with(text="Hello, how can I help you?")
        mock_content.assert_called_once()
        mock_content.assert_called_with(role="user", parts=[part_instance])
        self.assertEqual(result, [content_instance])

    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Content")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Part")
    def test_convert_user_multimodal_message(self, mock_part, mock_content):
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
        mock_part.from_text.return_value = text_part_instance
        mock_part.from_data.return_value = image_part_instance

        content_instance = MagicMock()
        mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        mock_part.from_text.assert_called_with(text="Look at this image:")
        mock_part.from_data.assert_called_with(data="abc123", mime_type="image/png")
        mock_content.assert_called_with(
            role="user", parts=[text_part_instance, image_part_instance]
        )
        self.assertEqual(result, [content_instance])

    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Content")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Part")
    def test_convert_system_message(self, mock_part, mock_content):
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ]

        part_instance = MagicMock()
        mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)
        # mock_part
        mock_part.from_text.assert_has_calls(
            [call(text="System message."), call(text="User message.")]
        )
        mock_content.assert_has_calls(
            [
                call(role="user", parts=[part_instance]),
                call(role="user", parts=[part_instance]),
            ]
        )
        self.assertEqual(result, [content_instance, content_instance])

    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Content")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Part")
    def test_convert_assistant_message(self, mock_part, mock_content):
        messages = [{"role": "assistant", "content": "Sure, I can help with that."}]

        part_instance = MagicMock()
        mock_part.from_text.return_value = part_instance

        content_instance = MagicMock()
        mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        mock_part.from_text.assert_called_with(text="Sure, I can help with that.")
        mock_content.assert_called_with(role="model", parts=[part_instance])
        self.assertEqual(result, [content_instance])

    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Content")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.types.Part")
    def test_convert_tool_message(self, mock_part, mock_content):
        messages = [
            {
                "role": "tool",
                "content": "tool response",
                "name": "tool1",
            }
        ]

        part_instance = MagicMock()
        mock_part.from_function_response.return_value = part_instance

        content_instance = MagicMock()
        mock_content.return_value = content_instance

        result = _convert_openai_messages_to_gemini(messages)

        mock_part.from_function_response.assert_called_with(
            name="tool1", response={"content": "tool response"}
        )
        mock_content.assert_called_with(parts=[part_instance])
        self.assertEqual(result, [content_instance])

    def test_invalid_role(self):
        messages = [{"role": "invalid_role", "content": "This should raise an error."}]

        with self.assertRaises(ValidationError):
            _convert_openai_messages_to_gemini(messages)


@mock_utils.safe_mocks(mock_utils.mock_openai_package, mock_utils.mock_google_genai_package)
class TestParseGeneration(TestCase):
    """Unit tests for parse_generation method."""

    @parameterized.parameters(
        [
            # Test case format: (name, response, expected_result_count, expected_content,
            #  expected_reasoning,
            #  expected_tool_calls, expected_function_calls, should_call_message)
            ("empty_candidates", {"candidates": []}, 1, "", None, 0, [], True),
            ("no_candidates_key", {}, 1, "", None, 0, [], True),
            (
                "candidate_no_content",
                {"candidates": [{"finish_reason": "STOP"}, {"content": None}]},
                0,
                None,
                None,
                0,
                [],
                False,
            ),
            (
                "text_content",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello, how can I help you?", "thought": False}]
                            }
                        }
                    ]
                },
                1,
                "Hello, how can I help you?",
                None,
                0,
                [],
                True,
            ),
            (
                "reasoning_content",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Let me think about this...", "thought": True}]
                            }
                        }
                    ]
                },
                1,
                "",
                "Let me think about this...",
                0,
                [],
                True,
            ),
            (
                "mixed_text_and_reasoning",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Let me think...", "thought": True},
                                    {"text": "The answer is 42.", "thought": False},
                                ]
                            }
                        }
                    ]
                },
                1,
                "The answer is 42.",
                "Let me think...",
                0,
                [],
                True,
            ),
            (
                "function_call",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "function_call": {
                                            "name": "get_weather",
                                            "args": {"location": "San Francisco"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
                1,
                "",
                None,
                1,
                [("get_weather", '{"location": "San Francisco"}')],
                True,
            ),
            (
                "multiple_function_calls",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "function_call": {
                                            "name": "get_weather",
                                            "args": {"location": "San Francisco"},
                                        }
                                    },
                                    {
                                        "function_call": {
                                            "name": "get_time",
                                            "args": {"timezone": "PST"},
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                },
                1,
                "",
                None,
                2,
                [
                    ("get_weather", '{"location": "San Francisco"}'),
                    ("get_time", '{"timezone": "PST"}'),
                ],
                True,
            ),
            (
                "invalid_function_call",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"function_call": None},  # None function_call
                                    {"function_call": {}},  # Missing name
                                    {
                                        "function_call": {
                                            "name": "valid_function",
                                            "args": {"param": "value"},
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                },
                1,
                "",
                None,
                1,
                [("valid_function", '{"param": "value"}')],
                True,
            ),
            (
                "multiple_candidates",
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "First response", "thought": False}]}},
                        {"content": {"parts": [{"text": "Second response", "thought": False}]}},
                    ]
                },
                2,
                "First response",
                None,
                0,
                [],
                True,  # We'll check both messages separately
            ),
            (
                "mixed_content_and_function_calls",
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "I'll help you with that.", "thought": False},
                                    {
                                        "function_call": {
                                            "name": "search_web",
                                            "args": {"query": "weather forecast"},
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                },
                1,
                "I'll help you with that.",
                None,
                1,
                [("search_web", '{"query": "weather forecast"}')],
                True,
            ),
        ]
    )
    @patch(f"{_MODULE_ROOT}.open_api.gemini._generate_call_id")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.Function")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.ChatCompletionMessageToolCall")
    @patch(f"{_MODULE_ROOT}.open_api.gemini.ChatCompletionMessage")
    def test_parse_generation(
        self,
        name,
        response,
        expected_result_count,
        expected_content,
        expected_reasoning,
        expected_tool_calls,
        expected_function_calls,
        should_call_message,
        mock_message,
        mock_tool_call,
        mock_function,
        mock_generate_id,
    ):
        """Unified test for parse_generation method covering all scenarios."""

        # Set up mock generate_id
        if expected_tool_calls > 0:
            mock_generate_id.side_effect = [f"call_test{i}" for i in range(expected_tool_calls)]

        # Set up mock function calls
        mock_functions = []
        mock_tool_calls = []
        for i, (func_name, func_args) in enumerate(expected_function_calls):
            mock_func = MagicMock()
            mock_func.name = func_name
            mock_func.arguments = func_args
            mock_functions.append(mock_func)

            mock_tc = MagicMock()
            mock_tc.function = mock_func
            mock_tc.type = "function"
            mock_tc.id = f"call_test{i}"
            mock_tool_calls.append(mock_tc)

        if mock_functions:
            mock_function.side_effect = mock_functions
            mock_tool_call.side_effect = mock_tool_calls

        # Set up mock messages
        mock_messages = []
        for i in range(expected_result_count):
            mock_msg = MagicMock()
            mock_msg.role = "assistant"

            if name == "multiple_candidates":
                mock_msg.content = "First response" if i == 0 else "Second response"
            else:
                mock_msg.content = expected_content or ""

            if expected_reasoning:
                mock_msg.reasoning_content = expected_reasoning

            if expected_tool_calls > 0:
                mock_msg.tool_calls = mock_tool_calls

            mock_messages.append(mock_msg)

        if len(mock_messages) == 1:
            mock_message.return_value = mock_messages[0]
        else:
            mock_message.side_effect = mock_messages

        # Execute the test
        result = GeminiClient.parse_generation(response)

        # Verify results
        self.assertEqual(len(result), expected_result_count)

        if should_call_message:
            if expected_result_count > 1:
                self.assertEqual(mock_message.call_count, expected_result_count)
            else:
                mock_message.assert_called_once_with(role="assistant", content="")

            # Verify message content
            if expected_result_count == 1:
                if expected_content is not None:
                    self.assertEqual(mock_messages[0].content, expected_content)
                if expected_reasoning:
                    self.assertEqual(mock_messages[0].reasoning_content, expected_reasoning)
                if expected_tool_calls > 0:
                    self.assertEqual(len(mock_messages[0].tool_calls), expected_tool_calls)
            elif name == "multiple_candidates":
                self.assertEqual(mock_messages[0].content, "First response")
                self.assertEqual(mock_messages[1].content, "Second response")

            # Verify function calls
            if expected_function_calls:
                expected_calls = [
                    call(name=func_name, arguments=func_args)
                    for func_name, func_args in expected_function_calls
                ]
                mock_function.assert_has_calls(expected_calls)
        else:
            mock_message.assert_not_called()
