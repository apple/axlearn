# Copyright © 2024 Apple Inc.

"""Utils for unit test mocking."""

import sys
import types
from unittest.mock import MagicMock


def mock_openai_package():
    """Initialize openai package for unit tests."""
    # Create mock for the openai module and its submodules.
    mock_openai = types.ModuleType("openai")
    mock_async_openai = MagicMock()
    mock_openai.AsyncOpenAI = mock_async_openai
    mock_rate_limit_error = MagicMock()
    mock_openai.RateLimitError = mock_rate_limit_error

    mock_openai_types = types.ModuleType("openai.types")
    mock_openai_types_chat = types.ModuleType("openai.types.chat")
    mock_chat_completion_message = types.ModuleType("openai.types.chat.chat_completion_message")
    mock_chat_completion_message.ChatCompletionMessage = MagicMock()

    mock_openai_types_completion = types.ModuleType("openai.types.completion")
    mock_completion = MagicMock()

    # Set up the mock module structure.
    mock_openai.types = mock_openai_types
    mock_openai.types.chat = mock_openai_types_chat
    mock_openai.types.chat.ChatCompletion = MagicMock()
    mock_openai.types.chat.chat_completion_message = mock_chat_completion_message
    mock_openai.types.completion = mock_openai_types_completion
    mock_openai.types.completion.Completion = mock_completion

    # Patch sys.modules to replace the openai package with our mock.
    sys.modules["openai"] = mock_openai
    sys.modules["openai.types"] = mock_openai_types
    sys.modules["openai.types.chat"] = mock_openai_types_chat
    sys.modules["openai.types.chat.chat_completion_message"] = mock_chat_completion_message
    sys.modules["openai.types.completion"] = mock_openai_types_completion
