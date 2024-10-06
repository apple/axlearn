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
    mock_chat_completion_message_tool_call = types.ModuleType(
        "openai.types.chat.chat_completion_message_tool_call"
    )

    mock_chat_completion_message.ChatCompletionMessage = MagicMock()
    mock_chat_completion_message.ChatCompletionMessageToolCall = MagicMock()
    mock_chat_completion_message_tool_call.Function = MagicMock()

    mock_openai_types_completion = types.ModuleType("openai.types.completion")
    mock_completion = MagicMock()

    # Set up the mock module structure.
    mock_openai.types = mock_openai_types
    mock_openai.types.chat = mock_openai_types_chat
    mock_openai.types.chat.ChatCompletion = MagicMock()
    mock_openai.types.chat.chat_completion_message = mock_chat_completion_message
    mock_openai.types.chat.chat_completion_message_tool_call = (
        mock_chat_completion_message_tool_call
    )

    mock_openai.types.completion = mock_openai_types_completion
    mock_openai.types.completion.Completion = mock_completion

    # Patch sys.modules to replace the openai package with our mock.
    sys.modules["openai"] = mock_openai
    sys.modules["openai.types"] = mock_openai_types
    sys.modules["openai.types.chat"] = mock_openai_types_chat
    sys.modules["openai.types.chat.chat_completion_message"] = mock_chat_completion_message
    sys.modules[
        "openai.types.chat.chat_completion_message_tool_call"
    ] = mock_chat_completion_message_tool_call
    sys.modules["openai.types.completion"] = mock_openai_types_completion


def mock_vertexai_package():
    """Initialize vertexai package for unit tests."""
    # Create mock for the vertexai module and its submodules.
    mock_vertexai = types.ModuleType("vertexai")
    mock_generative_models = types.ModuleType("vertexai.generative_models")

    # Create mocks for each class in the generative_models submodule.
    mock_content = MagicMock()
    mock_function_declaration = MagicMock()
    mock_generation_config = MagicMock()
    mock_generative_model = MagicMock()
    mock_part = MagicMock()
    mock_tool = MagicMock()

    # Set up the mock module structure.
    mock_generative_models.Content = mock_content
    mock_generative_models.FunctionDeclaration = mock_function_declaration
    mock_generative_models.GenerationConfig = mock_generation_config
    mock_generative_models.GenerativeModel = mock_generative_model
    mock_generative_models.Part = mock_part
    mock_generative_models.Tool = mock_tool

    mock_vertexai.generative_models = mock_generative_models

    # Patch sys.modules to replace the vertexai package with our mock.
    sys.modules["vertexai"] = mock_vertexai
    sys.modules["vertexai.generative_models"] = mock_generative_models


def mock_anthropic_package():
    """Initializes anthropic package for unit tests."""
    # Create mock for the anthropic module and its submodules.
    mock_anthropic = types.ModuleType("anthropic")
    mock_async_anthropic = MagicMock()
    mock_rate_limit_error = type("RateLimitError", (BaseException,), {})
    mock_anthropic.AsyncAnthropic = mock_async_anthropic
    mock_anthropic.RateLimitError = mock_rate_limit_error

    mock_anthropic_types = types.ModuleType("anthropic.types")
    mock_anthropic_message = types.ModuleType("anthropic.types.message")

    # Mock the Message class within message
    mock_message = MagicMock()
    mock_anthropic_message.Message = mock_message

    # Set up the mock module structure.
    mock_anthropic.types = mock_anthropic_types
    mock_anthropic.types.message = mock_anthropic_message

    # Patch sys.modules to replace the anthropic package with our mock.
    sys.modules["anthropic"] = mock_anthropic
    sys.modules["anthropic.types"] = mock_anthropic_types
    sys.modules["anthropic.types.message"] = mock_anthropic_message


def mock_huggingface_hub_package():
    """Initialize huggingface hub package for unit tests."""
    # Create mock for the openai module and its submodules.
    mock_hf_hub = types.ModuleType("huggingface_hub")
    mock_hf_hub.snapshot_download = MagicMock()
    # Patch sys.modules to replace the huggingface_hub package with our mock.
    sys.modules["huggingface_hub"] = mock_hf_hub
