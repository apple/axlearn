# Copyright © 2024 Apple Inc.

"""Implements of Gemini style API endpoint via
https://github.com/googleapis/python-aiplatform"""

import copy
import json
import logging
import os
import random
import string
from typing import Any, Optional

# isort: off
from axlearn.open_api.common import BaseClient, ClientRateLimitError, ValidationError

# pylint: disable=import-error
# pytype: disable=import-error
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

from google.genai import Client, types

# pylint: enable=import-error
# pytype: enable=import-error
# isort: on


class GeminiClient(BaseClient):
    """Gemini endpoint client."""

    def _create_client(self) -> Client:
        """Creates a client for Gemini."""
        project = os.environ.get("VERTEX_AI_PROJECT")
        location = os.environ.get("VERTEX_AI_LOCATION")
        return Client(vertexai=True, project=project, location=location)

    async def async_generate(
        self,
        *,
        request: dict[str, Any],
        **kwargs,
    ) -> str:
        """Generates response asynchronously from the client.

        Args:
            request: OpenAI style request.
            **kwargs: API request keyword arguments.

        Returns:
            Response in string format.

        Raises:
            ValidationError: Field messages must be in request.
        """
        cfg = self.config
        if "messages" not in request:
            raise ValidationError("Field messages must be in request.")
        _format_request(request=request)
        contents = _convert_openai_messages_to_gemini(messages=request["messages"])
        if request.get("tools", None) is not None:
            gemini_tools = _convert_openai_tools_to_gemini(tools=request["tools"])
        else:
            gemini_tools = None
        client: Client = self._client
        extra_body = copy.deepcopy(cfg.extra_body)
        try:
            response = await client.aio.models.generate_content(
                model=cfg.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=kwargs.get("temperature", None),
                    top_k=kwargs.get("top_k", None),
                    top_p=kwargs.get("top_p", None),
                    max_output_tokens=kwargs.get("max_tokens", None),
                    stop_sequences=kwargs.get("stop_sequences", None),
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=extra_body.get("thinking_budget", 1024),
                        include_thoughts=extra_body.get("include_thoughts", True),
                    ),
                    tools=gemini_tools,
                ),
            )
            return json.dumps(response.model_dump(mode="json"))
        # pylint: disable-next=broad-except,broad-exception-caught
        except Exception as e:
            if "resource has been exhausted" in str(e).lower():
                raise ClientRateLimitError("Rate limiting") from e
            else:
                self._maybe_reduce_tokens(e, request_kwargs=kwargs)
                raise e

    def _maybe_reduce_tokens(self, exception: Exception, request_kwargs: dict):
        """Reduces completion tokens based on the exception message.

        Args:
            exception: Exception from the request.
            request_kwargs: Request kwargs to update.
        """
        exception: str = str(exception)
        if "reduce" not in exception:
            return
        request_kwargs["max_tokens"] = int(request_kwargs["max_tokens"] * 0.8)
        if request_kwargs["max_tokens"] == 0:
            logging.error("Prompt is already longer than max context length.")
        logging.warning(
            "Reducing target length to %d, Retrying...",
            request_kwargs["max_tokens"],
        )

    @classmethod
    def parse_generation(cls, response: dict[str, Any]) -> list[ChatCompletionMessage]:
        """Parse generation from response.

        Args:
           response: A dictionary of response.

        Returns:
            A string of generation or a list of tool calls.
        """
        if len(response.get("candidates", [])) == 0:
            return [ChatCompletionMessage(role="assistant", content="")]

        generations = []
        for candidate in response["candidates"]:
            if candidate.get("content", None) is None:
                continue

            tool_calls = []
            message = ChatCompletionMessage(role="assistant", content="")
            for part in candidate["content"].get("parts", []):
                if "text" in part:
                    if part["thought"]:
                        message.reasoning_content = part["text"]
                    else:
                        message.content = part["text"]
                if (
                    "function_call" not in part
                    or part["function_call"] is None
                    or "name" not in part["function_call"]
                ):
                    continue
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        function=Function(
                            name=part["function_call"]["name"],
                            arguments=json.dumps(part["function_call"]["args"]),
                        ),
                        type="function",
                        id=_generate_call_id(),
                    )
                )
            if len(tool_calls) > 0:
                message.tool_calls = tool_calls
            generations.append(message)
        return generations


# Current gemini model has a length limit for tool name. Set it as 32.
_max_tool_name_length = 32


def _format_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    """Formats tool role message to reduce tool name length."""
    if "tool_calls" in message and message["tool_calls"]:
        new_tool_calls = []
        for tool_call in message["tool_calls"]:
            tool_call["function"]["name"] = tool_call["function"]["name"][-_max_tool_name_length:]
            new_tool_calls.append(tool_call)
        message["tool_calls"] = new_tool_calls
    if message["role"] == "tool" and "name" in message:
        message["name"] = message["name"][-_max_tool_name_length:]
    return message


def _aggregate_tool_role_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregates all tool role messages into one."""
    aggregated_messages = []
    for message in messages:
        if message["role"] != "tool":
            aggregated_messages.append(message)
            continue
        if len(aggregated_messages) > 0 and aggregated_messages[-1]["role"] == "tool":
            aggregated_messages[-1]["tool_messages"].append(message)
            continue

        aggregated_messages.append({"role": "tool", "tool_messages": [message]})

    return aggregated_messages


def _format_request(request: dict[str, Any]):
    """Formats request to follow Gemini request rules."""
    if "messages" in request:
        request["messages"] = [
            _format_tool_message(message=message) for message in request["messages"]
        ]
    if "target_message" in request:
        message = request["target_message"]
        request["target_message"] = _format_tool_message(message=message)
    if "tools" in request:
        new_tools = []
        for tool in request["tools"]:
            tool["function"]["name"] = tool["function"]["name"][-_max_tool_name_length:]
            new_tools.append(tool)
        request["tools"] = new_tools


def _convert_openai_messages_to_gemini(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Converts OpenAI messages to Gemini Content.

    Note: system messages are converted into user messages due to a design limitation that system
    prompts need to be set when GenerativeModel is created instead of per request. See
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions

    Args:
        messages: OpenAI format messages.

    Returns:
        A list of Gemini Content format messages.

    Raises:
        ValidationError: Invalid role.
        ValidationError: Invalid content type.
    """
    # Aggregate tool role messages into one.
    messages = _aggregate_tool_role_messages(messages=messages)

    gemini_messages = []
    for message in messages:
        role = message["role"]
        if role == "system":
            role = "user"
        if role == "user":
            if isinstance(message["content"], str):
                content = types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(text=message["content"]),
                    ],
                )
            elif isinstance(message["content"], list):
                # Supports multi modal content.
                parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(types.Part.from_text(text=content["text"]))
                    elif content["type"] == "image_url":
                        mime_type, data = (
                            content["image_url"]["url"].split("data:")[1].split(";base64,")
                        )
                        parts.append(types.Part.from_data(data=data, mime_type=mime_type))
                content = types.Content(
                    role=role,
                    parts=parts,
                )
            else:
                raise ValidationError(f"Invalid content type {type(message['content'])}")
        elif role == "assistant":
            role = "model"
            if "tool_calls" in message and message["tool_calls"]:
                parts = []
                for tool_call in message["tool_calls"]:
                    args = tool_call["function"]["arguments"]

                    if isinstance(args, str):
                        args = json.loads(args)
                    part = types.Part.from_function_call(
                        name=tool_call["function"]["name"],
                        args=args,
                    )
                    parts.append(part)
                content = types.Content(
                    role=role,
                    parts=parts,
                )
            else:
                content = types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(text=message["content"]),
                    ],
                )
        elif role == "tool":
            content = types.Content(
                parts=[
                    types.Part.from_function_response(
                        name=m["name"],
                        response={
                            "content": m["content"],
                        },
                    )
                    for m in message["tool_messages"]
                ],
            )
        else:
            raise ValidationError(f"Invalid role {role}")

        gemini_messages.append(content)
    return gemini_messages


def _convert_openai_tools_to_gemini(tools: Optional[list[Any]]) -> list[types.Tool]:
    """Converts openai tools to Gemini FunctionDeclaration."""

    def _convert_parameters(params: dict[str, Any]) -> dict[str, Any]:
        if "properties" not in params:
            return params
        for param in params["properties"].values():
            # Gemini only support string enums.
            # Converting it to strings if the type is not a string.
            if "enum" in param and param["type"] != "string":
                enums = [str(e) for e in param["enum"]]
                param["enum"] = enums
                param["type"] = "string"
        return params

    gemini_tools = copy.deepcopy(tools)
    funcs = []
    for tool in gemini_tools:
        tool["function"]["name"] = tool["function"]["name"][-_max_tool_name_length:]
        funcs.append(
            types.FunctionDeclaration(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=_convert_parameters(tool["function"]["parameters"]),
            )
        )
    return [types.Tool(function_declarations=funcs)]


def _generate_call_id(length: int = 24) -> str:
    """Generates call id like call_AUBkf9IL3lGQ2CUu2JmDs9Vf."""
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))
