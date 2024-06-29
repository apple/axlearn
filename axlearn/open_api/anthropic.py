# Copyright © 2024 Apple Inc.

"""Implements of Anthropic style API endpoint via
https://github.com/anthropics/anthropic-sdk-python"""

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

# isort: off
from axlearn.open_api.common import BaseClient, ClientRateLimitError, ValidationError

# pylint: disable=import-error
# pytype: disable=import-error
from anthropic import AsyncAnthropic, RateLimitError
from anthropic.types.message import Message
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

# pylint: enable=import-error
# pytype: enable=import-error
# isort: on

_system_parallel_tools_prompt = "Try to use parallel tool calls as much as possible!"


class AnthropicClient(BaseClient):
    """Anthropic endpoint client."""

    def _create_client(self) -> AsyncAnthropic:
        """Creates an AsyncAnthropic client."""
        return AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "EMPTY"))

    async def async_generate(
        self,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generates response asynchronously from the client.

        Args:
            messages: OpenAI requests style messages.
            tools: OpenAI tools definitions.
            prompt: OpenAI prompt style.
            **kwargs: API request keyword arguments.

        Returns:
            Response in string format.

        Raises:
            ClientRateLimitError: Hits rate limiting for retries.
        """
        cfg: AnthropicClient.Config = self.config
        client: AsyncAnthropic = self._client
        request_kwargs = copy.deepcopy(kwargs)
        anthropic_tools = None
        if tools is not None:
            anthropic_tools = _convert_openai_tools_to_anthropic(tools=tools)
        anthropic_messages = _convert_openai_messages_to_anthropic(messages=messages)
        try:
            # A temporary solution to encourage claude models to generate parallel tool calls.
            if request_kwargs is not None and request_kwargs.get(
                "add_system_parallel_tools", False
            ):
                request_kwargs.update({"system": _system_parallel_tools_prompt})
                del request_kwargs["add_system_parallel_tools"]
            response: Message = await client.messages.create(
                messages=anthropic_messages,
                tools=anthropic_tools,
                extra_body=cfg.extra_body,
                **request_kwargs,
            )
            return response.model_dump_json()
        except RateLimitError as e:
            raise ClientRateLimitError("Rate limiting") from e
        # pylint: disable-next=broad-except,broad-exception-caught
        except Exception as e:
            self._maybe_reduce_tokens(e, request_kwargs=kwargs)
            raise e

    def _maybe_reduce_tokens(self, exception: Exception, request_kwargs: dict):
        """Reduces completion tokens based on the exception message.

        Args:
            exception: Exception from the request.
            request_kwargs: Request kwargs to update.
        """
        exception: str = str(exception)
        if "Please reduce" not in exception:
            return
        request_kwargs["max_tokens"] = int(request_kwargs["max_tokens"] * 0.8)
        if request_kwargs["max_tokens"] == 0:
            logging.error("Prompt is already longer than max context length.")
        logging.warning("Reducing target length to %d, Retrying...", request_kwargs["max_tokens"])

    @classmethod
    def parse_generation(cls, response: Dict[str, Any]) -> List[ChatCompletionMessage]:
        """Parses generation from response.

        Args:
           response: A dictionary of response.

        Returns:
            A string of generation or a list of tool calls.
        """
        if len(response.get("content", [])) == 0:
            return [ChatCompletionMessage(role="assistant", content="")]

        tool_calls = []
        text = ""

        for content in response["content"]:
            if content["type"] == "tool_use":
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        function=Function(
                            name=content["name"], arguments=json.dumps(content["input"])
                        ),
                        type="function",
                        id=content["id"],
                    )
                )
            elif content["type"] == "text":
                text = content["text"]

        text_messages: List[ChatCompletionMessage] = []
        tool_calls_messages: List[ChatCompletionMessage] = []

        if len(tool_calls) > 0:
            tool_calls_messages.append(
                ChatCompletionMessage(role="assistant", content="", tool_calls=tool_calls)
            )
        else:
            text_messages.append(ChatCompletionMessage(role="assistant", content=text))

        if len(tool_calls_messages) > 0:
            return tool_calls_messages
        return text_messages


def _convert_openai_messages_to_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts OpenAI style messages to Anthropic.

    Args:
        messages: A list of OpenAI style messages.

    Returns:
        A list of messages in Anthropic style.

    Raises:
        ValidationError: Unknown content type.
    """

    def _contains_tool_results(message: Dict) -> bool:
        if "content" in message and isinstance(message["content"], list):
            for c in message["content"]:
                if c["type"] == "tool_result":
                    return True
        return False

    copied_messages = copy.deepcopy(messages)
    processed_messages = []
    for message in copied_messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            # Handle multimodal requests.
            formatted_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    formatted_content.append(content)
                elif content["type"] == "image_url":
                    mime_type, data = (
                        content["image_url"]["url"].split("data:")[1].split(";base64,")
                    )
                    formatted_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": data,
                            },
                        }
                    )
                else:
                    raise ValidationError(f"Unknown content type: {content['type']}")
            processed_messages.append(
                {
                    "role": message["role"],
                    "content": formatted_content,
                }
            )
        elif message["role"] == "tool":
            new_content = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            if len(processed_messages) > 0 and _contains_tool_results(processed_messages[-1]):
                processed_messages[-1]["content"].append(new_content)
            else:
                new_message = {}
                new_message["role"] = "user"
                new_message["content"] = [new_content]
                processed_messages.append(new_message)
        elif message["role"] == "assistant" and "tool_calls" in message:
            content = []
            if message["content"]:
                content.append({"type": "text", "text": message["content"]})
            content.extend(
                [
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]),
                    }
                    for tool_call in message["tool_calls"]
                ]
            )
            new_message = {
                "role": "assistant",
                "content": content,
            }
            processed_messages.append(new_message)
        else:
            processed_messages.append(message)
    return processed_messages


def _convert_openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts OpenAI tools to Anthropic tools."""
    new_tools = []
    copied_tools = copy.deepcopy(tools)
    for tool in copied_tools:
        func = tool["function"]
        func["input_schema"] = func["parameters"]
        del func["parameters"]
        new_tools.append(func)
    return new_tools
