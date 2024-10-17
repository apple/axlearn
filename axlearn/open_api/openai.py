# Copyright © 2024 Apple Inc.

"""Implements of OpenAI style API endpoint via
https://github.com/openai/openai-python

This would work for both ChatGPT and vLLM based open source models.
"""
import copy
import json
import logging
import os
import re
from typing import Any

from axlearn.open_api.common import (
    BaseClient,
    ClientRateLimitError,
    EvalGeneratorType,
    ValidationError,
)

# isort: off
# pylint: disable=import-error
# pytype: disable=import-error
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion import Completion

# pylint: enable=import-error
# pytype: enable=import-error
# isort: on


class OpenAIClient(BaseClient):
    """OpenAI endpoint client.

    Compatible with vLLM service OpenAI endpoint.
    """

    def _create_client(self) -> AsyncOpenAI:
        """Creates an AsyncOpenAI client."""
        cfg: OpenAIClient.Config = self.config
        default_headers = None
        if os.environ.get("COOKIE") is not None:
            default_headers = {"Cookie": os.environ["COOKIE"]}
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = None
        if cfg.generator_type == EvalGeneratorType.GRADER:
            api_key = os.environ.get("GRADER_OPENAI_API_KEY", api_key)
            base_url = os.environ.get("GRADER_OPENAI_BASE_URL", "https://api.openai.com/v1")

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            timeout=cfg.timeout,
        )

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
            ClientRateLimitError: Hits rate limiting for retries.
            ValidationError: Both prompt and messages are None.
        """
        cfg: OpenAIClient.Config = self.config
        client: AsyncOpenAI = self._client
        prompt = request.get("prompt", None)
        messages = request.get("messages", None)
        if prompt is None and messages is None:
            raise ValidationError("Both prompt and messages are None.")
        try:
            if prompt is not None:
                response: Completion = await client.completions.create(
                    prompt=request["prompt"],
                    extra_body=cfg.extra_body,
                    **kwargs,
                )
            else:
                req_kwargs = copy.deepcopy(kwargs)
                if "tools" in request:
                    req_kwargs.update({"tools": request["tools"]})
                response: ChatCompletion = await client.chat.completions.create(
                    messages=messages,
                    extra_body=cfg.extra_body,
                    **req_kwargs,
                )
        except RateLimitError as e:
            raise ClientRateLimitError("Rate limiting") from e
        # pylint: disable-next=broad-except,broad-exception-caught
        except Exception as e:
            self._maybe_reduce_tokens(e, request_kwargs=kwargs)
            raise e
        return response.model_dump_json()

    def _maybe_reduce_tokens(self, exception: Exception, request_kwargs: dict):
        """Reduces completion tokens based on the exception message.

        Args:
            exception: Exception from the request.
            request_kwargs: Request kwargs to update.
        """
        exception: str = str(exception)
        if "Please reduce" not in exception:
            return
        max_completion_tokens = _calculate_openai_max_completion_tokens(exception)
        if max_completion_tokens < 0:
            logging.error("Prompt is already longer than max context length.")
        elif max_completion_tokens > 0:
            request_kwargs["max_tokens"] = max_completion_tokens
        else:
            request_kwargs["max_tokens"] = int(request_kwargs["max_tokens"] * 0.8)
            if request_kwargs["max_tokens"] == 0:
                logging.error("Prompt is already longer than max context length.")
        logging.warning("Reducing target length to %d, Retrying...", request_kwargs["max_tokens"])

    @classmethod
    def _parse_generation_from_message(cls, message: dict[str, Any]) -> ChatCompletionMessage:
        """Parse generation from a message.

        Args:
           message: A dictionary of message.

        Returns:
            A string of generation or a list of tool calls.
        """
        message: ChatCompletionMessage = ChatCompletionMessage(**message)
        if message.content is not None and message.content != "":
            generation = message.content
            # Strip left side space for some SPM generations.
            generation = generation.replace("\n", "<n>").lstrip().replace("<n>", "\n")
            message.content = generation
            return message

        return message

    @classmethod
    def parse_generation(cls, response: dict[str, Any]) -> list[ChatCompletionMessage]:
        """Parse generation from response.

        Args:
           response: A dictionary of response.

        Returns:
            A string of generation or a list of tool calls.
        """
        if len(response.get("choices", [])) == 0:
            return [ChatCompletionMessage(role="assistant", content="")]

        # Extract text generation.
        generations = []
        for choice in response["choices"]:
            generations.append(OpenAIClient._parse_generation_from_message(choice["message"]))
        return generations

    @classmethod
    def format_message(cls, message: dict[str, Any]) -> ChatCompletionMessage:
        """Format message with requirements.

        Args:
           message: A dictionary of message.

        Returns:
            A formatted ChatCompletionMessage.
        """
        if "content" in message and isinstance(message["content"], dict):
            message["content"] = json.dumps(message["content"])
        if "tool_calls" in message:
            new_tool_calls = []
            for tool_call in message["tool_calls"]:
                if isinstance(tool_call["function"]["arguments"], dict):
                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"], sort_keys=True
                    )
                new_tool_calls.append(tool_call)
            message["tool_calls"] = new_tool_calls
        return ChatCompletionMessage(**message)


def _calculate_openai_max_completion_tokens(err_msg: str) -> int:
    """Use regular expression to get max completion tokens based on model max length.

    Args:
        err_msg: A string of error message.

    Returns:
        An integer of calculated max tokens.
    """
    # Use regular expression to find model max length.
    match = re.search(r"maximum context length is (\d+) tokens", err_msg)

    if match:
        model_max_tokens = int(match.group(1))
    else:
        logging.warning("Didn't find max model tokens")
        return 0

    # Use regular expression to find messages token.
    match = re.search(r"\((\d+) in the messages", err_msg)

    if match:
        message_tokens = int(match.group(1))
    else:
        logging.warning("Didn't find message tokens")
        return 0
    if message_tokens >= model_max_tokens:
        return -1
    logging.info("Max completion tokens is %d", model_max_tokens - message_tokens)
    return model_max_tokens - message_tokens
