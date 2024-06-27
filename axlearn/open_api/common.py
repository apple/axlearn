# Copyright © 2024 Apple Inc.

"""Common utils for generating responses from open API or open source models."""

import asyncio
import copy
import json
import logging
import os
import time
from collections import defaultdict
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import requests

# isort: off
from tqdm.asyncio import tqdm

from axlearn.common.config import REQUIRED, Configurable, Required, config_class

# pylint: disable=import-error
# pytype: disable=import-error
from openai.types.chat.chat_completion_message import ChatCompletionMessage

# pylint: enable=import-error
# pytype: enable=import-error
# isort: on

# Default decoding parameters.
_default_decode_parameters = {"max_tokens": 1024, "temperature": 0.0}


_openai_decode_parameters = [
    "best_of",
    "echo",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "top_p",
    "temperature",
    "suffix",
    # Not used in openai api but popular in other frameworks.
    "top_k",
]


class ClientType(Enum):
    """Type of different clients."""

    OPENAI = "openai"


class ClientRateLimitError(ValueError):
    """Exception for client rate limit request."""

    pass


class ValidationError(ValueError):
    """Validation failure (e.g. input request format)."""

    pass


class BaseClient(Configurable):
    """Defines the client for Open API style model endpoint for decoding
    and response parsing."""

    @config_class
    class Config(Configurable.Config):
        """Configures BaseClient."""

        # The model name.
        model: Required[str] = REQUIRED
        # Seconds for timeout requests.
        timeout: int = 120
        # A dict of extra body for requests.
        extra_body: Optional[Dict[str, Any]] = None

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Creates the endpoint client.
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Initializes the client.

        Returns:
            Any: a client for the target endpoint.
        """
        raise NotImplementedError(type(self))

    @classmethod
    def parse_generation(cls, response: Dict[str, Any]) -> Sequence[ChatCompletionMessage]:
        """Parses generation from response.

        Args:
           response: A dictionary consisting of the response obtained from the generator.

        Returns:
            A list of ChatCompletionMessage generation.
        """
        raise NotImplementedError(cls)

    async def async_generate(
        self,
        *,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generates response asynchronously from the client.

        Args:
            client: Endpoint client.
            messages: OpenAI requests style messages. Ref:
                https://github.com/openai/openai-python/blob/f3e6e634a86d5789ab1274ae27f43adc842f4ba8/src/openai/types/chat/chat_completion_message.py#L25
            tools: OpenAI tools definitions.
            prompt: OpenAI prompt style.
            **kwargs: API request keyword arguments.

        Returns:
            Response in string format.

        Raises:
            RateLimitError: Hits rate limiting for retries.
        """
        raise NotImplementedError(type(self))


class Generator(Configurable):
    """Defines the generator for Open API style models decoding and generation
    with sync and async concurrency implementation."""

    @config_class
    class Config(Configurable.Config):
        """Configures Generator."""

        # Seconds for timeout requests.
        timeout: int = 120
        # Number of concurrent clients for generations.
        concurrency: int = 8
        # Max number of retries for a request due to non rate limit error.
        max_non_rate_limit_retries: int = 5
        # Max number of retries for a request due to rate limit error.
        max_rate_limit_retries: int = 25
        # Seconds for retry sleep time when hitting rate limit.
        retry_sleep_in_seconds: int = 4
        # True to allow non rate limit error and store empty response.
        allow_non_rate_limit_error: bool = True
        # A dict of decoding parameters.
        # If None, max_tokens is 1024 and temperature is 0.0 for greedy decoding.
        decode_parameters: Optional[Dict[str, Any]] = None
        # Client for API endpoint.
        client: Required[BaseClient.Config] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Create number of clients for concurrent requests.
        self._clients = [cfg.client.instantiate() for _ in range(cfg.concurrency)]
        self._semaphore = asyncio.Semaphore(cfg.concurrency)

    async def _async_generate_from_request(
        self,
        client: BaseClient,
        *,
        request: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Processes individual request asynchronously using the configured client.

        Args:
            client: Target endpoint client like AsyncOpenAI.
            request: Prompt request data to be processed.
            kwargs: Additional keyword arguments.

        Returns:
            Processed prompt data.

        Raises:
            ValueError: Exceed max retries.
        """
        cfg: Generator.Config = self.config
        async with self._semaphore:
            non_rate_limit_retries = 0
            rate_limit_retries = 0
            while True:
                try:
                    response = await client.async_generate(
                        messages=request.get("messages", None),
                        tools=request.get("tools", None),
                        prompt=request.get("prompt", None),
                        **kwargs,
                    )
                    break
                except ValidationError as e:
                    raise e
                except ClientRateLimitError as e:
                    logging.error("Hit request with rate limit error: %s.", str(e))
                    rate_limit_retries += 1
                    if rate_limit_retries >= cfg.max_rate_limit_retries:
                        response = json.dumps(
                            {"error": "Exceed max retries of rate limiting error"}
                        )
                        break
                    await asyncio.sleep(cfg.retry_sleep_in_seconds)
                # pylint: disable-next=broad-except,broad-exception-caught
                except Exception as e:
                    logging.error("Hit request with non rate limit error: %s.", str(e))
                    non_rate_limit_retries += 1
                    if not cfg.allow_non_rate_limit_error:
                        raise ValueError(
                            "Non rate limit error happened. Please inspect the issue."
                        ) from e
                    elif non_rate_limit_retries >= cfg.max_non_rate_limit_retries:
                        response = json.dumps(
                            {"error": "Exceed max retries of non rate limiting error."}
                        )
                        break

            request["response"] = response
            return request

    async def async_generate_from_requests(
        self,
        gen_requests: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generates from OpenAI style requests.

        Args:
           gen_requests: A list of OpenAI style request.
           kwargs: Extra decode parameters, eg: model=xxx,top_p=xxx,max_tokens=xxx.
                Ref: https://platform.openai.com/docs/api-reference/chat/create

        Returns:
            A list of copied requests where each request has a new field response.
        """
        cfg: Generator.Config = self.config
        # Collect responses for each prompt.
        tasks = []
        # Run async requests for each prompt with limited concurrency.
        for index, request in enumerate(gen_requests):
            client_index = index % cfg.concurrency
            if "async_index" in request:
                raise ValueError("Please do not add key async_index in the data.")
            request["async_index"] = index
            client: BaseClient = self._clients[client_index]
            task = self._async_generate_from_request(
                client=client,
                request=request,
                **kwargs,
            )
            tasks.append(task)
        responses: List[Dict[str, Any]] = []
        for task_ in tqdm.as_completed(tasks, total=len(tasks), desc="Generating Response"):
            responses.append(await task_)

        if len(responses) > 0 and "async_index" in responses[0]:
            responses = sorted(responses, key=lambda x: x["async_index"])
            for item in responses:
                if "async_index" in item:
                    del item["async_index"]
        return responses


def check_vllm_readiness(timeout: timedelta, base_url: str):
    """Checks the readiness of vllm server.

    Args:
        timeout: The time duration to wait for readiness.
        base_url: The base URL for the VLLM like http://0.0.0.0:8000/v1.

    Raises:
        TimeoutError: Server did not start within the specified timeout.
    """
    base_url = base_url.rstrip("/")
    url = f"{base_url}/models"
    start_time = time.time()

    while time.time() - start_time < timeout.total_seconds():
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                break  # Server is up, exit the loop.
        except requests.RequestException:
            logging.info("Server is not available. Retrying...")
        time.sleep(5)
    else:
        raise TimeoutError("Server did not start within the specified timeout.")

    logging.info("VLLM server is up and running at %s", url)


def parse_decode_parameters(
    decode_parameters_str: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parses all decode parameters into API supported parameters and request body.


    Args:
        decode_parameters_str: A json string of decode parameters. If None is parsed,
            defaults to max_tokens=1024 and temperature=0.0 (greedy decoding).
        model: Model name. If not None, will be added to decoding parameters
            with the key "model".

    Returns:
        A tuple of:
           decode_parameters: Parsed decode parameters in pre-defined arguments.
           extra_body: Parsed extra key value setting.
    """
    # Create a dictionary from key-value pairs.
    extra_body = {}
    all_decode_parameters = _default_decode_parameters
    if decode_parameters_str is not None:
        all_decode_parameters.update(json.loads(decode_parameters_str))
    decode_parameters = {}
    logging.info("Decode parameters are:")
    log_info = ""
    for key, value in all_decode_parameters.items():
        log_info += f"Key=[{key}] and Value=[{value}]\n"
        if key in _openai_decode_parameters:
            decode_parameters[key] = value
        else:
            extra_body[key] = value
    logging.info(log_info)
    if model is not None:
        decode_parameters.update({"model": model})
    return decode_parameters, extra_body


def repeat_requests(
    gen_requests: List[Dict[str, Any]],
    num_repeats: int,
) -> List[Dict[str, Any]]:
    """Duplicates repuests to generate multiple samples.

    Args:
        gen_requests: A list of OpenAI style requests.
        num_repeats: Number of repeats for each request.

    Returns:
        A list of OpenAI style requests expanded with num_repeats copies.
    """
    new_requests: List[Dict[str, Any]] = []
    id_key = "id" if "id" in gen_requests[0] else "deliverable_id"
    for request in gen_requests:
        for i in range(num_repeats):
            aug_id = f"{request[id_key]}:::{i}"
            new_request = copy.deepcopy(request)
            new_request.update({id_key: aug_id})
            new_requests.append(new_request)
    logging.info("Generating %d samples per request.", num_repeats)
    return new_requests


def flatten_responses(
    responses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flattens responses from repeated requests.

    Args:
        responses: A list of repeated OpenAI style requests with responses.

    Returns:
        A flatten list of OpenAI style requests with responses and candidates.
    """
    new_responses: List[Dict[str, Any]] = []
    candidate_pool = defaultdict(list)
    id_key = "id" if "id" in responses[0] else "deliverable_id"
    for resp in responses:
        original_id = resp[id_key].split(":")[0]
        candidate_pool[original_id].append(resp["response"])
    for resp in responses:
        original_id, sample_id = resp[id_key].split(":::")
        if sample_id == "0":
            resp.update({id_key: original_id, "n_responses": candidate_pool[original_id]})
            new_responses.append(resp)
    return new_responses


def load_requests(
    file_path: str,
    *,
    max_instances: int,
) -> List[Dict[str, Any]]:
    """Loads JSON prompt objects from a file.

    Args:
        file_path: A string representing the path to the file containing JSON prompts.
        max_instances: An integer specifying the maximum number of prompt instances to load.

    Returns:
        A list of dictionaries, where each dictionary represents a prompt object
            loaded from the file.
    """
    gen_requests: List[Dict[str, Any]] = []
    logging.info("Loading prompts.")
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            gen_requests.append(json_data)
    if max_instances is not None:
        gen_requests = gen_requests[:max_instances]
    logging.info("Loaded %d prompts.", len(gen_requests))
    return gen_requests


def write_responses(responses: Sequence[Dict[str, Any]], *, file_path: str):
    """Writes responses to a JSONL file.

    Args:
        responses: A list of model responses.
        file_path: The output file path.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logging.info("Writing responses to %s.", file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        for chat_completion in responses:
            json.dump(chat_completion, file)
            file.write("\n")


def parse_responses(
    client: Type[BaseClient], responses: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Parses responses from a generator.

    Args:
        client: A client type.
        responses: A list of responses from the generator.

    Returns:
        A list of parsed generations.
    """
    parsed_responses: List[Dict[str, Any]] = []
    for response in responses:
        response = copy.deepcopy(response)
        try:
            response_body = response["response"]
            if isinstance(response_body, str):
                response_body = json.loads(response_body)
            parsed = client.parse_generation(response_body)
            parsed = [c.model_dump_json() for c in parsed]
            response["parsed_response"] = parsed
            parsed_responses.append(response)
        # pylint: disable-next=broad-except,broad-exception-caught
        except Exception as e:
            logging.error("Parsing error: %s.", str(e))
            parsed_responses.append(response)
    return parsed_responses
