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
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Type

import requests
from absl import flags

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
        request: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Generates response asynchronously from the client.

        Args:
            client: Endpoint client.
            request: OpenAI style request. Ref:
                https://github.com/openai/openai-python/blob/50371bf3151ebb1a43017abfe205d4d9b2e5faac/src/openai/resources/chat/completions.py#L237
                https://github.com/openai/openai-python/blob/f3e6e634a86d5789ab1274ae27f43adc842f4ba8/src/openai/types/chat/chat_completion_message.py#L25
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
        # By default, max_tokens is 1024 and temperature is 0.0 for greedy decoding.
        decode_parameters: Dict[str, Any] = _default_decode_parameters
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
                        request=request,
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
        # Passes decode parameters if kwargs is not set.
        kwargs = {**cfg.decode_parameters, **(kwargs or {})}
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

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines flags for generator.py."""
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # For client.
        flags.DEFINE_string("model", None, "The model name.", **common_kwargs)
        flags.DEFINE_string("client_name", "openai", "Open api client name.", **common_kwargs)
        flags.DEFINE_integer("timeout", 120, "Seconds for timeout requests.", **common_kwargs)
        # For generator.
        flags.DEFINE_integer(
            "concurrency", 8, "Number of concurrent clients for generations.", **common_kwargs
        )
        flags.DEFINE_integer(
            "max_non_rate_limit_retries",
            5,
            "Max number of retries for a request due to non rate limit error.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "max_rate_limit_retries",
            25,
            "Max number of retries for a request due to rate limit error.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "retry_sleep_in_seconds",
            4,
            "Seconds for retry sleep time when hitting rate limit.",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "allow_non_rate_limit_error",
            True,
            "True to allow non rate limit error and store empty response.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "decode_parameters",
            None,
            "A json string of decoding parameters."
            "If None, max_tokens is 1024 and temperature is 0.0 for greedy decoding.",
            **common_kwargs,
        )
        # For file input.
        flags.DEFINE_string(
            "input_file",
            None,
            "Path to the input data file. Each line is a json string of OpenAI request style.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "output_file",
            None,
            "Path to the output data file. Each line is a json string of OpenAI request style.",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "check_vllm_readiness",
            True,
            "True to verify the readiness of vllm server.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_timeout",
            1200,
            "The timeout in seconds for checking the server readiness.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "max_instances", None, "Maximum number of instances to test.", **common_kwargs
        )
        flags.DEFINE_boolean(
            "repeat_requests_for_n",
            True,
            "Repeats requests for n in decode parameters.",
            **common_kwargs,
        )
        flags.DEFINE_boolean("debug", False, "True to enable debug mode.", **common_kwargs)


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
    gen_requests: List[Dict[str, Any]] = _load_jsonl_file(file_path=file_path)
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


class EvalGeneratorType(Enum):
    """The type of generator in Evaluator.

    Attributes:
        RESPONSE: The original response generator.
        GRADER: The LLM grader generator.
    """

    RESPONSE = "response"
    GRADER = "grader"


class MetricFn(Protocol):
    """Defines a protocol of metric calculation function."""

    def __call__(
        self,
        *,
        responses: List[Dict[str, Any]],
        generators: Dict[EvalGeneratorType, Generator],
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Implements a protocol to compute metrics from responses.

        Args:
            responses: A list of responses in dictionary.
            generators: A dict of generator instances.
            debug: True to add debug information in the metric.

        Returns:
            A dictionary of metrics.
        """


class Evaluator(Configurable):
    """Defines an evaluator to load generated responses and compute metrics."""

    @config_class
    class Config(Configurable.Config):
        """Configures Evaluator."""

        # A dict of generators including RESPONSE, GRADER type.
        generators: Required[Dict[EvalGeneratorType, Generator.Config]] = REQUIRED
        # True to add debug information in metrics.
        debug: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Initializes generators.
        generators = {}
        for generator_type, generator_cfg in cfg.generators.items():
            generators[generator_type] = generator_cfg.instantiate()
        self._generators = generators
        self._metrics = {}

    def evaluate(
        self,
        *,
        input_file: str,
        output_file: str,
        metric_fn: MetricFn,
    ) -> Dict[str, Any]:
        """Evaluates from generated response from an input file
            and writes out metrics to an output file.

        Args:
            input_file: The input path of generated responses.
            output_file: The output path of generated responses.
            metric_fn: A callable function to compute metrics.

        Returns:
            A dict of metrics.
        """
        responses = _load_jsonl_file(file_path=input_file)
        metrics = metric_fn(
            responses=responses,
            generators=self._generators,
            debug=self.config.debug,
        )
        _write_metrics(metrics=metrics, file_path=output_file)
        return metrics

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines extra flags for evaluator.py."""
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string(
            "grader_model", "gpt-3.5-turbo-0125", "The model name.", **common_kwargs
        )
        flags.DEFINE_string(
            "grader_client_name", "openai", "Open api client name.", **common_kwargs
        )
        flags.DEFINE_string(
            "grader_decode_parameters",
            None,
            "A json string of decoding parameters for grader."
            "If None, it will use decode_parameters in response generator.",
            **common_kwargs,
        )
        flags.DEFINE_string("metric_name", None, "The name of metric.", **common_kwargs)


def _write_metrics(metrics: Dict[str, Any], *, file_path: str):
    """Writes to a json file with computed metrics.

    Args:
        file_path: Path to the metrics output path.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logging.info("Writing metrics %s to %s", metrics, file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def _load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads a file with each line in json line.

    Args:
        file_path: Path to the model outputs file in JSON Lines format.

    Returns:
        A list of responses.
    """
    logging.info("Loading file [%s]", file_path)
    responses = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            response = json.loads(line)
            responses.append(response)
    return responses
