# Copyright © 2024 Apple Inc.

"""Generates responses from open api or open source models.
Input format is json line with each line in OpenAI requests style.
"""
import asyncio
import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Type

from absl import app, flags, logging
from absl.flags import FlagValues

from axlearn.open_api.common import (
    BaseClient,
    Generator,
    check_vllm_readiness,
    flatten_responses,
    load_requests,
    parse_decode_parameters,
    repeat_requests,
    write_responses,
)
from axlearn.open_api.registry import ClientRegistry

FLAGS = flags.FLAGS


def generator_config_from_flags(fv: FlagValues) -> Generator.Config:
    """Creates config from flags.

    Args:
        fv: The flag values of generator.py.

    Returns:
        An instance of Generator.

    Raises:
        ValueError: Client class must not be empty.
    """
    client_cls: Optional[Type[BaseClient]] = ClientRegistry.load_client_cls(fv.client_name)
    if client_cls is None:
        raise ValueError("Client class must not be empty.")
    decode_parameters, extra_body = parse_decode_parameters(fv.decode_parameters, model=fv.model)
    # Initializes generator.
    cfg: Generator.Config = Generator.default_config().set(
        concurrency=fv.concurrency,
        max_non_rate_limit_retries=fv.max_non_rate_limit_retries,
        allow_non_rate_limit_error=fv.allow_non_rate_limit_error,
        max_rate_limit_retries=fv.max_rate_limit_retries,
        decode_parameters=decode_parameters,
        retry_sleep_in_seconds=fv.retry_sleep_in_seconds,
        client=client_cls.default_config().set(
            model=fv.model, timeout=fv.timeout, extra_body=extra_body
        ),
    )
    return cfg


async def generate_from_requests(
    *, gen_requests: List[Dict[str, Any]], fv: FlagValues
) -> List[Dict[str, Any]]:
    """Generates responses for open api clients given a file path.

    Args:
        gen_requests: A list of OpenAI style requests. Ref:
            https://platform.openai.com/docs/api-reference/chat/create.
        fv: The flag values of generator.py.

    Returns:
        A list of generation responses.

    Raises:
        ValueError: Decode parameters n must be larger than 0.
    """
    cfg: Generator.Config = generator_config_from_flags(fv=fv)
    generator: Generator = cfg.instantiate()
    # Repeat requests if n in decode parameters is larger than 1.
    n = cfg.decode_parameters.get("n", 1)
    if fv.repeat_requests_for_n and n > 1:
        gen_requests = repeat_requests(gen_requests, n)
        del cfg.decode_parameters["n"]
    elif n < 1:
        raise ValueError("Decode parameters n must be larger than 0.")
    # Checks vllm readiness.
    if fv.check_vllm_readiness and "OPENAI_BASE_URL" in os.environ:
        check_vllm_readiness(
            timeout=timedelta(seconds=fv.readiness_timeout),
            base_url=os.environ["OPENAI_BASE_URL"],
        )
    # Generates responses.
    responses = await generator.async_generate_from_requests(
        gen_requests=gen_requests, **cfg.decode_parameters
    )
    # Parses responses.
    if n > 1:
        responses = flatten_responses(responses)

    return responses


async def generate_from_file(fv: FlagValues):
    """Generates responses for open api clients given a file path.

    Args:
        fv: The flag values of generator.py.

    Returns:
        None. The results are written directly to a file and logged.
    """
    # Loads requests.
    gen_requests = load_requests(
        fv.input_file,
        max_instances=fv.max_instances,
    )
    responses = await generate_from_requests(gen_requests=gen_requests, fv=fv)

    # Writes responses.
    write_responses(responses, file_path=fv.output_file)


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    asyncio.run(generate_from_file(FLAGS))


if __name__ == "__main__":
    Generator.define_flags(FLAGS)
    app.run(main)
