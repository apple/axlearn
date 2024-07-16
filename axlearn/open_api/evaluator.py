# Copyright © 2024 Apple Inc.

"""Evaluates and computes metrics for responses from generator."""
import logging
import os
from datetime import timedelta
from typing import Optional, Type

from absl import app, flags, logging
from absl.flags import FlagValues

from axlearn.open_api.common import (
    BaseClient,
    EvalGeneratorType,
    Evaluator,
    Generator,
    check_vllm_readiness,
    parse_decode_parameters,
)
from axlearn.open_api.generator import generator_config_from_flags
from axlearn.open_api.registry import ClientRegistry, MetricRegistry

FLAGS = flags.FLAGS


def evaluate_from_file(fv: FlagValues):
    """Evaluates and computes for responses from open api clients given a file path.

    Args:
        fv: The flag values of evaluator.py.

    Returns:
        None. The results are written directly to a file and logged.

    Raises:
        ValueError: Client class must not be empty.
        ValueError: Grader client class must not be empty.
        ValueError: Metric function must not be empty.
        ValueError: Either default model name or grader model name must be set.
    """
    generator_cfg: Generator.Config = generator_config_from_flags(fv=fv)
    # Checks vllm readiness.
    if fv.check_vllm_readiness and "OPENAI_BASE_URL" in os.environ:
        check_vllm_readiness(
            timeout=timedelta(seconds=fv.readiness_timeout),
            base_url=os.environ["OPENAI_BASE_URL"],
        )

    grader_generator_cfg = generator_cfg.clone()
    grader_decode_parameters = grader_generator_cfg.decode_parameters
    if fv.grader_decode_parameters is not None:
        grader_decode_parameters, grader_extra_body = parse_decode_parameters(
            fv.grader_decode_parameters, model=fv.grader_model or fv.model
        )
    else:
        grader_extra_body = generator_cfg.client.extra_body
    if fv.grader_model is not None:
        grader_decode_parameters.update({"model": fv.grader_model})
    grader_generator_cfg.set(decode_parameters=grader_decode_parameters)
    if fv.grader_client_name is not None:
        if fv.model is None and fv.grader_model is None:
            raise ValueError("Either default model name or grader model name must be set.")
        # Initializes grader generator.
        grader_client_cls: Optional[Type[BaseClient]] = ClientRegistry.load_client_cls(
            fv.grader_client_name
        )
        if grader_client_cls is None:
            raise ValueError("Grader client class must not be empty.")
        grader_client_cfg = grader_client_cls.default_config().set(
            model=fv.grader_model or fv.model,
            timeout=fv.timeout,
            extra_body=grader_extra_body,
        )
        grader_generator_cfg.client = grader_client_cfg

    evaluator: Evaluator = (
        Evaluator.default_config()
        .set(
            generators={
                EvalGeneratorType.RESPONSE: generator_cfg,
                EvalGeneratorType.GRADER: grader_generator_cfg,
            },
            debug=fv.debug,
        )
        .instantiate()
    )

    metric_fn = MetricRegistry.load_metric(metric_name=fv.metric_name)
    if metric_fn is None:
        raise ValueError("Metric function must not be empty.")
    evaluator.evaluate(input_file=fv.input_file, output_file=fv.output_file, metric_fn=metric_fn)


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    evaluate_from_file(FLAGS)


if __name__ == "__main__":
    Generator.define_flags(fv=FLAGS)
    Evaluator.define_flags(fv=FLAGS)
    app.run(main)
