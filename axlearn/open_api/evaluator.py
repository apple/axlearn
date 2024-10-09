# Copyright © 2024 Apple Inc.

"""Evaluates and computes metrics for responses from generator."""
import asyncio
import logging
import os
from datetime import timedelta
from typing import Optional

from absl import app, flags, logging
from absl.flags import FlagValues

from axlearn.open_api.common import (
    BaseClient,
    EvalGeneratorType,
    EvalSet,
    Evaluator,
    Generator,
    check_vllm_readiness,
    load_jsonl_file,
    load_requests,
    parse_decode_parameters,
    write_metrics,
    write_responses,
)
from axlearn.open_api.generator import generate_from_requests, generator_config_from_flags
from axlearn.open_api.registry import ClientRegistry, EvalSetRegistry, MetricRegistry

FLAGS = flags.FLAGS


def initialize_evaluator(fv: FlagValues) -> Evaluator:
    """Initializes an Evaluator from flag values.

    Args:
        fv: The flag values of evaluator.py.

    Returns:
        An instance of Evaluator.

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
        grader_client_cls: Optional[type[BaseClient]] = ClientRegistry.load_client_cls(
            fv.grader_client_name
        )
        if grader_client_cls is None:
            raise ValueError("Grader client class must not be empty.")
        grader_client_cfg = grader_client_cls.default_config().set(
            model=fv.grader_model or fv.model,
            timeout=fv.timeout,
            extra_body=grader_extra_body,
            generator_type=EvalGeneratorType.GRADER,
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
    return evaluator


def evaluate_from_file(fv: FlagValues):
    """Evaluates and computes for responses from open api clients given a file path.

    Args:
        fv: The flag values of evaluator.py.

    Returns:
        None. The results are written directly to a file and logged.
    """
    evaluator = initialize_evaluator(fv=fv)

    metric_fn = MetricRegistry.load_metric(metric_name=fv.metric_name)
    if metric_fn is None:
        raise ValueError("Metric function must not be empty.")
    evaluator.evaluate(input_file=fv.input_file, output_file=fv.output_file, metric_fn=metric_fn)


def evaluate_from_eval_set(fv: FlagValues):
    """Evaluates and computes for responses from open api clients given an eval set name.
    For inputs, it is defined by load_requests for each eval set, which can come from
    local directory or huggingface datasets repo.

    For outputs, it will generate a metrics json for each eval and an aggregated metrics json file.

    Args:
        fv: The flag values of evaluator.py.

    Returns:
        ValueError: Couldn't find eval set.
    """
    eval_set_cls: Optional[type[EvalSet]] = EvalSetRegistry.load(name=fv.eval_set_name)
    if eval_set_cls is None:
        raise ValueError(f"Couldn't find {fv.eval_set_name}")
    eval_set = eval_set_cls()
    evaluator = initialize_evaluator(fv=fv)
    if fv.output_file is not None:
        os.makedirs(fv.output_file, exist_ok=True)
    metrics_set = []
    for metric_name in eval_set.get_metrics():
        # Load requests.
        if fv.input_file is not None:
            gen_requests = load_requests(
                os.path.join(fv.input_file, metric_name + ".jsonl"),
                max_instances=fv.max_instances,
            )
        else:
            gen_requests = eval_set.load_requests(metric_name=metric_name, local_dir=fv.local_dir)
            if fv.max_instances is not None:
                gen_requests = gen_requests[: fv.max_instances]
        # Generate responses.
        responses = None
        response_output_file = None
        if fv.output_file is not None:
            response_output_file = os.path.join(
                fv.output_file, f"{fv.model}_{metric_name}_response.jsonl"
            )
            if os.path.exists(response_output_file):
                responses = load_jsonl_file(response_output_file)
        if responses is None:
            responses = asyncio.run(generate_from_requests(gen_requests=gen_requests, fv=fv))
            if response_output_file is not None:
                write_responses(responses, file_path=response_output_file)

        metric_fn = MetricRegistry.load_metric(metric_name=metric_name)
        metrics = metric_fn(
            responses=responses,
            generators=evaluator.generators,
            debug=evaluator.config.debug,
        )
        metrics_set.append(metrics)

        logging.info("Eval metric name: %s and metrics: %s", metric_name, metrics)
        if fv.output_file is not None:
            output_file = os.path.join(fv.output_file, f"{fv.model}_{metric_name}_metrics.json")
            write_metrics(metrics=metrics, file_path=output_file)
    if len(metrics_set) > 1:
        aggregated_metrics = eval_set.aggregate_metrics(metrics=metrics_set)
        logging.info("Eval aggregated metrics: %s", aggregated_metrics)
        if fv.output_file is not None:
            output_file = os.path.join(fv.output_file, f"{fv.model}_aggregated_metrics.json")
            write_metrics(metrics=aggregated_metrics, file_path=output_file)


def main(_):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    if FLAGS.eval_set_name is not None:
        evaluate_from_eval_set(FLAGS)
        return
    evaluate_from_file(FLAGS)


if __name__ == "__main__":
    Generator.define_flags(fv=FLAGS)
    Evaluator.define_flags(fv=FLAGS)
    app.run(main)
