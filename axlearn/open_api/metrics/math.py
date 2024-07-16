# Copyright © 2024 Apple Inc.

"""Math evaluation tasks.

LLM grader would be used to judge the math solution.
"""
import asyncio
import copy
import json
import logging
import re
from typing import Any, Dict, List, Type

from axlearn.open_api.common import BaseClient, EvalGeneratorType, Generator

# pylint: disable=line-too-long
_math_prompt_template = """As a math expert, you will be provided with three items: a #Question#, an #Answer#, and the #Ground Truth#.
Your task is to determine whether the #Answer# matches the #Ground Truth#.
You need to considering mathmetical theorems when verifing the answer. For example, '$(c + 9)(c - 4)$' and '(c - 4)*(c + 9)'should be
the same expressions based on factorization theorems.
If they align, respond with 'correct'. If they do not, respond with 'wrong'.
Ensure that your #Response# is the final line and consists solely of the word 'correct' or 'wrong', without any additional commentary or explanation.

#Question#:
{question}

#Answer#:
{answer}

#Ground Truth#:
{ground_truth}

#Response#:
"""
# pylint: enable=line-too-long


def _get_judgement_from_generation(resp: str) -> str:
    """Gets judgment from generation."""
    matches = re.findall(r"```python(.*?)```", resp, re.DOTALL)
    if matches:
        return matches[-1].strip().lower()
    return resp.strip().lower()


def metric_fn(
    *,
    responses: List[Dict[str, Any]],
    generators: Dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> Dict[str, Any]:
    """Implements math accuracy metric following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    # Valid answers from first round of generation.
    total_valid_answers = 0
    # Judgements from LLM grading.
    correct_judgments = 0
    incorrect_judgments = 0
    invalid_judgments = 0
    judgement_requests = []
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    client_cls: Type[BaseClient] = generator_cfg.client.klass
    for response in responses:
        pred = client_cls.parse_generation(response=json.loads(response["response"]))
        if len(pred) == 0:
            logging.debug(
                "deliverable_id: %s with empty response.",
                response.get("deliverable_id", response.get("id", "")),
            )
            continue
        if "#ANSWER#" not in pred[0].content:
            logging.debug(
                "deliverable_id: %s", response.get("deliverable_id", response.get("id", ""))
            )
            logging.debug("Unable to find #ANSWER# in %s", pred[0].content)
            continue
        total_valid_answers += 1
        judgement_prompt = _math_prompt_template.format(
            question=response["raw_problem"], answer=pred, ground_truth=response["gt_answer"]
        )
        # Make a copy to prepare a new request.
        resp = copy.deepcopy(response)
        del resp["response"]
        resp.update({"messages": [{"role": "user", "content": judgement_prompt}]})
        judgement_requests.append(resp)

    grader_generator = generators[EvalGeneratorType.GRADER]
    judgement_responses = asyncio.run(
        grader_generator.async_generate_from_requests(gen_requests=judgement_requests)
    )
    grader_client_cls: Type[BaseClient] = grader_generator.config.client.klass
    for response in judgement_responses:
        pred = grader_client_cls.parse_generation(response=json.loads(response["response"]))
        if len(pred) == 0:
            invalid_judgments += 1
            continue
        judgement = _get_judgement_from_generation(pred[0].content.strip())

        if judgement == "correct":
            correct_judgments += 1
        elif judgement == "wrong":
            incorrect_judgments += 1
        else:
            invalid_judgments += 1

    if total_valid_answers - invalid_judgments == 0:
        accuracy = 0.0
    else:
        accuracy = correct_judgments / (total_valid_answers - invalid_judgments)
    metrics = {
        "number_of_examples": len(responses),
        "accuracy": accuracy,
        "total_valid_answers": total_valid_answers,
        "correct_judgments": correct_judgments,
        "invalid_judgments": invalid_judgments,
        "incorrect_judgments": incorrect_judgments,
    }
    if debug:
        metrics.update({"judgement_responses": judgement_responses})
    return metrics
