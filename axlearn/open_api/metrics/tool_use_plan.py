# Copyright © 2024 Apple Inc.

"""Tool use plan evaluation task.

Generating an DAG graph based a list of tools
and then ask the model to select the right plan
from multi choices.
"""
import json
import logging
import re
from typing import Any, Dict, List, Type

from axlearn.open_api.common import BaseClient, EvalGeneratorType, Generator


def metric_fn(
    *,
    responses: List[Dict[str, Any]],
    generators: Dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> Dict[str, Any]:
    """Implements the tool use plan accuracy metric following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    total_matches = 0
    if_error = 0
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    client_cls: Type[BaseClient] = generator_cfg.client.klass
    for response in responses:
        matched = False
        target_plan_number = response.get("target_plan_number", None)
        if target_plan_number is None:
            raise ValueError("ToolUsePlan must have target_plan_number in each response.")
        pred = client_cls.parse_generation(response=json.loads(response["response"]))
        if len(pred) > 0:
            matches = re.findall(r"\*{0,2}Chosen Plan:\*{0,2}\s+(\d+)", pred[0].content)

            if matches:
                try:
                    pred_plan_number = int(matches[0])
                    if target_plan_number == pred_plan_number:
                        matched = True
                except ValueError:
                    logging.error("Unable to cast %s", matches[0])
                    if_error += 1
            else:
                if_error += 1
        if debug and not matched:
            deliverable_id = response.get("deliverable_id", response.get("id", ""))
            debug_info = (
                f"deliverable_id: {deliverable_id}\n"
                + f"target plan number: {target_plan_number}\npred: {pred}"
            )
            logging.debug(debug_info)
        if matched:
            total_matches += 1
    metrics = {
        "accuracy": total_matches / len(responses),
        "instruction_following_error": if_error / len(responses),
        "number_of_examples": len(responses),
    }
    return metrics
