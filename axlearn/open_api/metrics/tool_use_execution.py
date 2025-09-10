# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# ShishirPatil/gorilla:
# Copyright 2023 The Gorilla Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Tool use execution evaluation task.

Following https://platform.openai.com/docs/guides/function-calling
for target message.

The file contains the code for several tool use metrics:
* Standard tool use metrics
* Lenient tool use metric
* Bag of word tool (BOW) use metric.

The lenient matching is similar to the standard metric. It performs the following steps:
* Transforms the argument strings to lower case.
* Removes punctuations.
* Removes stop words. The stop words are statically predefined.
* Compares the results argument string.

The bag of word tool use metric transforms the argument strings in the same way as the
lenient matching. But instead of comparing the resulting strings it checks if the words
in the ground truth arguments are contained in the predicted arguments.
"""

import copy
import dataclasses
import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Optional

import dateutil
import dateutil.parser

from axlearn.open_api.common import BaseClient, EvalGeneratorType, Generator
from axlearn.open_api.metrics.tool_use_execution_utils import ArgumentMatchType, check_arguments
from axlearn.open_api.openai import OpenAIClient


def _standardize_string(input_string: str) -> str:
    """Standardizes the string by removing all the spaces,
    ",./-_*^" punctuation and converting it to lowercase.
    It will also convert all the single quotes to double quotes and used to compare the model output
    with the possible answers. We don't want to punish model for answer
    like April 1, 2024 vs April 1,2024, vs April 1 2024.

    This is adopted from Gorilla:
    https://github.com/ShishirPatil/gorilla/blob/cb694ea16ecd23317e4b8d93bc9772dc0f09c4d2/berkeley-function-call-leaderboard/eval_checker/checker.py#L151
    """
    regex_string = r"[ \,\.\/\-\_\*\?\+\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def _default_value_match(pred_v: Any, target_v: Any) -> bool:
    """Checks whether a string value matches."""
    if isinstance(pred_v, str):
        pred_v = _standardize_string(pred_v)
    if isinstance(target_v, str):
        target_v = _standardize_string(target_v)
    return pred_v == target_v


def _date_match(val: str, gt: str) -> bool:
    """Interprets strings as date and then compare."""
    try:
        date_val = dateutil.parser.parse(val)
        date_gt = dateutil.parser.parse(gt)
    except dateutil.parser.ParserError:
        return False
    return date_val == date_gt


def _extract_arguments(tool_call: dict) -> dict:
    """Extracts arguments from tool call."""
    args = tool_call["arguments"]
    return json.loads(args) if isinstance(args, str) else args


def _match_argument(
    *,
    arg_name: str,
    arg_rule: dict,
    pred_arg: Any,
    target_arg: Optional[Any],
) -> bool:
    """Matches a predicted argument against an argument rule.

    Args:
        arg_name: The name of the argument.
        arg_rule: The argument rule.
        pred_arg: The predicted argument value.
        target_get: The target argument value.

    Returns:
        True if the argument is matched against the rule, False other.

    Raises:
        ValueError: If the argument rule does not contain a valid sub-rule.
    """
    if "regex_match" in arg_rule:
        if not isinstance(pred_arg, str):
            return False
        pattern = re.compile(arg_rule["regex_match"], re.IGNORECASE)
        return pattern.match(pred_arg) is not None

    if "multi_choices" in arg_rule:
        return any(
            _default_value_match(pred_arg, candidate) for candidate in arg_rule["multi_choices"]
        )

    if "date_match" in arg_rule:
        if arg_rule["date_match"] is True:
            if not isinstance(pred_arg, str):
                return False
            assert isinstance(target_arg, str)
            return _date_match(pred_arg, target_arg)
        else:
            return _default_value_match(pred_arg, target_arg)

    raise ValueError(f"Unknown argument rules with keys {list(arg_rule.keys())} fpr {arg_name}")


def _match_tool_call_with_rules(
    *,
    pred_tool_call: dict,
    match_rule: dict,
    target_tool_call: dict,
) -> bool:
    """Matches a tool call against a target tool call or a match rule. It works as follows:
    - Check the function name
    - Iterate over all match attributes:
        - Check all arguments with the attribute_rule
    - Iterate over all unseen arguments and compare them against the target attributes.

    Args:
        pred_tool_call: The predicted tool call.
        match_rule: The match rule.
        target_tool_call: The target tool call.

    Returns:
        True if the tool call matches against the target tool call or the match rule, false
        otherwise.
    """

    if target_tool_call["name"] != pred_tool_call["name"]:
        return False

    try:
        pred_args = _extract_arguments(pred_tool_call)
        target_args = _extract_arguments(target_tool_call)
    except (json.JSONDecodeError, KeyError):
        logging.error(
            "Unable to decode arguments from predicted call %s or target call %s",
            pred_tool_call["arguments"],
            target_tool_call["arguments"],
        )
        return False

    if not isinstance(pred_args, dict):
        logging.error("Arguments are not a dictionary %s", str(pred_args))
        return False

    seen_arguments = []
    if "arguments" in match_rule:
        for arg_name, rule in match_rule["arguments"].items():
            if arg_name not in pred_args:
                return False
            if not _match_argument(
                arg_name=arg_name,
                arg_rule=rule,
                pred_arg=pred_args[arg_name],
                target_arg=target_args.get(arg_name, None),
            ):
                return False
            seen_arguments.append(arg_name)

    for arg in set(list(pred_args.keys()) + list(target_args.keys())):
        if arg not in seen_arguments:
            pred_value = pred_args.get(arg, None)
            target_value = target_args.get(arg, None)
            if not _default_value_match(pred_value, target_value):
                return False

    return True


def _compare_tool_calls(
    *,
    pred_tool_calls: Sequence[dict[str, Any]],
    target_tool_calls: list[dict[str, Any]],
    match_rules: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """Compares the predicted tool calls with the target, potentially with matching rules.

    Args:
        pred_tool_calls: Predicted tool calls.
        target_tool_calls: Target tool calls.
        match_rules: A dict containing the matching rules corresponding to each target tool call.
            E.g.
            ```[{
                "param1": {
                    "multi_choices": [..,],
                    "regex_match": ...,
                    "flexible_date_match": True,
                },
                "param2": ...
            }]```
            The match rules should correspond to each target function call. If there's no rule for
            a corresponding function call, an empty dict should be used.

    Returns:
        True if predicted tool calls match the target or False otherwise.

    Raises:
        ValueError: If length of target_tool_calls don't match that of match_rules.
    """
    if match_rules is not None and len(target_tool_calls) != len(match_rules):
        raise ValueError(
            f"match_rules ({len(match_rules)}) should match the "
            f"number of target_tool_calls ({len(target_tool_calls)})."
        )

    target_tool_calls = copy.deepcopy(target_tool_calls)
    match_rules = copy.deepcopy(match_rules)
    match_rules = [{}] * len(target_tool_calls) if match_rules is None else match_rules
    for pred_tool in pred_tool_calls:
        tool_matched = False

        for call_idx, target_tool in enumerate(target_tool_calls):
            tool_matched = _match_tool_call_with_rules(
                pred_tool_call=pred_tool["function"],
                target_tool_call=target_tool["function"],
                match_rule={} if match_rules is None else match_rules[call_idx],
            )
            if tool_matched:
                target_tool_calls.pop(call_idx)
                match_rules.pop(call_idx)
                tool_matched = True
                break
        if not tool_matched:
            return False
    if target_tool_calls:
        return False
    return True


@dataclasses.dataclass
class _DetailedMatchResult:
    """Represents the tool matches for different metrics.

    Argument matches require that function name matches too. It is not possible that the
    `func_name_match` is false, while one of the arg match members is true.

    Attributes:
        func_name_match: The predicted function name matches the target function name.
        strict_arg_match: The predicted tool arguments match strictly all target tool arguments.
        lenient_arg_match: The predicted tool arguments match all target tool arguments via the
           lenient comparison.
        lenient_bow_arg_match: The predicted tool arguments match all target tool arguments via
           the lenient bag-of-work comparison.
    """

    func_name_match: bool = False
    strict_arg_match: bool = False
    lenient_arg_match: bool = False
    lenient_bow_arg_match: bool = False


def _compare_tool_call_detailed(
    *,
    pred_tool_calls: Sequence[dict[str, Any]],
    target_tool_calls: Sequence[dict[str, Any]],
) -> list[_DetailedMatchResult]:
    """Performs a detailed comparison of the predicted tool calls with the target tool calls and
    returns different metrics.

    The number of predicted and target tool calls can be different. The comparsion works by
    iterating over all predicted calls and finding the corresponding target call. Correspondence
    is defined here by the following rules:
    - Strict metric: The function name matches and all arguments are identical.
    - Lenient metric: The function name matches and all arguments match by the lenient argument
      comparision.
    - Lenient BOW metric: The function name matches and all arguments match by the lenient BOW
      comparison.

    Note that this function return the matching results for every predicted tool call.

    It is possible that strict, lenient, or lenient-bow matches correspond to different tool
    calls.

    Invalid tool calls which fail to parse or decode are ignored and considered as non
    matches.

    An example of a tool call is

    ```
    {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": "add_contact",
            "arguments": {
                "name": "John Doe",
                "phone": "1234567890",
                "email": "johndoe@example.com"
            }
        }
    }
    ```

    Args:
        pred_tool_calls: Predicted tool calls.
        target_tool_calls: Target tool calls.

    Returns:
        A list of DetailedMatchResults with the same length as `pred_tool_calls`. Each result
        indicates whether the corresponding predicted tool call matches any target tool call.
    """

    target_tool_calls = list(target_tool_calls)
    pred_tool_calls = list(pred_tool_calls)

    target_funcs = []
    for t in target_tool_calls:
        try:
            t["function"]["arguments"] = _extract_arguments(t["function"])
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(
                "Unable to decode arguments from target call %s, Error %s", t["function"], e
            )
            continue
        target_funcs.append(t["function"])

    strict_target_funcs = list(target_funcs)
    lenient_target_funcs = list(target_funcs)
    lenient_bow_target_funcs = list(target_funcs)

    results = []
    for pred_tool in pred_tool_calls:
        try:
            pred_func = pred_tool["function"]
            pred_func["arguments"] = _extract_arguments(pred_func)
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(
                "Unable to decode arguments from predicted call %s, Error %s", pred_tool, e
            )
            results.append(_DetailedMatchResult())
            continue

        fname_match = False
        strict_match = False
        lenient_match = False
        lenient_bow_match = False

        for idx, target_func in enumerate(strict_target_funcs):
            if pred_func["name"] == target_func["name"]:
                fname_match = True
            else:
                continue

            if check_arguments(
                pred_args=pred_func["arguments"],
                target_args=target_func["arguments"],
                match_type=ArgumentMatchType.STRICT,
            ):
                strict_match = True
                del strict_target_funcs[idx]
                break

        for idx, target_func in enumerate(lenient_target_funcs):
            if pred_func["name"] != target_func["name"]:
                continue

            if check_arguments(
                pred_args=pred_func["arguments"],
                target_args=target_func["arguments"],
                match_type=ArgumentMatchType.LENIENT,
            ):
                lenient_match = True
                del lenient_target_funcs[idx]
                break

        for idx, target_func in enumerate(lenient_bow_target_funcs):
            if pred_func["name"] != target_func["name"]:
                continue

            if check_arguments(
                pred_args=pred_func["arguments"],
                target_args=target_func["arguments"],
                match_type=ArgumentMatchType.LENIENT_BAG_OF_WORD,
            ):
                lenient_bow_match = True
                del lenient_bow_target_funcs[idx]
                break

        results.append(
            _DetailedMatchResult(
                func_name_match=fname_match,
                strict_arg_match=strict_match,
                lenient_arg_match=lenient_match,
                lenient_bow_arg_match=lenient_bow_match,
            )
        )
    return results


def metric_fn(
    *,
    responses: list[dict[str, Any]],
    generators: dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> dict[str, Any]:
    """Implements the tool use execution accuracy metric following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
        - number_of_examples: Total number of examples.
        - number_of_parsing_errors: Total number of examples with parsing errors.
        - number_of_generation_errors: Total number of examples with generation errors.
        - accuracy: Accuracy after standardization of argument values including removing spaces,
            punctuation, and converting to lowercases. Also apply matching rules if provided.
            See `_default_value_match()` for string standardization.
        - strict_accuracy: Strict accuracy where argument values exactly matches the
            ground truth or not.
        - lenient_accuracy: Accuracy after removing spaces, punctuation and a static list of
            stop words in the argument values.
            See `_is_arg_value_equal()` in ./tool_use_execution_utils.py for details.
        - bow_accuracy: Bag-of-words accuracy. Transforms the argument strings in the same way
            as the lenient matching. But instead of comparing the resulting strings it checks
            if the words in the ground truth argument values are contained in the predicted
            argument values.
            See `_is_arg_value_equal()` in ./tool_use_execution_utils.py for details.
        - func_name_accuracy: Accuracy of matching the ground truth tool names.
        - number_of_expected_tool_calls: Number of tool calls in ground truth.
        - number_of_func_call_intents_ground_truth: Number of assistant turns with tool calls
            in the ground truth.
        - number_of_func_call_intents_pred: Number of assistant turns with tool calls
            in predictions.
        - func_intent_recall: Recall of tool calls intent.
        - func_intent_precision: Precision of tool calls intent.

    """
    if len(responses) == 0:
        return {
            "accuracy": 0,
            "number_of_examples": 0,
            "number_of_errors": 0,
        }

    def get_tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Extracts tool call arguments in message."""

        if "tool_calls" not in message:
            return [{}]
        new_tool_calls = []
        for tool_call in message["tool_calls"]:
            if "id" in tool_call:
                del tool_call["id"]
            new_tool_calls.append(tool_call)
        return new_tool_calls

    total_matches = 0

    # The counters for the detailed metrics.
    total_tool_calls = 0
    total_func_name_matches = 0
    total_strict_matches = 0
    total_lenient_matches = 0
    total_bow_matches = 0

    number_of_func_call_intents_ground_truth = 0
    number_of_func_call_intents_pred = 0
    number_of_func_call_intents_ground_truth_pred = 0

    number_of_parsing_errors = 0
    number_of_generation_errors = 0
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    client_cls: type[BaseClient] = generator_cfg.client.klass
    for response in responses:
        matched = False
        target_message = response.get("target_message", None)
        if target_message is None:
            raise ValueError("ToolUseExecution must have target message in each response.")
        try:
            pred_messages = client_cls.parse_generation(response=json.loads(response["response"]))
        # pylint: disable-next=broad-except
        except Exception as e:
            logging.error("Found error %s: %s", type(e), e)
            logging.error("Response %s", response["response"])
            pred_messages = []
            number_of_parsing_errors += 1
        if (
            len(pred_messages) == 1
            and pred_messages[0].content == ""
            and pred_messages[0].tool_calls is None
            and pred_messages[0].function_call is None
        ):
            # If the content is empty and there are no tool or function calls we usually have
            # a generation error. In this case, there is no content field generated, but
            # sometimes an error field.
            number_of_generation_errors += 1
        pred_tool_calls, target_tool_calls = None, None

        target = OpenAIClient.format_message(target_message)

        if target.tool_calls:
            target_tool_calls = get_tool_calls_from_message(target.model_dump())
            total_tool_calls += len(target_tool_calls)
            number_of_func_call_intents_ground_truth += 1

        if len(pred_messages) > 0:
            pred = pred_messages[0]
            if pred.tool_calls:
                number_of_func_call_intents_pred += 1
                if target.tool_calls:
                    number_of_func_call_intents_ground_truth_pred += 1

            # Check string match.
            if (
                target.content is not None
                and target.content != ""
                and target.content == pred.content
            ):
                matched = True
            elif (
                target.tool_calls
                and pred.tool_calls
                and len(target.tool_calls) == len(pred.tool_calls)
            ):
                pred_tool_calls = get_tool_calls_from_message(pred.model_dump())
                match_rules = response.get("target_message_match_rules", None)
                matched = _compare_tool_calls(
                    pred_tool_calls=pred_tool_calls,
                    target_tool_calls=target_tool_calls,
                    match_rules=match_rules,
                )

            if target.tool_calls and pred.tool_calls:
                # Run the detailed too call matching.
                pred_tool_calls = get_tool_calls_from_message(pred.model_dump())
                detailed_results = _compare_tool_call_detailed(
                    pred_tool_calls=pred_tool_calls, target_tool_calls=target_tool_calls
                )
                total_func_name_matches += sum(1 for d in detailed_results if d.func_name_match)
                total_strict_matches += sum(1 for d in detailed_results if d.strict_arg_match)
                total_lenient_matches += sum(1 for d in detailed_results if d.lenient_arg_match)
                total_bow_matches += sum(1 for d in detailed_results if d.lenient_bow_arg_match)

        if debug and not matched:
            deliverable_id = response.get("deliverable_id", response.get("id", ""))
            target = target_tool_calls or json.dumps(target_message, sort_keys=True)
            pred = (
                pred_tool_calls or json.dumps(pred_messages[0].model_dump(), sort_keys=True)
                if len(pred_messages) > 0
                else ""
            )
            debug_info = (
                f"deliverable_id: {deliverable_id}\n"
                + f"target: {target}\n"
                + f"pred: {pred}\n"
                + f"resp: {response['response']}"
            )
            logging.debug(debug_info)
        if matched:
            total_matches += 1

    func_intent_recall = number_of_func_call_intents_ground_truth_pred / max(
        number_of_func_call_intents_ground_truth, 1
    )
    func_intent_precision = number_of_func_call_intents_ground_truth_pred / max(
        number_of_func_call_intents_pred, 1
    )
    return {
        "accuracy": total_matches / len(responses),
        "number_of_examples": len(responses),
        "number_of_parsing_errors": number_of_parsing_errors,
        "number_of_generation_errors": number_of_generation_errors,
        "func_name_accuracy": total_func_name_matches / max(1, total_tool_calls),
        "strict_accuracy": total_strict_matches / max(1, total_tool_calls),
        "lenient_accuracy": total_lenient_matches / max(1, total_tool_calls),
        "bow_accuracy": total_bow_matches / max(1, total_tool_calls),
        "number_of_expected_tool_calls": total_tool_calls,
        "number_of_func_call_intents_ground_truth": number_of_func_call_intents_ground_truth,
        "number_of_func_call_intents_pred": number_of_func_call_intents_pred,
        "func_intent_recall": func_intent_recall,
        "func_intent_precision": func_intent_precision,
    }
