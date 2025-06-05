# Copyright © 2024 Apple Inc.

"""CodeContests evaluation task.

Evaluate the correctness of generated code, problem understanding or plans.
"""
import asyncio
import copy
import json
import logging
from enum import Enum
from typing import Any

from axlearn.open_api.common import (
    BaseClient,
    EvalGeneratorType,
    Generator,
    flatten_responses,
    repeat_requests,
)
from axlearn.open_api.metrics.code_execute import check_correctness

# pylint: disable=line-too-long
_plan2code_prompt_template = """You are an AI with advanced code generation and instruction following capabilities.

When you encounter a specific problem (labeled #PROBLEM#) and a solving plan (labeled #PLAN#), your goal is to produce a valid python code that correctly solves the problem.
Make sure to fully address the problem goals following the rules and constraints. Refer to the plan to generate the code.
The code should be robust and general. It should output correct answers under any valid inputs, not only just the example inputs given in the problem description.

#PROBLEM#:
{description}

#PLAN#:
{plan}

guidelines:
- Generate only code, without any additional explanations or comments.
- Make sure to include all the necessary module imports, properly initialize the variables, and address the problem constraints.
- The code needs to be self-contained, and executable as-is.

The code output must follow this structure:
```
def f1(...):
    ...
    return ...

def f2(...):
    ...
    return ...
...

if __name__ == "__main__":
    ...
```
The code should read the input using the 'input()' method. Make sure to properly parse the input, according to the problem description.
The output should be printed without additional words using the 'print()' method.

answer:
```python

"""
# pylint: enable=line-too-long


class CodeContestStatus(str, Enum):
    """Possible status of solving CodeContests problems."""

    SUCCESS = "success"
    ERROR = "runtime_error"
    MISMATCH = "wrong_output"
    EMPTY = "empty_solution"


def _parse_answer(gen: str) -> str:
    """Extracts answer from the response.

    Args:
        gen: Generated response content str.

    Returns:
        Extracted answer str.
    """
    # The instruction asks the model to generate answer after #ANSWER#:.
    if "#ANSWER#" in gen:
        temp = gen.split("#ANSWER#:")[-1]
        return temp
    elif "ANSWER" in gen:
        # Relaxes the instruction template for model outputs that do not have #.
        temp = gen.split("ANSWER")[-1]
        # Solves corner case with ANSWER\n.
        if "\n" in temp:
            temp = temp.split("\n")[1]
        return temp
    else:
        return ""


def _format_gen_code(gen: str) -> str:
    """Formats the generated code into executable python code.

    Args:
        gen: Generated code str.

    Returns:
        Re-formated python code.
    """
    # Some models will generate some non-code explanations in the end.
    # Hence we extract the code block between ```python and ```.
    if "```python" in gen:
        gen = gen.split("```python")[1]
    if "```" in gen:
        gen = gen.split("```")[0]
    # Remove if main for local execution.
    if "if __name__" in gen:
        lines = gen.split("\n")
        new_lines = []
        enter_main = False
        for l in lines:
            if "if __name__" in l:
                enter_main = True
            else:
                if enter_main:
                    new_lines.append(l[4:])
                else:
                    new_lines.append(l)
        gen = "\n".join(new_lines)
    return gen.strip()


def _check_execution_output_match(pred: str, *, expected: str) -> bool:
    """Checks whether the generated code's output matches the expected.

    Args:
        pred: Generated output.
        expected: Expected groundtruth output.

    Returns:
        Bool value of whether the outputs match.
    """
    pred = pred.strip()
    expected = expected.strip()
    if pred.replace(" ", "") == expected.replace(" ", ""):
        return True
    try:
        if abs(float(pred) - float(expected)) / max(1.0, abs(expected)) <= 1e-4:
            return True
    # ValueError: If pred or expected are string type.
    # TypeError: If pred or expected are dictionary or list type.
    except (ValueError, TypeError):
        return False
    return False


def _check_understand_str_match(pred: str, *, expected: str) -> bool:
    """Checks whether the predicted understanding string matches the expected.

    Args:
        pred: Predicted output.
        expected: Expected groundtruth output.

    Returns:
        Bool value of whether the outputs match.
    """
    # Check matches line by line.
    # Ignore start and end space via strip.
    pred = [p for p in pred.split("\n") if p]
    expected = [e for e in expected.split("\n") if e]

    if len(pred) == len(expected):
        for p, e in zip(pred, expected):
            if p.strip() != e.strip():
                return False
        return True
    return False


def _exec_and_check(
    gen_code: str,
    *,
    problem: dict[str, Any],
    problem_id: str,
    candidate_id: str,
    save_errors: bool = False,
) -> dict[str, Any]:
    """Executes generated code and checks whether it can pass all the tests of a problem.

    Args:
        gen_code: Generated code string.
        problem: Expected groundtruth output.
        problem_id: An identifier of the problem.
        candidate_id: An identifier of the generated code candidate.
        save_errors: Whether to save the error log.

    Returns:
        A dictionary of execution results.
    """
    if not gen_code:
        return {"passed": False, "score": 0.0, "errors": {"status": CodeContestStatus.EMPTY}}

    total_tests = 0
    total_passed = 0
    err_log = []

    for eval_type in ["public_tests", "private_tests", "generated_tests"]:
        eval_tests = problem.get(eval_type, None)
        if not eval_tests:
            continue
        eval_inputs = eval_tests.get("input", [])
        eval_outputs = eval_tests.get("output", [])
        results = check_correctness(
            check_program=gen_code,
            inputs=eval_inputs,
            timeout=3.0,
            task_id=problem_id,
            completion_id=candidate_id,
        )
        # Results format:
        # Success: {"task_id": task_id, "completion_id": completion_id, "result": result}.
        # Failure: {"error": "timeout", "result_list": []}.
        if results["result"]["error"] is not None:
            total_tests += len(eval_inputs)
        else:
            assert (
                len(results["result"]["result_list"]) == len(eval_inputs) == len(eval_outputs)
            ), ValueError("Non matched length exec_res, test_input, expected_output")
            for exec_res, test_input, expected_output in zip(
                results["result"]["result_list"],
                eval_inputs,
                eval_outputs,
            ):
                total_tests += 1
                if not exec_res["passed"]:
                    if save_errors:
                        err_log.append(
                            {
                                "status": CodeContestStatus.ERROR,
                                "test_input": test_input,
                                "error_message": exec_res["error"],
                            }
                        )
                elif _check_execution_output_match(
                    pred=exec_res["output"], expected=expected_output
                ):
                    total_passed += 1
                elif save_errors:
                    err_log.append(
                        {
                            "status": CodeContestStatus.MISMATCH,
                            "test_input": test_input,
                            "expected_output": expected_output,
                            "code_output": exec_res["output"],
                        }
                    )

    if total_tests == 0:
        logging.error("No test found for problem %s", problem_id)
        return {"passed": False, "score": 0.0, "errors": []}

    return {
        "passed": total_tests == total_passed,
        "score": total_passed * 1.0 / total_tests,
        "errors": err_log,
    }


def code_execution_metric_fn(
    *,
    responses: list[dict[str, Any]],
    generators: dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> dict[str, Any]:
    """Implements code contests metric following axlearn.open_api.common.MetricFn.
    This computes the correctness of the generated code, returns pass rate and
    average score of the generated code on all test cases per problem,
    corresponding to the E2E standard task in the paper.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    total_passed = 0
    num_candidates = []
    global_stats = {}
    errors = {}
    avg_scores = []
    client_cls: type[BaseClient] = generators[EvalGeneratorType.RESPONSE].config.client.klass
    for resp_id, response in enumerate(responses):
        candidates: list[str] = []
        pid = response["id"]
        if "n_responses" in response:
            for resp in response["n_responses"]:
                gen = client_cls.parse_generation(response=json.loads(resp))
                if len(gen) > 0:
                    candidates.append(_format_gen_code(gen[0].content))
                else:
                    logging.debug(
                        "deliverable_id: %s with empty response.",
                        response.get("deliverable_id", response.get("id", "")),
                    )
                    candidates.append("")
        else:
            gen = client_cls.parse_generation(response=json.loads(response["response"]))
            if len(gen) > 0:
                candidates.append(_format_gen_code(gen[0].content))
            else:
                logging.debug(
                    "deliverable_id: %s with empty response.",
                    response.get("deliverable_id", response.get("id", "")),
                )
                candidates.append("")
        num_candidates.append(len(candidates))
        correct_candidates = 0
        scores = []
        errors[pid] = {}
        for cand_id, candidate in enumerate(candidates):
            result = _exec_and_check(
                gen_code=candidate,
                problem=response,
                problem_id=pid,
                candidate_id=f"candidate_{cand_id}",
                save_errors=debug,
            )
            scores.append(result["score"])
            if result["passed"]:
                correct_candidates += 1
            errors[pid][f"candidate_{cand_id}"] = result["errors"]
        if len(candidates) > 0 and resp_id % 10 == 0:
            logging.info(
                "problem %d acceptance rate: %.4f.",
                resp_id,
                correct_candidates * 1.0 / len(candidates),
            )
        else:
            logging.debug("found no solution to problem %d.", resp_id)
        if correct_candidates > 0:
            total_passed += 1
        avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
        avg_scores.append(avg_score)
        global_stats[resp_id] = {
            "id": pid,
            "total_candidates": len(candidates),
            "passed_candidates": correct_candidates,
            "avg_score": avg_score,
        }

    metrics = {
        "pass_rate": total_passed * 1.0 / len(responses),
        "average_score": sum(avg_scores) / len(responses),
        "candidates_per_example": sum(num_candidates) / len(responses),
        "number_of_examples": len(responses),
    }
    if debug:
        metrics.update({"errors": errors, "global_stats": global_stats})
    return metrics


def understand_metric_fn(
    *,
    responses: list[dict[str, Any]],
    generators: dict[EvalGeneratorType, Generator],
    debug: bool = False,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """Implements code contests understanding metric following axlearn.open_api.common.MetricFn.
    Checks whether the understanding output from a coding question can match target string.
    No execution is needed.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    correct = 0
    total_valid_ans = 0
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    client_cls: type[BaseClient] = generator_cfg.client.klass
    for response in responses:
        target_output = response.get("target_output", None)
        if target_output is None:
            raise ValueError(
                "CodeContests Understand tasks must have target_output in each response."
            )
        pred = client_cls.parse_generation(response=json.loads(response["response"]))
        if len(pred) == 0:
            logging.debug(
                "deliverable_id: %s with empty response.",
                response.get("deliverable_id", response.get("id", "")),
            )
            continue

        answer = _parse_answer(pred[0].content)
        if not answer:
            logging.debug(
                "deliverable_id: %s", response.get("deliverable_id", response.get("id", ""))
            )
            logging.debug("Unable to find #ANSWER# in %s", pred[0].content)
            continue
        total_valid_ans += 1
        if _check_understand_str_match(pred=answer, expected=target_output):
            correct += 1

    return {
        "accuracy": correct * 1.0 / len(responses),
        "valid_ans_rate": total_valid_ans * 1.0 / len(responses),
        "number_of_examples": len(responses),
    }


# Repeat requests by the following number to do sampling for solving a code problem.
_solver_candidates = 5


def plan_metric_fn(
    *,
    responses: list[dict[str, Any]],
    generators: dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> dict[str, Any]:
    """Implements code contests plan metric following axlearn.open_api.common.MetricFn.
    Based on the plan, generate the code, execute the code and measure the accuracy.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    grader_generator = generators[EvalGeneratorType.GRADER]
    plan2code_requests = []
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    client_cls: type[BaseClient] = generator_cfg.client.klass
    for response in responses:
        plan = client_cls.parse_generation(response=json.loads(response["response"]))
        if len(plan) == 0:
            logging.debug(
                "deliverable_id: %s with empty response.",
                response.get("deliverable_id", response.get("id", "")),
            )
            continue
        plan = plan[0].content
        plan2code_prompt = _plan2code_prompt_template.format(
            description=response["description"], plan=plan
        )
        # Make a copy to prepare a new request.
        resp = copy.deepcopy(response)
        del resp["response"]
        if "prompt" in resp:
            del resp["prompt"]
        resp.update({"messages": [{"role": "user", "content": plan2code_prompt}]})
        plan2code_requests.append(resp)

    plan2code_requests = repeat_requests(plan2code_requests, _solver_candidates)
    plan2code_responses = asyncio.run(
        grader_generator.async_generate_from_requests(gen_requests=plan2code_requests)
    )
    plan2code_responses = flatten_responses(plan2code_responses)

    # Evaluate generated code.
    metrics = code_execution_metric_fn(
        responses=plan2code_responses,
        generators={
            EvalGeneratorType.GRADER: grader_generator,
            EvalGeneratorType.RESPONSE: grader_generator,
        },
        debug=debug,
    )
    if debug:
        for resp in plan2code_responses:
            del (
                resp["description"],
                resp["public_tests"],
                resp["private_tests"],
                resp["generated_tests"],
            )
        metrics.update({"solver_responses": plan2code_responses})
    return metrics


# Repeat requests by the following number
# to do sampling for solving a code problem as a retry from previous failed error.
_retry_candidates = 5
_system_prompt = (
    "A conversation between a user and a helpful code-generation assistant. "
    "The assistant can directly generate code to help with user queries. "
    "Retry the code generation if it does not succeed or if there are errors "
    "in the previous generated code."
)
_wrong_output_template = """\nExecution result {num}: Wrong Output.
Test input: {input}
Expected output: {expected}
Actual output: {output}\n
"""
_runtime_error_template = """\nExecution result {num}: Runtime Error.
Test input: {input}
Error Message: {error}\n
"""


def _gen_retry_request(
    responses: list[dict[str, Any]],
    *,
    generator: Generator,
    eval_results: dict[str, Any],
    error_num: int = 1,
) -> list[Any]:
    """Generates retry requests for failed responses.

    Args:
        responses: A list of previous responses in dictionary.
        generator: An instance of generator.
        eval_results: A dictionary of evaluation results.
        error_num: Number of error.

    Returns:
        A list of requests.
    """
    retry_requests = []
    for resp_id, response in enumerate(responses):
        pid = response["id"]
        assert pid == eval_results["global_stats"][resp_id]["id"], "Order and id does not match."
        if eval_results["global_stats"][resp_id]["passed_candidates"] > 0:
            continue
        messages = copy.deepcopy(response["messages"])
        # Add system messages.
        messages[0]["content"] = f"system\n{_system_prompt}\n\nuser\n" + messages[0]["content"]
        # Load the first response.
        gen = generator.config.client.klass.parse_generation(
            response=json.loads(response["response"])
        )
        if len(gen) > 0:
            messages.append(
                {
                    "content": gen[0].content,
                    "role": "assistant",
                }
            )
        else:
            # edge case: no solution generated
            messages.append(
                {
                    "content": ".",
                    "role": "assistant",
                }
            )
        err_msg = ""
        if messages[-1]["content"] == ".":
            err_msg = "\n\nNo solution is generated. Please try again.\n```python"
        elif isinstance(eval_results["errors"][pid]["candidate_0"], list):
            errors = eval_results["errors"][pid]["candidate_0"][:error_num]
            for eid, err in enumerate(errors):
                if err["status"] == "wrong_output":
                    err_msg += _wrong_output_template.format(
                        num=eid,
                        input=err["test_input"],
                        expected=err["expected_output"],
                        output=err["code_output"],
                    )
                elif err["status"] == "runtime_error":
                    err_msg += _runtime_error_template.format(
                        num=eid, input=err["test_input"], error=err["error_message"]
                    )
            err_msg += "\n\nPlease try again.\n```python"
        else:
            err_msg += "\n\nFailed to parse or compile the solution. Please try again.\n```python"
        messages.append({"content": err_msg, "role": "user"})
        retry_requests.append(
            {
                "id": response["id"],
                "messages": messages,
                "public_tests": response["public_tests"],  # tests for evaluation
                "private_tests": response["private_tests"],
                "generated_tests": response["generated_tests"],
            }
        )
    return retry_requests


def retry_metric_fn(
    *,
    responses: list[dict[str, Any]],
    generators: dict[EvalGeneratorType, Generator],
    debug: bool = False,
) -> dict[str, Any]:
    """Implements code contests retry metric following axlearn.open_api.common.MetricFn.
    Retry failed problems to test whether the model can run self-correction.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """

    # Evaluate the first-round responses.
    logging.debug("Evaluating initial generated code.")
    init_metrics = code_execution_metric_fn(responses=responses, generators=generators, debug=debug)
    # Generate retry responses.
    retry_requests = _gen_retry_request(
        responses, generator=generators[EvalGeneratorType.RESPONSE], eval_results=init_metrics
    )
    logging.debug("Retrying for %d failed problems.", len(retry_requests))

    grader_generator = generators[EvalGeneratorType.GRADER]
    retry_requests = repeat_requests(retry_requests, _retry_candidates)
    retry_responses = asyncio.run(
        grader_generator.async_generate_from_requests(gen_requests=retry_requests)
    )
    retry_responses = flatten_responses(retry_responses)

    # Then evaluate generated code.
    logging.debug("Evaluating re-generated code.")
    return code_execution_metric_fn(
        responses=retry_responses,
        generators=generators,
        debug=debug,
    )
