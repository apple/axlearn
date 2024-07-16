# Copyright © 2024 Apple Inc.

"""Code interpreter for kaggle style evaluation tasks.

Three parts:
- code generation: generate a code based on previous conversations.
- code execution: execute the generated code.
- qa: question answering on previous conversations.

Four settings:
- end-to-end: evaluated model generates code, and interprets results.
- oracle: evaluated model interprets results from oracle code execution.
- gpt4-qa: evaluated model generates code, and GPT-4 interprets results to verify code relevance.
- retry: retry failed attempts with execution feedback.

Two Modes:
- Text-only: when the grader_generator is not multimodal, only text output is evaluated.
- Multi-modal: all outputs are evaluated if the grader_generator is multimodal.
"""
import asyncio
import base64
import copy
import io
import json
import logging
import os
import re
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from tqdm import tqdm

from axlearn.open_api.common import BaseClient, EvalGeneratorType, Generator

# List of possible image MIME types.
_IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
]
_PLOTLY_MIME_TYPE = "application/vnd.plotly.v1+json"

# pylint: disable=line-too-long
_execution_feedback_template = """Your code execution failed. Please check the error message below and try again:

{error_message}

You MUST exactly follow nbformat structure when generating markdown and code cells.
Respond ONLY with the cells in json format. DO NOT say anything else or put it in a ```json``` code block.
You are only allowed to use the following math and visualization packages: numpy, pandas, scipy, scikit-learn, sympy, statsmodels, matplotlib, seaborn, plotly, wordcloud. Python standard libraries are also allowed. Make sure to import the necessary packages in the code cell.
"""

_question_template = """Given the following execution output and multiple choice questions, respond with the answers to the questions:

Execution Output:
{execution_output}

Questions:
{questions}

You MUST answer in a list of uppercase single letter strings for multiple choice questions.
"""
# pylint: enable=line-too-long


def _post_process_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhances cells with unique IDs and ensure consistent formatting.

    Args:
        cells: A list of cell dictionaries.

    Returns:
        A list of processed cell dictionaries with consistent formatting.
    """
    for cell in cells:
        if isinstance(cell, dict):
            cell["id"] = uuid.uuid4().hex[:8]
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                cell["source"] = [
                    line + "\n" if not line.endswith("\n") else line for line in source
                ]
                cell["outputs"] = []
                cell["execution_count"] = None
    return cells


def _strip_ansi_codes(text: str) -> str:
    """Removes ANSI color codes from a string."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def _execute_notebook(nb_json: str, timeout: int = 180) -> Tuple[Any, bool]:
    """Executes a Jupyter notebook from a JSON string and return the executed notebook
    along with a success status. This function strips any ANSI color codes that may
    appear in error messages.

    Args:
        nb_json: The notebook JSON as a string.
        timeout: The timeout for notebook execution in seconds. Defaults to 180.

    Returns:
        A tuple containing the executed notebook JSON and a boolean flag.
    """
    # pylint: disable=import-error,import-outside-toplevel
    # pytype: disable=import-error
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    # pytype: enable=import-error
    # pylint: enable=import-error,import-outside-toplevel
    try:
        nb = nbformat.read(io.StringIO(nb_json), as_version=4)
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb)
        return nb, True
    except nbformat.reader.NotJSONError:
        error_message = "Provided string could not be parsed as JSON."
    except TimeoutError:
        error_message = f"Notebook execution exceeded the time limit of {timeout} seconds."
    except Exception as e:  # pylint: disable=broad-except
        traceback_details = traceback.format_exc(limit=2)
        error_message = _strip_ansi_codes(
            f"An unexpected error occurred: {str(e)}\n"
            + f"Traceback (last two levels):\n{traceback_details}"
        )

    return error_message, False


def _execute_notebooks_batch(nb_jsons: List[str], timeout: int = 180, max_workers: int = 5) -> List:
    """Executes a batch of notebooks in parallel using a ThreadPoolExecutor.

    Args:
        nb_jsons: List of notebook JSON strings.
        timeout: The timeout for execution in seconds.
        max_workers: The maximum number of worker threads to use.

    Returns:
        List of results from executing the notebooks.
    """
    results = [None] * len(nb_jsons)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_execute_notebook, nb_json, timeout): i
            for i, nb_json in enumerate(nb_jsons)
        }
        for future in tqdm(
            as_completed(future_to_index), total=len(nb_jsons), desc="Executing notebooks"
        ):
            index = future_to_index[future]
            results[index] = future.result()
    return results


def _plotly_json_to_base64_png(plotly_json: Dict[str, Any]) -> str:
    """Converts Plotly JSON data to a base64 PNG image."""
    # pylint: disable-next=import-error,import-outside-toplevel
    import plotly.graph_objects as go  # pytype: disable=import-error

    fig = go.Figure(data=plotly_json["data"], layout=plotly_json.get("layout", {}))
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format="png")
    img_bytes.seek(0)  # Reset the buffer to start
    return base64.b64encode(img_bytes.read()).decode("utf-8")


def _organize_cell_output(
    cell: Dict[str, Any], image_expected: bool
) -> Tuple[Dict[str, Any], List]:
    """Organizes the cell output to replace image data with placeholders and collect image data
    in a dedicated list."""
    if not image_expected:
        return cell, []

    if cell is None or "outputs" not in cell or len(cell["outputs"]) == 0:
        return cell, []

    # Initializes the response format and keep track of images.
    image_data_list = []
    image_counter = 1

    # Replaces image data with placeholders and collect image URLs.
    relevant_outputs = []
    for output in cell.get("outputs", []):
        if "data" in output:
            for mime_type in _IMAGE_MIME_TYPES:
                if mime_type in output["data"]:
                    placeholder = f"<image_{image_counter}>"
                    image_data_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{output['data'][mime_type]}"
                            },
                        }
                    )
                    output["data"][mime_type] = placeholder
                    image_counter += 1

            if _PLOTLY_MIME_TYPE in output["data"]:
                base64_png = _plotly_json_to_base64_png(output["data"][_PLOTLY_MIME_TYPE])
                placeholder = f"<image_{image_counter}>"
                image_data_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_png}"},
                    }
                )
                output["data"][_PLOTLY_MIME_TYPE] = placeholder
                output["data"].pop("text/html", None)
                image_counter += 1

            if "text/html" in output["data"]:
                if "plotly.js" in "".join(output["data"]["text/html"]):
                    del output["data"]["text/html"]

            relevant_outputs.append(output)

    cell["outputs"] = relevant_outputs
    return cell, image_data_list


@contextmanager
def _set_code_kaggle_execution_directory():
    """Changes the working directory temporarily to a newly created tmp directory
    within the specified path.

    Yields:
        None

    Raises:
        ValueError: If the CODE_KAGGLE_DATA_DIR environment variable is not set.
    """
    current_dir = os.getcwd()
    # Checks if the CODE_KAGGLE_DATA_DIR environment variable is set.
    if "CODE_KAGGLE_DATA_DIR" not in os.environ:
        raise ValueError("CODE_KAGGLE_DATA_DIR environment variable is not set.")
    new_dir = os.path.join(os.environ["CODE_KAGGLE_DATA_DIR"], "tmp")
    # Creates the temporary directory if it does not exist.
    os.makedirs(new_dir, exist_ok=True)
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(current_dir)
        # Removes the temporary directory after finishing operations.
        shutil.rmtree(new_dir, ignore_errors=True)


def _is_multimodal_model(model_name: str) -> bool:
    """Checks if the model is multimodal by checking if its name contains any known multimodal model
    identifiers.

    Args:
        model_name: The name of the model to check.

    Returns:
        True if the model is multimodal.
    """
    # Sets of known multimodal model identifiers.
    multimodal_identifiers = {"gpt-4", "gemini", "claude-3"}

    # Returns True if any identifier is a substring of the model name.
    return any(identifier in model_name for identifier in multimodal_identifiers)


def _post_process_responses(responses: List[Dict[str, Any]], *, client_cls: Type[BaseClient]):
    """Extracts executable notebooks from responses.

    Args:
        responses: A list of responses in dictionary.
        client_cls: The type of BaseClient.
    """
    for response in responses:
        response.update({"solutions": {}})

        # Loads some necessary data from the response.
        notebook = response.get("notebook", None)
        if notebook is None:
            logging.error("No notebook in the response.")
            continue

        code_responses = []
        for resp in response.get("n_responses", [response["response"]]):
            code_response = client_cls.parse_generation(response=json.loads(resp))
            code_responses.extend(code_response)

        if len(code_responses) == 0:
            logging.error("No code responses.")
            continue

        solutions = {}
        for code_id, code_response in enumerate(code_responses):
            # Gemini cannot follow instruction to omit the ```json block.
            response_text = code_response.content.replace("```json", "").replace("```", "").strip()
            try:
                response_cells = json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.error("Error decoding the code response (%s): %s", str(e), response_text)
                continue
            response_cells = _post_process_cells(response_cells)
            preceding_conversation = copy.deepcopy(response["messages"])
            if not isinstance(preceding_conversation, list):
                continue
            preceding_conversation.append({"role": "assistant", "content": code_response.content})
            solution = copy.deepcopy(notebook)
            solution["cells"] += response_cells
            solutions[code_id] = {
                "messages": preceding_conversation,
                "notebook": json.dumps(solution),
                "execution_success": False,
                "executed_notebook": None,
                "last_cell": None,
                "answers": None,
            }
        response["solutions"] = solutions


def _execute_notebooks(responses: List[Dict[str, Any]]):
    """Executes the notebooks in the results and store the execution results.

    Args:
        responses: A dictionary of results.
    """
    execution_indices = []
    execution_notebooks = []
    for response_id, response in enumerate(responses):
        for code_id, solution in response["solutions"].items():
            if "notebook" in solution and not solution["execution_success"]:
                execution_indices.append((response_id, code_id))
                execution_notebooks.append(solution["notebook"])

    if not execution_notebooks:
        logging.error("No notebooks to execute.")
        return

    with _set_code_kaggle_execution_directory():
        execution_results = _execute_notebooks_batch(execution_notebooks)

    if not execution_results:
        logging.error("No execution results.")
        return

    executed_notebooks, execution_successes = zip(*execution_results)

    for (response_id, code_id), executed_notebook, execution_success in zip(
        execution_indices, executed_notebooks, execution_successes
    ):
        responses[response_id]["solutions"][code_id]["execution_success"] = execution_success
        responses[response_id]["solutions"][code_id]["executed_notebook"] = executed_notebook
        if execution_success:
            # Extract the last code cell with output
            try:
                last_code_cell = next(
                    cell
                    for cell in reversed(executed_notebook.cells)
                    if cell.cell_type == "code" and cell.outputs
                )
            except StopIteration:
                logging.debug("No code cell with output in the executed notebook.")
                continue
            responses[response_id]["solutions"][code_id]["last_cell"] = last_code_cell


def _populate_qa_requests(responses: List[Dict[str, Any]], *, is_mutimodal: bool):
    """Populates QA requests from execution results.

    Args:
        responses: A dictionary of responses.
        multimodal: True if it is multimodal.
    """
    for response in responses:
        for solution in response["solutions"].values():
            if not solution["execution_success"]:
                continue
            # Skip if image is expected but grader_generator is not multimodal.
            if response["image_expected"] and not is_mutimodal:
                continue
            processed_cell, image_data_list = _organize_cell_output(
                solution["last_cell"], response["image_expected"]
            )
            question_prompt = _question_template.format(
                execution_output=json.dumps(processed_cell),
                questions=json.dumps(response["questions"]),
            )
            if len(image_data_list) > 0:
                question_prompt = [{"type": "text", "text": question_prompt}] + image_data_list
            solution["messages"].append({"role": "user", "content": question_prompt})


def _generate_answers_from_qa_requests(
    responses: List[Dict[str, Any]], *, grader_generator: Generator
):
    """Generates answers from QA requests.

    Args:
        responses: A dictionary of responses.
        grader_generator: An instance of grader generator.
    """
    qa_indices = []
    qa_requests = []
    for response_id, response in enumerate(responses):
        for code_id, solution in response["solutions"].items():
            if solution["last_cell"] is not None:
                qa_requests.append({"messages": solution["messages"]})
                qa_indices.append((response_id, code_id))

    if len(qa_requests) > 0:
        answer_responses = asyncio.run(
            grader_generator.async_generate_from_requests(gen_requests=qa_requests)
        )
        for (idx, code_idx), answer_response in zip(qa_indices, answer_responses):
            answer_response = grader_generator.config.client.klass.parse_generation(
                response=json.loads(answer_response["response"])
            )
            if len(answer_response) == 0:
                logging.error("No answer response.")
                continue
            answer_response = answer_response[0].content
            answer_response = answer_response.replace("```json", "").replace("```", "").strip()
            answer_response = answer_response.replace("'", '"')
            try:
                answer = json.loads(answer_response)
                responses[idx]["solutions"][code_idx]["answers"] = answer
            except json.JSONDecodeError as e:
                logging.error(
                    "Error decoding the answer response (%s): %s", str(e), answer_response
                )
                continue


def _compute_metrics_from_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics based on responses.

    Args:
        responses: A list of responses in dictionary.

    Returns:
        A dictionary of metrics.
    """
    question_counts = []
    correct_counts = []
    image_output = []
    execution_success_count = 0
    for response in responses:
        image_output.append(response["image_expected"])
        n_questions = len(response["gt_answers"])
        question_counts.append(n_questions)
        gt_answers = np.array(response["gt_answers"])
        correct_answers = np.zeros(n_questions, dtype=bool)
        any_execution_success = False
        for solution in response["solutions"].values():
            any_execution_success = any_execution_success or solution["execution_success"]
            if solution["answers"] is not None:
                pred_answers = np.array(solution["answers"])
                if len(pred_answers) != n_questions:
                    continue
                correct_answers = np.logical_or(correct_answers, pred_answers == gt_answers)
        correct_counts.append(np.sum(correct_answers))
        if any_execution_success:
            execution_success_count += 1

    question_counts = np.array(question_counts, dtype=np.float32)
    correct_counts = np.array(correct_counts, dtype=np.float32)
    image_output = np.array(image_output, dtype=bool)
    per_response_accuracy = correct_counts / question_counts
    overall_accuracy = np.sum(correct_counts) / np.sum(question_counts)
    text_per_response_accuracy = np.mean(per_response_accuracy[~image_output])
    text_overall_accuracy = np.sum(correct_counts[~image_output]) / np.sum(
        question_counts[~image_output]
    )
    image_per_response_accuracy = np.mean(per_response_accuracy[image_output])
    image_overall_accuracy = np.sum(correct_counts[image_output]) / np.sum(
        question_counts[image_output]
    )

    return {
        "execution_success_rate": float(execution_success_count) / len(responses),
        "overall_accuracy": float(overall_accuracy),
        "text_overall_accuracy": float(text_overall_accuracy),
        "image_overall_accuracy": float(image_overall_accuracy),
        "per_response_accuracy": float(per_response_accuracy.mean()),
        "text_per_response_accuracy": float(text_per_response_accuracy),
        "image_per_response_accuracy": float(image_per_response_accuracy),
        "execution_success_count": execution_success_count,
        "total_questions": float(np.sum(question_counts)),
        "total_correct": float(np.sum(correct_counts)),
        "total_text_questions": float(np.sum(question_counts[~image_output])),
        "total_text_correct": float(np.sum(correct_counts[~image_output])),
        "total_image_questions": float(np.sum(question_counts[image_output])),
        "total_image_correct": float(np.sum(correct_counts[image_output])),
    }


def metric_fn(
    *,
    responses: List[Dict[str, Any]],
    generators: Dict[EvalGeneratorType, Generator],
    debug: bool = False,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Implements code kaggle metric following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    _post_process_responses(responses, client_cls=generator_cfg.client.klass)
    _execute_notebooks(responses)
    grader_generator = generators[EvalGeneratorType.GRADER]
    grader_generator_cfg: Generator.Config = grader_generator.config
    _populate_qa_requests(
        responses, is_mutimodal=_is_multimodal_model(grader_generator_cfg.client.model)
    )
    _generate_answers_from_qa_requests(responses, grader_generator=grader_generator)
    return _compute_metrics_from_responses(responses)


def _populate_retry_requests(responses: List[Dict[str, Any]]):
    """Prepares retry requests from execution results.

    Args:
        responses: A dictionary of responses.
    """
    for response in responses:
        for solution in response["solutions"].values():
            if not solution["execution_success"]:
                solution["messages"].append(
                    {
                        "role": "user",
                        "content": _execution_feedback_template.format(
                            error_message=solution["executed_notebook"]
                        ),
                    }
                )


def _generate_retry_solutions(responses: List[Dict[str, Any]], *, generator: Generator):
    """Generates retry requests from execution results and generate answers.

    Args:
        responses: A dictionary of responses.
        generator: An instance of generator.
    """
    retry_indices = []
    retry_requests = []
    for response_id, response in enumerate(responses):
        for code_id, solution in response["solutions"].items():
            if not solution["execution_success"]:
                retry_requests.append({"messages": solution["messages"]})
                retry_indices.append((response_id, code_id))

    if len(retry_requests) > 0:
        retry_responses = asyncio.run(
            generator.async_generate_from_requests(gen_requests=retry_requests)
        )
        for (idx, code_idx), retry_response in zip(retry_indices, retry_responses):
            retry_response = generator.config.client.klass.parse_generation(
                response=json.loads(retry_response["response"])
            )
            if len(retry_response) == 0:
                logging.error("No retry response.")
                continue
            retry_response = retry_response[0].content
            retry_response = retry_response.replace("```json", "").replace("```", "").strip()
            try:
                retry_solution = json.loads(retry_response)
            except json.JSONDecodeError as e:
                logging.error("Error decoding the retry response (%s): %s", str(e), retry_response)
                continue
            retry_cells = _post_process_cells(retry_solution)
            preceding_conversation = responses[idx]["solutions"][code_idx]["messages"]
            preceding_conversation.append({"role": "assistant", "content": retry_response})
            retry_solution = copy.deepcopy(responses[idx]["notebook"])
            retry_solution["cells"] += retry_cells
            responses[idx]["solutions"][code_idx]["notebook"] = json.dumps(retry_solution)


def retry_metric_fn(
    *,
    responses: List[Dict[str, Any]],
    generators: Dict[EvalGeneratorType, Generator],
    debug: bool = False,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Implements code kaggle with onlien retry metric following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    generator_cfg: Generator.Config = generators[EvalGeneratorType.RESPONSE].config
    _post_process_responses(responses, client_cls=generator_cfg.client.klass)
    _execute_notebooks(responses)
    _populate_retry_requests(responses)
    _generate_retry_solutions(responses, generator=generators[EvalGeneratorType.RESPONSE])
    _execute_notebooks(responses)
    grader_generator = generators[EvalGeneratorType.GRADER]
    grader_generator_cfg: Generator.Config = grader_generator.config
    _populate_qa_requests(
        responses, is_mutimodal=_is_multimodal_model(grader_generator_cfg.client.model)
    )
    _generate_answers_from_qa_requests(responses, grader_generator=grader_generator)
    return _compute_metrics_from_responses(responses)


def _post_process_responses_oracle(responses: List[Dict[str, Any]]):
    """Extracts executable notebooks from responses with oracle response.

    Args:
        responses: A list of responses in dictionary.
    """
    for response in responses:
        response["messages"].append(
            {
                "role": "assistant",
                "content": json.dumps(response["oracle_response"]),
            }
        )
        response["solutions"] = {
            0: {
                "messages": response["messages"],
                "notebook": response["notebook"],
                "execution_success": True,
                "last_cell": response["last_cell"],
                "answers": None,
            }
        }


def oracle_metric_fn(
    *,
    responses: List[Dict[str, Any]],
    generators: Dict[EvalGeneratorType, Generator],
    debug: bool = False,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Implements code kaggle with oracle response metric
        following axlearn.open_api.common.MetricFn.

    Args:
        responses: A list of responses in dictionary.
        generators: A dict of generator instances.
        debug: True to add debug information in the metric.

    Returns:
        A dictionary of metrics.
    """
    _post_process_responses_oracle(responses)
    grader_generator = generators[EvalGeneratorType.GRADER]
    grader_generator_cfg: Generator.Config = grader_generator.config
    _populate_qa_requests(
        responses, is_mutimodal=_is_multimodal_model(grader_generator_cfg.client.model)
    )
    _generate_answers_from_qa_requests(responses, grader_generator=grader_generator)
    return _compute_metrics_from_responses(responses)
