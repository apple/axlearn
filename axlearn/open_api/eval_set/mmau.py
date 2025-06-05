# Copyright © 2024 Apple Inc.

"""MMAU eval data loader."""
import glob
import os
from typing import Any, List, Optional

from huggingface_hub import snapshot_download

from axlearn.open_api.common import EvalSet, load_jsonl_file


def _load_requests_from_local_dir(
    *,
    local_dir: str,
    allow_pattern: str,
) -> list[dict[str, Any]]:
    """Loads Open API style requests from a local directory.

    Args:
        local_dir: The path of local directory.
        allow_pattern: The glob pattern to match the file.

    Returns:
        A list of input requests.
    """

    files = glob.glob(os.path.join(local_dir, allow_pattern))
    requests = []
    for file in files:
        requests.extend(load_jsonl_file(file_path=file))
    return requests


# Dataset name pattern can be found at https://huggingface.co/datasets/apple/mmau/tree/main.
_mmau_metric_to_file_pattern = {
    "tool_use_execution": "tool_use_execution_all_20240712.jsonl",
    "tool_use_plan": "tool_use_plan_with_reasoning_all_20240712.jsonl",
    "code_contests": "code_contests_regular_20240712.jsonl",
    "math": "math_standard_20240712.jsonl",
}


class MMAU(EvalSet):
    """Implements EvalSet for a set of MMAU evals."""

    def load_requests(
        self,
        *,
        metric_name: str,
        local_dir: Optional[str] = None,
    ) -> List[dict]:
        """Loads mmau dataset from huggingface."""

        allow_pattern = _mmau_metric_to_file_pattern[metric_name]
        local_dir = snapshot_download(
            repo_id="apple/mmau",
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_pattern,
        )
        return _load_requests_from_local_dir(local_dir=local_dir, allow_pattern=allow_pattern)

    def get_metrics(self) -> list[str]:
        """Returns a list of metric names."""
        return ["tool_use_execution", "tool_use_plan", "math", "code_contests"]

    def aggregate_metrics(self, *, metrics: list[dict]) -> dict:
        """Aggregates a list of metrics to a final dictionary of metric."""
        if len(metrics) == 0:
            raise ValueError("metrics can not be empty.")
        avg_accuracy = 0
        for metric in metrics:
            # Code contest uses pass_rate. Others use accuracy.
            if "pass_rate" in metric:
                avg_accuracy += metric["pass_rate"]
            elif "accuracy" in metric:
                avg_accuracy += metric["accuracy"]
            else:
                raise ValueError("Both pass_rate and accuracy are not found.")
        return {"score": avg_accuracy / len(metrics)}


class ToolUseExecution(MMAU):
    """Defines MMAU tool use execution eval set."""

    def get_metrics(self) -> list[str]:
        """Returns a list of metric names."""
        return ["tool_use_execution"]


class ToolUsePlan(MMAU):
    """Defines MMAU tool use plan eval set."""

    def get_metrics(self) -> list[str]:
        """Returns a list of metric names."""
        return ["tool_use_plan"]


class Math(MMAU):
    """Defines MMAU math eval set."""

    def get_metrics(self) -> list[str]:
        """Returns a list of metric names."""
        return ["math"]


class CodeContests(MMAU):
    """Defines MMAU code contests eval set."""

    def get_metrics(self) -> list[str]:
        """Returns a list of metric names."""
        return ["code_contests"]
