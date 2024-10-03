# Copyright © 2024 Apple Inc.

"""Unit tests for mmau.py."""
import unittest
from unittest.mock import patch

from axlearn.open_api.mock_utils import mock_huggingface_hub_package, mock_openai_package

mock_huggingface_hub_package()
mock_openai_package()

# pylint: disable=wrong-import-position
from axlearn.open_api.eval_set import mmau
from axlearn.open_api.eval_set.mmau import _load_requests_from_local_dir

# pylint: enable=wrong-import-position


class TestMMAUEvalSet(unittest.TestCase):
    """Unit tests for mmau eval set."""

    @patch(f"{mmau.__name__}.load_jsonl_file")
    @patch("glob.glob")
    def test_load_requests_from_local_dir(self, mock_glob, mock_load_jsonl_file):
        mock_load_jsonl_file.return_value = [{"example_key": "example_value"}]
        mock_glob.return_value = ["/fake/dir/file1.jsonl", "/fake/dir/file2.jsonl"]
        result = _load_requests_from_local_dir(local_dir="/fake/dir", allow_pattern="*.jsonl")

        # Assertions.
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result, [{"example_key": "example_value"}, {"example_key": "example_value"}]
        )
        mock_load_jsonl_file.assert_called_with(file_path="/fake/dir/file2.jsonl")

    @patch(f"{mmau.__name__}.snapshot_download")
    @patch(f"{mmau.__name__}._load_requests_from_local_dir")
    def test_load_requests(self, mock_load_requests_from_local_dir, mock_snapshot_download):
        mmau_instance = mmau.MMAU()
        mock_snapshot_download.return_value = "/mock/local/dir"
        mock_load_requests_from_local_dir.return_value = [{"data": "sample"}]

        requests = mmau_instance.load_requests(
            metric_name="tool_use_execution", local_dir="/mock/dir"
        )

        mock_snapshot_download.assert_called_once_with(
            repo_id="apple/mmau",
            repo_type="dataset",
            local_dir="/mock/dir",
            allow_patterns="tool_use_execution_all_20240712.jsonl",
        )
        mock_load_requests_from_local_dir.assert_called_once_with(
            local_dir="/mock/local/dir", allow_pattern="tool_use_execution_all_20240712.jsonl"
        )
        self.assertEqual(requests, [{"data": "sample"}])

    def test_get_metrics(self):
        mmau_instance = mmau.MMAU()

        metrics = mmau_instance.get_metrics()

        self.assertEqual(metrics, ["tool_use_execution", "tool_use_plan", "math", "code_contests"])

    def test_aggregate_metrics(self):
        mmau_instance = mmau.MMAU()
        metrics = [{"accuracy": 0.8}, {"accuracy": 0.6}, {"pass_rate": 0.9}]

        aggregated_metrics = mmau_instance.aggregate_metrics(metrics=metrics)

        self.assertEqual(aggregated_metrics, {"score": (0.8 + 0.6 + 0.9) / 3})

    def test_aggregate_metrics_empty(self):
        mmau_instance = mmau.MMAU()

        with self.assertRaises(ValueError) as context:
            mmau_instance.aggregate_metrics(metrics=[])
        self.assertEqual(str(context.exception), "metrics can not be empty.")

    def test_aggregate_metrics_invalid_metric(self):
        mmau_instance = mmau.MMAU()
        metrics = [{"invalid_metric": 0.5}]

        with self.assertRaises(ValueError) as context:
            mmau_instance.aggregate_metrics(metrics=metrics)
        self.assertEqual(str(context.exception), "Both pass_rate and accuracy are not found.")
