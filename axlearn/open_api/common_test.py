# Copyright © 2024 Apple Inc.

"""Unit tests for common.py."""
import json
import unittest
from datetime import timedelta
from unittest.mock import mock_open, patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# pylint: disable=wrong-import-position
from axlearn.open_api.common import (
    check_vllm_readiness,
    flatten_responses,
    load_requests,
    parse_decode_parameters,
    repeat_requests,
    write_responses,
)

# pylint: enable=wrong-import-position


class TestUtilities(unittest.TestCase):
    """Unit tests for utilities function."""

    @patch("requests.get")
    def test_check_vllm_readiness(self, mock_get):
        mock_get.return_value.status_code = 200
        # No exception should be raised
        check_vllm_readiness(timedelta(seconds=10), "http://0.0.0.0:8000/v1")

    @patch("json.loads")
    def test_parse_decode_parameters(self, mock_json_loads):
        mock_json_loads.return_value = {"max_tokens": 1024}
        decode_parameters, _ = parse_decode_parameters('{"max_tokens": 1024}')
        self.assertEqual(decode_parameters["max_tokens"], 1024)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_responses(self, mock_file, mock_exists):
        mock_exists.return_value = True
        write_responses([{"response": "data"}], file_path="path/to/file")
        mock_file.assert_called_with("path/to/file", "w", encoding="utf-8")

    def test_expand_requests(self):
        # Test expanding OpenAI style requests.
        requests = [
            {"id": "001", "prompt": "Hello, world!", "temperature": 0.5},
            {"id": "002", "prompt": "Goodbye, world!", "temperature": 0.5},
        ]
        expanded = repeat_requests(requests, 3)
        self.assertEqual(len(expanded), 6)
        self.assertEqual(expanded[0]["id"], "001:::0")
        self.assertEqual(expanded[1]["id"], "001:::1")
        self.assertEqual(expanded[4]["id"], "002:::1")

    def test_expand_requests_with_deliverable_id(self):
        # Test with 'deliverable_id' instead of 'id'.
        requests = [{"deliverable_id": "101", "prompt": "Test prompt", "temperature": 1.0}]
        expanded = repeat_requests(requests, 2)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0]["deliverable_id"], "101:::0")
        self.assertEqual(expanded[1]["deliverable_id"], "101:::1")

    def test_squeeze_responses(self):
        # Test squeezing responses to form a single entry per original ID.
        responses = [
            {"id": "001:::0", "response": "Hello!"},
            {"id": "001:::1", "response": "Hi!"},
            {"id": "002:::0", "response": "Goodbye!"},
        ]
        squeezed = flatten_responses(responses)
        self.assertEqual(len(squeezed), 2)
        self.assertIn("Hi!", squeezed[0]["n_responses"])
        self.assertIn("Hello!", squeezed[0]["n_responses"])
        self.assertEqual(squeezed[0]["id"], "001")
        self.assertEqual(squeezed[1]["id"], "002")


class TestLoadRequests(unittest.TestCase):
    """Unit test for load_requests"""

    def test_empty_file(self):
        """Test loading from an empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = load_requests("dummy_path", max_instances=10)
            self.assertEqual(result, [])

    def test_single_line(self):
        """Test loading a single line of JSON."""
        json_line = '{"name": "example"}'
        with patch("builtins.open", mock_open(read_data=json_line)):
            result = load_requests("dummy_path", max_instances=10)
            self.assertEqual(result, [{"name": "example"}])

    def test_multiple_lines(self):
        """Test loading multiple lines of JSON."""
        json_data = '{"name": "example1"}\n{"name": "example2"}'
        with patch("builtins.open", mock_open(read_data=json_data)):
            result = load_requests("dummy_path", max_instances=10)
            self.assertEqual(result, [{"name": "example1"}, {"name": "example2"}])

    def test_max_instances(self):
        """Test the max_instances limit."""
        json_data = '{"name": "example1"}\n{"name": "example2"}\n{"name": "example3"}'
        with patch("builtins.open", mock_open(read_data=json_data)):
            result = load_requests("dummy_path", max_instances=2)
            self.assertEqual(result, [{"name": "example1"}, {"name": "example2"}])

    def test_file_not_found(self):
        """Test file not found error handling."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with self.assertRaises(FileNotFoundError):
                load_requests("nonexistent_path", max_instances=10)

    def test_invalid_json(self):
        """Test behavior with invalid JSON."""
        json_data = '{"name": "valid"}\ninvalid_json'
        with patch("builtins.open", mock_open(read_data=json_data)):
            with self.assertRaises(json.JSONDecodeError):
                load_requests("dummy_path", max_instances=10)
