# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit test for code_contests.py.

TODO(gyin): Add more unit tests for code contests core functions.
"""

import unittest
from unittest.mock import patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# isort: off
# pylint: disable=wrong-import-position
from axlearn.open_api.metrics import code_contests

from axlearn.open_api.metrics.code_contests import (
    CodeContestStatus,
    _exec_and_check,
    _format_gen_code,
    _parse_answer,
)

# pylint: enable=wrong-import-position
# isort: on


class TestFormatGenCode(unittest.TestCase):
    """Unit tests for _format_gen_code."""

    def test_format_gen_code_with_python_tags(self):
        gen_code = "```python\nprint('Hello World')\n```"
        expected_output = "print('Hello World')"
        self.assertEqual(_format_gen_code(gen_code), expected_output)

    def test_format_gen_code_with_explanation(self):
        gen_code = "```python\nprint('Hello World')\n```\nThis is a test."
        expected_output = "print('Hello World')"
        self.assertEqual(_format_gen_code(gen_code), expected_output)

    def test_format_gen_code_with_if_main(self):
        gen_code = "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
        expected_output = "def main():\n    print('Hello World')\n\nmain()"
        self.assertEqual(_format_gen_code(gen_code), expected_output)

    def test_format_gen_code_with_all_cases(self):
        gen_code = (
            "```python\ndef main():\n    print('Hello World')\n\n"
            + "if __name__ == '__main__':\n    main()\n```\nThis is a test."
        )
        expected_output = "def main():\n    print('Hello World')\n\nmain()"
        self.assertEqual(_format_gen_code(gen_code), expected_output)

    def test_format_gen_code_without_special_cases(self):
        gen_code = "print('Hello World')"
        expected_output = "print('Hello World')"
        self.assertEqual(_format_gen_code(gen_code), expected_output)


class TestParseAnswer(unittest.TestCase):
    """Unit tests for _parse_answer."""

    def test_with_hash_answer(self):
        self.assertEqual(
            _parse_answer("Some text #ANSWER#: This is the answer."), " This is the answer."
        )

    def test_with_answer(self):
        self.assertEqual(
            _parse_answer("Some text ANSWER: This is the answer."), ": This is the answer."
        )

    def test_with_answer_newline(self):
        self.assertEqual(
            _parse_answer("Some text ANSWER\nThis is the answer."), "This is the answer."
        )

    def test_with_answer_no_content(self):
        self.assertEqual(_parse_answer("Some text ANSWER"), "")

    def test_without_answer_marker(self):
        self.assertEqual(_parse_answer("Some text without the answer marker."), "")

    def test_with_empty_string(self):
        self.assertEqual(_parse_answer(""), "")

    def test_with_only_hash_answer(self):
        self.assertEqual(_parse_answer("#ANSWER#: This is the answer."), " This is the answer.")

    def test_with_only_answer(self):
        self.assertEqual(_parse_answer("ANSWER\nThis is the answer."), "This is the answer.")


# pylint: disable=unused-argument
class TestExecAndCheck(unittest.TestCase):
    """Unit tests for _exec_and_check."""

    @patch(f"{code_contests.__name__}.check_correctness")
    def test_empty_gen_code(self, mock_check_correctness):
        result = _exec_and_check(
            gen_code="", problem={}, problem_id="prob_1", candidate_id="cand_1"
        )
        self.assertEqual(
            result, {"passed": False, "score": 0.0, "errors": {"status": CodeContestStatus.EMPTY}}
        )

    @patch(f"{code_contests.__name__}.check_correctness")
    def test_no_tests(self, mock_check_correctness):
        problem = {}
        result = _exec_and_check(
            gen_code="code", problem=problem, problem_id="prob_1", candidate_id="cand_1"
        )
        self.assertEqual(result, {"passed": False, "score": 0.0, "errors": []})

    @patch(f"{code_contests.__name__}.check_correctness")
    def test_all_tests_passed(self, mock_check_correctness):
        problem = {"public_tests": {"input": [1, 2], "output": ["3", "4"]}}
        mock_check_correctness.return_value = {
            "result": {
                "error": None,
                "result_list": [{"passed": True, "output": "3"}, {"passed": True, "output": "4"}],
            }
        }
        result = _exec_and_check(
            gen_code="code", problem=problem, problem_id="prob_1", candidate_id="cand_1"
        )
        self.assertEqual(result, {"passed": True, "score": 1.0, "errors": []})

    @patch(f"{code_contests.__name__}.check_correctness")
    def test_some_tests_failed(self, mock_check_correctness):
        problem = {"public_tests": {"input": [1, 2], "output": ["3", "4"]}}
        mock_check_correctness.return_value = {
            "result": {
                "error": None,
                "result_list": [
                    {"passed": True, "output": "3"},
                    {"passed": False, "error": "Some error"},
                ],
            }
        }
        result = _exec_and_check(
            gen_code="code", problem=problem, problem_id="prob_1", candidate_id="cand_1"
        )
        self.assertEqual(result, {"passed": False, "score": 0.5, "errors": []})

    @patch(f"{code_contests.__name__}.check_correctness")
    def test_save_errors(self, mock_check_correctness):
        problem = {"public_tests": {"input": [1, 2], "output": ["3", "4"]}}
        mock_check_correctness.return_value = {
            "result": {
                "error": None,
                "result_list": [
                    {"passed": True, "output": "3"},
                    {"passed": False, "error": "Some error"},
                ],
            }
        }
        result = _exec_and_check(
            gen_code="code",
            problem=problem,
            problem_id="prob_1",
            candidate_id="cand_1",
            save_errors=True,
        )
        self.assertEqual(
            result["errors"],
            [{"status": CodeContestStatus.ERROR, "test_input": 2, "error_message": "Some error"}],
        )


# pylint: enable=unused-argument
