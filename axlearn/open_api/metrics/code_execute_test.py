# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit test for code_execute.py.
"""
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# pylint: disable=wrong-import-position
from axlearn.open_api.metrics.code_execute import (
    TimeoutException,
    _unsafe_execute,
    check_correctness,
)

# pylint: enable=wrong-import-position


class TestCheckCorrectness(unittest.TestCase):
    """Unit tests for check_correctness."""

    @patch("multiprocessing.Process")
    @patch("multiprocessing.Manager")
    def test_check_correctness_success(self, mock_manager, mock_process):
        check_program = 'print("Hello, World!")'
        inputs = ["input1", "input2"]
        timeout = 2.0
        task_id = "task1"
        completion_id = "comp1"

        mock_result_list = [{"passed": True, "output": b"Hello, World!\n", "error": b""}]
        mock_manager.return_value.list.return_value = mock_result_list

        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        result = check_correctness(
            check_program=check_program,
            inputs=inputs,
            timeout=timeout,
            task_id=task_id,
            completion_id=completion_id,
        )

        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called_once_with(timeout=timeout + 1)
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["completion_id"], completion_id)
        self.assertEqual(result["result"], mock_result_list[0])

    @patch("multiprocessing.Process")
    @patch("multiprocessing.Manager")
    def test_check_correctness_timeout(self, mock_manager, mock_process):
        check_program = "while True: pass"
        inputs = ["input1"]
        timeout = 1.0
        task_id = "task2"
        completion_id = "comp2"

        mock_result_list = []
        mock_manager.return_value.list.return_value = mock_result_list

        mock_proc = MagicMock()
        mock_process.return_value = mock_proc
        mock_proc.is_alive.return_value = True

        result = check_correctness(
            check_program=check_program,
            inputs=inputs,
            timeout=timeout,
            task_id=task_id,
            completion_id=completion_id,
        )

        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called_once_with(timeout=timeout + 1)
        mock_proc.kill.assert_called_once()
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["completion_id"], completion_id)
        self.assertEqual(result["result"]["error"], "timeout")
        self.assertEqual(result["result"]["result_list"], [])

    @patch("multiprocessing.Process")
    @patch("multiprocessing.Manager")
    def test_check_correctness_exception(self, mock_manager, mock_process):
        check_program = 'raise ValueError("Test exception")'
        inputs = ["input1"]
        timeout = 2.0
        task_id = "task3"
        completion_id = "comp3"

        mock_result_list = [{"error": "failed: Test exception", "result_list": []}]
        mock_manager.return_value.list.return_value = mock_result_list

        mock_proc = MagicMock()
        mock_process.return_value = mock_proc

        result = check_correctness(
            check_program=check_program,
            inputs=inputs,
            timeout=timeout,
            task_id=task_id,
            completion_id=completion_id,
        )

        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called_once_with(timeout=timeout + 1)
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["completion_id"], completion_id)
        self.assertEqual(result["result"], mock_result_list[0])


class TestUnsafeExecute(unittest.TestCase):
    """Unit tests for _unsafe_execute."""

    @patch("axlearn.open_api.metrics.code_execute._create_tempdir")
    @patch("axlearn.open_api.metrics.code_execute._reliability_guard")
    @patch("axlearn.open_api.metrics.code_execute._swallow_io")
    @patch("axlearn.open_api.metrics.code_execute._time_limit")
    def test_successful_execution(
        self, mock_time_limit, mock_swallow_io, mock_reliability_guard, mock_create_tempdir
    ):
        check_program = """
def add(a, b):
    return a + b

result = add(2, 3)
"""
        inputs = ["test_input"]
        result = []
        timeout = 5.0

        mock_create_tempdir.return_value = MagicMock()
        mock_reliability_guard.return_value = None
        mock_time_limit.return_value.__enter__ = lambda *args: None
        mock_time_limit.return_value.__exit__ = lambda *args: None
        mock_stdout_stream = BytesIO()
        mock_stdout_stream.getvalue = lambda: b"test_input\n"
        mock_stderr_stream = BytesIO()
        mock_stderr_stream.getvalue = lambda: b""
        mock_swallow_io.return_value.__enter__ = lambda *args: (
            mock_stdout_stream,
            mock_stderr_stream,
        )
        mock_swallow_io.return_value.__exit__ = lambda *args: None

        _unsafe_execute(check_program, inputs, result, timeout)

        expected_result = {
            "error": None,
            "result_list": [{"passed": True, "output": b"test_input\n", "error": b""}],
        }

        self.assertEqual(result, [expected_result])

    @patch("axlearn.open_api.metrics.code_execute._create_tempdir")
    @patch("axlearn.open_api.metrics.code_execute._reliability_guard")
    @patch("axlearn.open_api.metrics.code_execute._swallow_io")
    @patch("axlearn.open_api.metrics.code_execute._time_limit")
    def test_execution_error(
        self, mock_time_limit, mock_swallow_io, mock_reliability_guard, mock_create_tempdir
    ):
        check_program = """
raise ValueError('test error')
"""
        inputs = [""]
        result = []
        timeout = 5.0

        mock_create_tempdir.return_value = MagicMock()
        mock_reliability_guard.return_value = None
        mock_time_limit.return_value.__enter__ = lambda *args: None
        mock_time_limit.return_value.__exit__ = lambda *args: None
        mock_stdout_stream = MagicMock()
        mock_stdout_stream.getvalue.return_value = b""
        mock_stderr_stream = MagicMock()
        mock_stderr_stream.getvalue.return_value = b""
        mock_swallow_io.return_value.__enter__ = lambda *args: (
            mock_stdout_stream,
            mock_stderr_stream,
        )
        mock_swallow_io.return_value.__exit__ = lambda *args: None

        _unsafe_execute(check_program, inputs, result, timeout)

        expected_result = {
            "error": None,
            "result_list": [
                {
                    "passed": False,
                    "output": "",
                    "error": "failed: test error",
                }
            ],
        }

        self.assertEqual(result, [expected_result])

    @patch("axlearn.open_api.metrics.code_execute._create_tempdir")
    @patch("axlearn.open_api.metrics.code_execute._reliability_guard")
    @patch("axlearn.open_api.metrics.code_execute._swallow_io")
    @patch("axlearn.open_api.metrics.code_execute._time_limit")
    def test_execution_timeout(
        self, mock_time_limit, mock_swallow_io, mock_reliability_guard, mock_create_tempdir
    ):
        check_program = """
while True:
    pass
"""
        inputs = [""]
        result = []
        timeout = 0.1

        mock_create_tempdir.return_value = MagicMock()
        mock_reliability_guard.return_value = None

        class MockTimeLimit:
            def __enter__(self):
                raise TimeoutException()

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        mock_time_limit.return_value = MockTimeLimit()
        mock_stdout_stream = MagicMock()
        mock_stdout_stream.getvalue.return_value = b""
        mock_stderr_stream = MagicMock()
        mock_stderr_stream.getvalue.return_value = b""
        mock_swallow_io.return_value.__enter__ = lambda *args: (
            mock_stdout_stream,
            mock_stderr_stream,
        )
        mock_swallow_io.return_value.__exit__ = lambda *args: None

        _unsafe_execute(check_program, inputs, result, timeout)

        expected_result = {
            "error": None,
            "result_list": [{"passed": False, "output": "", "error": "timeout"}],
        }

        self.assertEqual(result, [expected_result])
