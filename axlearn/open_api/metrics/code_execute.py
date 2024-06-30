# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# openai/human-eval:
# Copyright 2021 The OpenAI Authors. All Rights Reserved.
# Licensed under the MIT License.
#
# huggingface/evaluate:
# Copyright 2023 The Huggingface Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

# pylint: disable=reimported,import-outside-toplevel,redefined-outer-name
"""Executing and verifying python code locally with test inputs.

This file is adopted from
https://github.com/openai/human-eval/blob/312c5e5532f0e0470bf47f77a6243e02a61da530/human_eval/execution.py
with a slight modification on returning error message from execution.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
from typing import Any, Dict, List, Optional, Union


# TODO(gyin): Consider refactoring this with context manager.
def check_correctness(
    check_program: str, *, inputs: List[Any], timeout: float, task_id: str, completion_id: str
) -> Dict[str, Any]:
    """Evaluates the functional correctness of a completion by running a set of test
        inputs in the problem.

    Args:
        check_program: the python code to be checked.
        inputs: a list of test case inputs to the code.
        timeout: a timeout limit for code execution.
        task_id: an indicator of the task.
        completion_id: an indicator of the generated code sample.

    Returns:
        A dictionary containing the task_id, completion_id, and execution results.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(
        target=_unsafe_execute, args=(check_program, inputs, result, timeout)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    # Customized change compared with original humaneval implementation:
    # Add output and error stream.
    if not result:
        result.append({"error": "timeout", "result_list": []})

    if len(result) == 0:
        result.append({"error": "invalid output", "result_list": []})
    return {"task_id": task_id, "completion_id": completion_id, "result": result[0]}


def _unsafe_execute(check_program: str, inputs: List[Any], result: List[Any], timeout: float):
    """Evaluates the functional correctness of a completion by running a set of test
     inputs in the problem.

    Args:
        check_program: the python code to be checked.
        inputs: a list of test case inputs to the code.
        result: a process manager list to store all execution results.
        timeout: a timeout limit for code execution.
    """
    with _create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        _reliability_guard()

        # Run program.
        # Customized change compared with original humaneval implementation:
        # Add output and error stream.
        exec_results = {"error": None, "result_list": []}
        try:
            exec_globals = {}
            for inp in inputs:
                try:
                    input_stream = io.BytesIO(inp.encode())
                    input_stream.seek(0)
                    with _swallow_io(input_stream=input_stream) as (stdout_stream, stderr_stream):
                        with _time_limit(timeout):
                            exec(check_program, exec_globals)  # pylint: disable=exec-used
                    exec_results["result_list"].append(
                        {
                            "passed": True,
                            "output": stdout_stream.getvalue(),
                            "error": stderr_stream.getvalue(),
                        }
                    )
                except TimeoutException:
                    exec_results["result_list"].append(
                        {"passed": False, "output": "", "error": "timeout"}
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    exec_results["result_list"].append(
                        {"passed": False, "output": "", "error": f"failed: {e}"}
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            exec_results["error"] = f"failed: {e}"

        result.append(exec_results)
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def _time_limit(seconds: float):
    def signal_handler(signum: Any, frame: Any):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io(
    input_stream: Optional[Union[io.BytesIO, io.StringIO, io.BufferedIOBase, io.TextIOBase]] = None,
    binary: bool = False,
):
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdin = input_stream if binary else io.TextIOWrapper(input_stream)
    sys.stdout = io.BytesIO() if binary else io.StringIO()
    sys.stderr = io.BytesIO() if binary else io.StringIO()

    try:
        yield sys.stdout, sys.stderr
    finally:
        sys.stdin = original_stdin
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@contextlib.contextmanager
def _create_tempdir():
    """Creates a temporary directory."""
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    """Defines a timeout exception."""

    pass


# pylint: disable=unused-argument
class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


# pylint: enable=unused-argument


# pylint: disable-next=invalid-name, protected-access
class redirect_stdin(contextlib._RedirectStream):  # pytype: disable=module-attr
    _stream = "stdin"


@contextlib.contextmanager
def _chdir(root: str):
    """Changes the current working directory."""
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def _reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # pytype: disable=invalid-directive

    __builtins__["help"] = None  # pytype: disable=unsupported-operands

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
