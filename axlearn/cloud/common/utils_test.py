# Copyright © 2023 Apple Inc.

"""Tests general utils."""

import contextlib
import os
import pathlib
import shlex
import signal
import subprocess
import tempfile
import time
from filecmp import dircmp
from typing import Dict, Sequence, Union
from unittest import mock

from absl import app
from absl.testing import parameterized

from axlearn.cloud import ROOT_MODULE
from axlearn.cloud.common import utils


@contextlib.contextmanager
def _fake_module_root(module_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Make a top level module directory.
        temp_module = os.path.join(temp_dir, module_name)
        os.makedirs(temp_module)
        yield temp_module


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    def test_get_package_root(self):
        with _fake_module_root("test_module") as temp_module:
            with self.assertRaisesRegex(ValueError, "Not running within"):
                utils.get_package_root(temp_module)

        self.assertEqual(str(ROOT_MODULE), utils.get_package_root())

    def test_running_from_source(self):
        # Patch package root.
        with _fake_module_root("test_module") as temp_module:
            with mock.patch(f"{utils.__name__}.get_package_root", return_value=temp_module):
                self.assertFalse(utils.running_from_source())
                os.makedirs(os.path.join(os.path.dirname(temp_module), ".git"))
                self.assertTrue(utils.running_from_source())

    @parameterized.parameters(
        dict(
            kv_flags=["key1:value1", "key2:value2", "key1:value3"],
            expected=dict(
                key1="value3",
                key2="value2",
            ),
        ),
        dict(
            kv_flags=["key1=value1", "key2=value2", "key1=value3"],
            expected=dict(
                key1="value3",
                key2="value2",
            ),
            delimiter="=",
        ),
        dict(kv_flags=[], expected={}),
        dict(kv_flags=["malformatted"], expected=ValueError()),
        dict(kv_flags=["a:b"], delimiter="=", expected=ValueError()),
        dict(kv_flags=["a=b=c"], delimiter="=", expected=dict(a="b=c")),
    )
    def test_parse_kv_flags(
        self, kv_flags: Sequence[str], expected: Union[Dict, Exception], delimiter: str = ":"
    ):
        if issubclass(type(expected), ValueError):
            with self.assertRaises(type(expected)):
                utils.parse_kv_flags(kv_flags, delimiter=delimiter)
        else:
            self.assertEqual(expected, utils.parse_kv_flags(kv_flags, delimiter=delimiter))

    def test_format_table(self):
        headings = ["COLUMN1", "LONG_COLUMN2", "COL3"]
        rows = [
            ["long_value1", 123, {"a": "dict"}],
            ["short", 12345678, {}],
        ]
        expected = (
            "\n"
            "COLUMN1          LONG_COLUMN2      COL3               \n"
            "long_value1      123               {'a': 'dict'}      \n"
            "short            12345678          {}                 "
            "\n"
        )
        self.assertEqual(expected, utils.format_table(headings=headings, rows=rows))

    @parameterized.parameters(
        dict(argv=["cli", "activate"], expected="activate"),
        dict(argv=["cli", "cleanup"], expected="cleanup"),
        dict(argv=["cli", "list", "something"], expected="list"),
        dict(argv=["cli", "--flag1", "activate"], expected="activate"),
        dict(argv=["cli"], default="list", expected="list"),
        dict(argv=["cli", "invalid"], default="list", expected="list"),
        dict(argv=["cli", "invalid", "activate"], default="list", expected="list"),
        # Test failure case.
        dict(argv=[], expected=app.UsageError("")),
        dict(argv=["cli", "invalid"], expected=app.UsageError("")),
    )
    def test_parse_action(self, argv, expected, default=None):
        options = ["activate", "list", "cleanup"]
        if isinstance(expected, BaseException):
            with self.assertRaises(type(expected)):
                utils.parse_action(argv, options=options, default=default)
        else:
            self.assertEqual(expected, utils.parse_action(argv, options=options, default=default))

    def test_send_signal(self):
        """Tests send_signal by starting a subprocess which has child subprocesses.

        Unlike p.kill(), send_signal(p, sig=signal.SIGKILL) should recursively kill the children,
        which does not leave orphan processes running. This test will fail by replacing
        send_signal(p, sig=signal.SIGKILL) with p.kill().
        """
        test_script = os.path.join(os.path.dirname(__file__), "testdata/counter.py")
        with tempfile.NamedTemporaryFile("r+") as f:

            def _read_count():
                f.seek(0, 0)
                return int(f.read())

            # pylint: disable-next=consider-using-with
            p = subprocess.Popen(
                shlex.split(f"python3 {test_script} {f.name} parent"), start_new_session=True
            )
            time.sleep(1)
            # Check that the count has incremented.
            self.assertGreater(_read_count(), 0)
            # Kill the subprocess.
            utils.send_signal(p, sig=signal.SIGKILL)
            # Get the count again, after kill has finished.
            count = _read_count()
            self.assertGreater(count, 0)
            # Wait for some ticks.
            time.sleep(1)
            # Ensure that the count is still the same.
            self.assertEqual(_read_count(), count)

    def test_copy_blobs(self):
        with tempfile.TemporaryDirectory() as read_dir:
            read_dir_path = pathlib.Path(read_dir)
            file_a = read_dir_path / "file.txt"
            file_a.touch()
            sub_directory = read_dir_path / "subdir"
            sub_directory.mkdir()
            file_b = sub_directory / "subdirfile.txt"
            file_b.touch()
            with tempfile.TemporaryDirectory() as write_dir:
                utils.copy_blobs("file://" + read_dir, to_prefix=write_dir)
                dcmp = dircmp(read_dir, write_dir)
                mismatched_paths = dcmp.left_only + dcmp.right_only + dcmp.diff_files
                self.assertEqual(len(mismatched_paths), 0)
