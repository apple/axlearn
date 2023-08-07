# Copyright © 2023 Apple Inc.

"""Tests general GCP utils."""

import contextlib
import os
import tempfile
from typing import Dict, Sequence, Union
from unittest import mock

from absl.testing import parameterized

from axlearn.gcp import GCP_ROOT, utils


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

        self.assertEqual(os.path.dirname(GCP_ROOT), utils.get_package_root())

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
        dict(kv_flags=[], expected={}),
        dict(kv_flags=["malformatted"], expected=ValueError()),
    )
    def test_parse_kv_flags(self, kv_flags: Sequence[str], expected: Union[Dict, Exception]):
        if issubclass(type(expected), ValueError):
            with self.assertRaises(type(expected)):
                utils.parse_kv_flags(kv_flags)
        else:
            self.assertEqual(expected, utils.parse_kv_flags(kv_flags))

    @parameterized.parameters(
        dict(name="test--01-exp123", expected=True),
        dict(name="123-test", expected=False),  # Must begin with letter.
        dict(name="test+123", expected=False),  # No other special characters allowed.
    )
    def test_is_valid_resource_name(self, name: str, expected: bool):
        self.assertEqual(expected, utils.is_valid_resource_name(name))

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
