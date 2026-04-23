# Copyright © 2026 Apple Inc.

"""Tests for hf_module."""

import tempfile
from pathlib import Path

from absl.testing import absltest

from axlearn.huggingface.hf_module import _recursive_copy_dir


class RecursiveCopyDirTest(absltest.TestCase):
    def test_recursive_copy_dir(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            Path(src, "file1").write_text("foo", encoding="utf-8")
            Path(src, "dir1").mkdir()
            Path(src, "dir1", "file2").write_text("bar", encoding="utf-8")

            _recursive_copy_dir(src, dst)
            self.assertEqual(Path(dst, "file1").read_text(encoding="utf-8"), "foo")
            self.assertEqual(Path(dst, "dir1", "file2").read_text(encoding="utf-8"), "bar")


if __name__ == "__main__":
    absltest.main()
