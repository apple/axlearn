# Copyright © 2026 Apple Inc.

"""Tests for hf_module."""

import tempfile
from pathlib import Path

from axlearn.huggingface.hf_module import _recursive_copy_dir


def test_recursive_copy_dir():
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        # File directly under src.
        Path(src, "file1").write_text("foo", encoding="utf-8")
        # Subdirectory with a file under it.
        Path(src, "dir1").mkdir()
        Path(src, "dir1", "file2").write_text("bar", encoding="utf-8")

        _recursive_copy_dir(src, dst)
        assert Path(dst, "file1").read_text(encoding="utf-8") == "foo"
        assert Path(dst, "dir1", "file2").read_text(encoding="utf-8") == "bar"
