# Copyright © 2024 Apple Inc.

"""Tests writer utilities."""
# pylint: disable=protected-access

import os
import tempfile
import time
from datetime import datetime

from absl.testing import absltest

from axlearn.cloud.common import writer


class TfioWriterTest(absltest.TestCase):
    """Tests TfioWriter."""

    def test_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%Y")
            resolved_output_path = datetime.now().strftime(output_path)
            w = writer.TfioWriter(output_path=resolved_output_path)

            with self.assertRaisesRegex(ValueError, "context"):
                w.write("test0\n")
            self.assertFalse(os.path.exists(resolved_output_path))

            # Enter context.
            with w:
                # Check that thread exists.
                self.assertIsNotNone(w._flush_thread)
                w.write("test1\n")
                w.write("test2\n")

            # Check contents.
            with open(resolved_output_path, encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["test1", "test2"])
            # Check that thread is joined.
            self.assertIsNone(w._flush_thread)

    def test_concurrent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%Y")
            resolved_output_path = datetime.now().strftime(output_path)

            # Start writer, which flushes every second.
            with writer.TfioWriter(output_path=resolved_output_path, flush_seconds=1) as w:
                for i in range(100):
                    w.write(f"line{i}\n")

            # Ensure that all writes are captured and in-order.
            with open(resolved_output_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
                self.assertEqual(100, len(lines))
                for i, line in enumerate(lines):
                    self.assertEqual(line, f"line{i}")

    def test_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%S.%f")
            with writer.TfioWriter(output_path=output_path) as w:
                w.write("line\n")
                time.sleep(0.05)
                w._maybe_open()
                w.write("line\n")
            # At least 2 output files.
            self.assertGreater(len(os.listdir(temp_dir)), 1)
