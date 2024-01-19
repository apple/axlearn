# Copyright © 2024 Apple Inc.

"""Tests writer utilities."""
# pylint: disable=protected-access

import os
import tempfile
import time
from datetime import datetime
from io import StringIO
from unittest import mock

from absl import app, flags
from absl.testing import absltest

from axlearn.cloud.common import writer


class TfioWriterTest(absltest.TestCase):
    """Tests TfioWriter."""

    def test_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%Y")
            resolved_output_path = datetime.now().strftime(output_path)
            w = writer.TfioWriter(output_path=output_path)
            w.write("test0\n")
            w.write("test1\n")

            # Check that file does not exist.
            self.assertTrue(not os.path.exists(resolved_output_path))

            # Force a flush and check that file exists, with the desired format.
            w._flush()
            self.assertTrue(os.path.exists(resolved_output_path))
            with open(resolved_output_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["test0", "test1"])

    def test_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%Y")
            resolved_output_path = datetime.now().strftime(output_path)

            # Start writer flush thread.
            w = writer.TfioWriter(output_path=resolved_output_path)
            with w:
                # Check that thread exists.
                self.assertIsNotNone(w._flush_thread)
                w.write("test0\n")

            # Check that file exists.
            self.assertTrue(os.path.exists(resolved_output_path))
            # Check that thread is joined.
            self.assertIsNone(w._flush_thread)

    def test_concurrent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output-%Y")
            resolved_output_path = datetime.now().strftime(output_path)

            # Start writer flush thread, which fires frequently.
            with writer.TfioWriter(output_path=resolved_output_path, flush_seconds=0.1) as w:
                # Write frequently. The writes should span ~10 flushes.
                for i in range(100):
                    w.write(f"line{i}\n")
                    time.sleep(0.01)

            # Ensure that all writes are captured and in-order.
            with open(resolved_output_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                self.assertEqual(100, len(lines))
                for i, line in enumerate(lines):
                    self.assertEqual(line, f"line{i}")

    def test_from_spec(self):
        w = writer.TfioWriter.from_spec(["output_path=test"])
        self.assertEqual(w._output_path, "test")
        self.assertEqual(w._flush_seconds, 60)

        w = writer.TfioWriter.from_spec(["output_path=test", "flush_seconds=123"])
        self.assertEqual(w._output_path, "test")
        self.assertEqual(w._flush_seconds, 123)

        # Missing output_path.
        with self.assertRaisesRegex(ValueError, "output_path"):
            writer.TfioWriter.from_spec(["flush_seconds=123"])


class CliTest(absltest.TestCase):
    """Tests writer CLI entrypoint."""

    def test_invalid_writer(self):
        fv = flags.FlagValues()
        writer._private_flags(flag_values=fv)
        fv.set_default("writer", "unknown")
        fv.mark_as_parsed()

        with self.assertRaisesRegex(app.UsageError, "Unknown writer"):
            writer.main([], flag_values=fv)

    def test_writer_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output")

            fv = flags.FlagValues()
            writer._private_flags(flag_values=fv)
            fv.set_default("writer", "tfio")
            fv.set_default("writer_spec", [f"output_path={output_path}"])
            fv.mark_as_parsed()

            with mock.patch("sys.stdin", new=StringIO("a\nb\nc")):
                writer.main([], flag_values=fv)

            with open(output_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["a", "b", "c"])
