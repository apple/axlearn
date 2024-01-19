# Copyright © 2024 Apple Inc.

"""Utilities to support piping inputs from an input stream to one or more output streams.

Examples:

    # Writes "hello" to a gs path.
    echo "hello" | python3 -m axlearn.cloud.common.writer \
        --writer=tfio --writer_spec=output_path=gs://path/to/log-%Y%m%d

"""

import os
import sys
import threading
from datetime import datetime
from typing import List, Optional

from absl import app, flags, logging
from tensorflow import io as tf_io

from axlearn.cloud.common.utils import parse_kv_flags

FLAGS = flags.FLAGS


class BaseWriter:
    """Base interface for a writer."""

    def write(self, data: str):
        """Writes data in a non-blocking manner."""
        raise NotImplementedError(type(self))


class TfioWriter(BaseWriter):
    """A writer that uses tf.io."""

    TYPE = "tfio"

    # TODO(markblee): Support flushing the buffer by size.
    def __init__(self, *, output_path: str, flush_seconds: float = 60):
        """Constructs TfioWriter.

        Args:
            output_path: The output path to write to. Can contain a valid strftime format; for
                example, "log-%Y%m%d" writes to a new file daily.
                Note that writes are buffered, and the output path is only resolved on flush (hence
                a format like "%s" will only reflect the seconds at flush time).
            flush_seconds: Interval of time between flushes to tf.io, to minimize RPC calls.
                This is mostly suitable for non-bursty input streams.
        """
        self._output_path = output_path
        self._flush_seconds = flush_seconds
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._flush_thread = None
        self._stopping = None
        self._within_context = False

    @classmethod
    def from_spec(cls, spec: List[str]) -> "TfioWriter":
        """Constructs a TfioWriter from flags."""
        kwargs = parse_kv_flags(spec, delimiter="=")
        if "output_path" not in kwargs:
            raise ValueError("output_path is required.")
        if "flush_seconds" in kwargs:
            kwargs["flush_seconds"] = float(kwargs["flush_seconds"])
        return TfioWriter(**kwargs)  # pylint: disable=missing-kwoa

    def __enter__(self):
        if self._within_context:
            raise ValueError("Already in a context.")
        self._within_context = True
        self.start()
        return self

    def __exit__(self, *args) -> Optional[bool]:
        del args
        self.stop()
        self._within_context = False

    def start(self):
        if self._flush_thread is None:
            self._stopping = threading.Event()
            self._flush_thread = threading.Thread(name="flush_thread", target=self._flush_loop)
            self._flush_thread.start()

    def stop(self):
        if self._flush_thread is not None:
            self._stopping.set()
            self._flush_thread.join()
            self._flush_thread = None

    def _flush_loop(self):
        while not self._stopping.wait(timeout=self._flush_seconds):
            self._flush()
        self._flush()  # Flush again at exit.

    def _flush(self):
        output_path = datetime.now().strftime(self._output_path)
        tf_io.gfile.makedirs(os.path.dirname(output_path))

        with self._buffer_lock:
            buffer = self._buffer
            self._buffer = []

        if buffer:
            with tf_io.gfile.GFile(output_path, mode="a") as f:
                for entry in buffer:
                    f.write(entry)

    def write(self, data: str):
        """Writes data to a buffer, which is periodically flushed."""
        if not self._flush_thread:
            logging.warning("write called without an active flush thread.")
        with self._buffer_lock:
            # Write to the buffer.
            self._buffer.append(data)


_WRITERS = {TfioWriter.TYPE: TfioWriter}


def _private_flags(flag_values: flags.FlagValues = FLAGS):
    # TODO(markblee): Support specifying different input type besides stdin.
    flags.DEFINE_string("writer", None, "Writer type.", required=True, flag_values=flag_values)
    flags.DEFINE_multi_string(
        "writer_spec", [], "Writer spec as sequence of key=value.", flag_values=flag_values
    )


def main(_, *, flag_values: flags.FlagValues = FLAGS):
    if flag_values.writer not in _WRITERS:
        raise app.UsageError(
            f"Unknown writer: {flag_values.writer}. Valid options are: {_WRITERS.keys()}"
        )
    with _WRITERS[flag_values.writer].from_spec(flag_values.writer_spec) as writer:
        for line in sys.stdin:
            writer.write(line)


if __name__ == "__main__":
    _private_flags()
    app.run(main)
