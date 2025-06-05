# Copyright © 2024 Apple Inc.

"""Utilities to support piping inputs from an input stream to one or more output streams.

Examples:
    ```
    # Writes "hello" to a gs path.
    echo "hello" | python3 -m axlearn.cloud.common.writer --output_path=gs://path/to/log-%Y%m%d
    ```
"""

import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from absl import app, flags

from axlearn.common.file_system import makedirs
from axlearn.common.file_system import open as fs_open

FLAGS = flags.FLAGS
_Done = object()


class BaseWriter:
    """Base interface for a writer."""

    def write(self, data: str):
        """Writes data in a non-blocking manner."""
        raise NotImplementedError(type(self))


class TfioWriter(BaseWriter):
    """A writer that uses tf.io.

    It is typically used as a context manager, e.g:
        ```
        with TfioWriter(output_path="...") as w:
            w.write("data")
        ```
    """

    def __init__(self, *, output_path: str, flush_seconds: float = 1):
        """Constructs TfioWriter.

        Args:
            output_path: The output path to write to. Can contain a valid strftime format; for
                example, "log-%Y%m%d" writes to a new file daily.
            flush_seconds: Minimum interval of time between flushes to tf.io, to minimize RPC calls.
        """
        self._output_path = output_path
        self._flush_seconds = flush_seconds
        self._flush_thread = None
        self._within_context = False
        self._queue = None
        self._file = None
        self._last_flush = None

    def __enter__(self):
        if self._within_context:
            raise ValueError("Already in a context.")
        self._within_context = True
        self._start()
        return self

    def __exit__(self, *args) -> Optional[bool]:
        del args
        self._stop()
        self._within_context = False

    def _start(self):
        """Starts the flush thread."""
        if self._flush_thread is None:
            self._queue = queue.Queue()
            self._flush_thread = threading.Thread(name="flush_thread", target=self._flush_loop)
            self._flush_thread.start()

    def _stop(self):
        """Stops the flush thread. Writes made after this will not be flushed."""
        if self._flush_thread is not None:
            self._queue.put(_Done)
            self._flush_thread.join()
            self._flush_thread = None
            self._queue = None

    def _flush_loop(self):
        """Writes and flushes data to output file on-demand."""
        while True:
            try:
                # If we don't receive writes, allow a timeout so that we still occasionally flush.
                data = self._queue.get(block=True, timeout=self._flush_seconds)
            except queue.Empty:
                data = None
            if data == _Done:
                break
            self._maybe_open()
            if data is not None:
                self._file.write(data)
            now = time.time()
            if self._last_flush is None or (now - self._last_flush >= self._flush_seconds):
                self._last_flush = now
                self._file.flush()
        self._maybe_close()

    def _maybe_open(self):
        """Opens the output file for writing, if:
        - The file has not been opened.
        - The file name has changed (in which case, we close the old file).
        """
        output_path = datetime.now().strftime(self._output_path)
        if self._file is None or self._file.name != output_path:
            self._maybe_close()
            makedirs(os.path.dirname(output_path))
            self._file = fs_open(output_path, mode="a")

    def _maybe_close(self):
        """Closes the output file if it exists."""
        if self._file:
            self._file.close()
            self._file = None

    def write(self, data: str):
        """Writes data to an internal buffer, to be flushed."""
        if not self._flush_thread:
            raise ValueError("write called outside of context.")
        # Note that this will never raise queue.Full, since the queue size is unbounded.
        self._queue.put_nowait(data)


def _private_flags(flag_values: flags.FlagValues = FLAGS):
    flags.DEFINE_string("output_path", None, "Output path.", required=True, flag_values=flag_values)


def main(_, *, flag_values: flags.FlagValues = FLAGS):
    with TfioWriter(output_path=flag_values.output_path) as writer:
        for line in sys.stdin:
            writer.write(line)


if __name__ == "__main__":
    _private_flags()
    app.run(main)
