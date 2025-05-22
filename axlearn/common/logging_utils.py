# Copyright © 2025 Apple Inc.

"""Logging utility classes"""

import logging
import time


class LoggingContext:
  """A context manager for monitoring the execution time of operations and
  logging warnings based on defined thresholds.

  This context manager can be used to:
  - Log a warning if an operation's total execution time exceeds a specified
  threshold.

  Example usage:
  >>> with LoggingContext(msg="File processing", threshold=10.0)
  ...     # Perform some operation that might take time
  ...     time.sleep(7)
  ... # If the operation takes more than 10 seconds,
  ... # a threshold warning will be logged upon exit.
  """

  def __init__(self, msg: str, threshold: float = 0):
      """Initializes the LoggingContext with parameters for monitoring.

      Args:
          msg (str): A descriptive message to be included in the log warnings,
            helping to identify the operation being monitored.
          threshold (float): The maximum allowed duration in seconds for the
            operation. If the operation's total execution time exceeds this value,
            a warning will be logged upon exit. Set to 0 to disable threshold
            warnings.
      """
      self._threshold = threshold
      self._msg = msg
      self._start_time: float | None = None

  def __enter__(self):
      """Enters the runtime context related to this object."""
      self._start_time = time.time()

  def __exit__(self, exec_type, exc_val, exc_tb):
      """Exits the runtime context related to this object."""
      elapse = time.time() - self._start_time
      if elapse > self._threshold:
        logging.warning(
            "**OPERATION DURATION WARNING:** The operation '%s'"
            " completed in %.2f seconds, which exceeded the defined"
            " threshold of %.2f seconds.",
            self._msg,
            elapse,
            self._threshold,
        )
