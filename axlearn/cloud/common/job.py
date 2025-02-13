# Copyright © 2023 Apple Inc.

"""Utilities for executing on remote compute."""

import logging
import time
import traceback
from typing import Any, Optional

from absl import flags

from axlearn.cloud.common.bundler import Bundler, bundler_flags
from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import REQUIRED, Required, config_class


class Job(FlagConfigurable):
    """Base Job definition.

    Job's main API method is `execute`, which sets up the environment according to `bundler`,
    runs the specified `command`, and retries if necessary.

    Subclasses of `Job` further specify the platform (e.g., TPUs on GCP) on which the job
    should run.

    The implementation of `execute` should be idempotent---invoking `execute` multiple times
    should be equivalent to invoking it once.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures Job."""

        # TODO(markblee): Convert all comments into config class docstrings.
        # Job name.
        name: Required[str] = REQUIRED
        # Max attempts to execute the Job.
        max_tries: Required[int] = REQUIRED
        # Retry interval in seconds.
        retry_interval: Required[float] = REQUIRED
        # Command to execute on remote compute.
        command: Optional[str] = None
        # Bundler. See `axlearn.cloud.common.bundler` for valid bundlers.
        bundler: Optional[Bundler.Config] = None

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._bundler = None
        if cfg.bundler:
            self._bundler: Bundler = cfg.bundler.instantiate()

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines absl flags to be read by `from_flags()`."""
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # Note: don't use generate_job_name() here, as not all environments define $USER.
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_integer("max_tries", None, "Max attempts to execute the job.", **common_kwargs)
        flags.DEFINE_integer(
            "retry_interval",
            None,
            "Interval in seconds between attempts.",
            **common_kwargs,
        )
        # Allow bundler to be optional.
        bundler_flags(required=False, **common_kwargs)

    @property
    def bundler(self):
        return self._bundler

    def _delete(self):
        """Cleans up the job. Called on termination when all retries are exhausted.

        Note that `_delete` is not called when `_execute` finishes successfully. It is up
        to the implementation of `_execute` to clean up properly.
        """

    def _execute(self) -> Any:
        """Performs some computation. The return value can be implementation dependent."""
        raise NotImplementedError(type(self))

    # TODO(markblee): Expand the API to include create/delete/start/stop.
    def execute(self) -> Any:
        """Wraps _execute with retries. All args and kwargs are forwarded.

        Retries are triggered automatically when _execute throws an exception.
        If all retries are exhausted, _delete is invoked to cleanup the job.
        """
        cfg = self.config

        try:
            return _with_retry(
                self._execute,
                interval=cfg.retry_interval,
                max_tries=cfg.max_tries,
            )
        except Exception:  # pylint: disable=broad-except
            # Cleanup on unexpected failure or exhausted retries.
            self._delete()
            raise


# TODO(markblee): Consider adding exponential backoff.
def _with_retry(fn, *args, max_tries: int = 10, interval: float = 0.1, **kwargs) -> Any:
    """Attempts fn(*args, **kwargs) `max_tries` times, sleeping `interval` seconds in between.

    Returns the output of the `fn` if successful, else raises a ValueError.
    """
    i = 1
    while max_tries < 0 or i <= max_tries:
        try:
            logging.info(
                "Execution attempt %s of %s to run %s with args %s.", i, max_tries, fn, args
            )
            return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Execution failed with error: %s", traceback.format_exc())
            if max_tries < 0 or i < max_tries:
                if interval > 1:
                    logging.warning("Sleeping for %ss after a failed attempt...", interval)
                time.sleep(interval)
            else:
                raise ValueError(f"Failed to execute {fn} within {max_tries} attempts.") from e
        i += 1
