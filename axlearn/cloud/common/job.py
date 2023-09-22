# Copyright © 2023 Apple Inc.

"""Utilities for executing on remote compute."""

import logging
import time
import traceback
from typing import Any, Optional

from absl import flags

from axlearn.cloud.common.bundler import Bundler
from axlearn.common.config import REQUIRED, Configurable, Required, config_class


class Job(Configurable):
    """Base Job definition."""

    @config_class
    class Config(Configurable.Config):
        """Configures Job."""

        # Job name.
        name: Required[str] = REQUIRED
        # Max attempts to execute the Job.
        max_tries: Required[int] = REQUIRED
        # Retry interval in seconds.
        retry_interval: Required[float] = REQUIRED
        # Command to execute on remote compute.
        command: Required[str] = REQUIRED
        # Bundler. See `axlearn.cloud.common.bundler` for valid bundlers.
        bundler: Optional[Bundler.Config] = None

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        if cfg.bundler:
            self._bundler: Bundler = cfg.bundler.instantiate()

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        """Populate config partially using parsed absl flags."""
        flag_values = {**fv.flag_values_dict(), **kwargs}
        cfg = cls.default_config()
        return cfg.set(
            **{field: flag_values[field] for field in cfg.keys() if field in flag_values}
        )

    def _delete(self):
        """Cleans up the job. Called on termination."""

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
