# Copyright © 2023 Apple Inc.

"""Utilities for executing on remote compute.

On configuration:

Jobs are `FlagConfigurable`s and are typically configured via command line flags.

While flexible, flags are global objects and may potentially complicate the design of the configs.
For example, it can be hard to identify where configs or flags (and their default values) are
defined, or which configs are propagated from parent to child as opposed to being read from flags.

To avoid such ambiguities, we apply a few rules of thumb when implementing a Job:

1. Define flags in the same class that defines the configs.
* This keeps the flag interface and the config interface of a module consistent and encapsulated.
* Default to `override=True` so that other classes can define the same flag.
    This allows other classes to minimize assumptions about what config flags have already been
    defined.
* Avoid defining defaults in `define_flags` so that defaults do not need to be repeated across
    flag definitions.
    Instead, consider whether the default can be inherited, can be local to the class, or needs to
    be a global default that propagates to descendant configs.
    * If a default can be inherited, simply implement `define_flags` with defaults left as None.
        `FlagConfigurable` will handle the default propagation from parents.
    * In the local case, prefer to assign a default to the instantiated config after `from_flags`;
        this ensures that the default never "leaks" into child configs.
    * If a global default is justified, use `fv.set_default` in `set_defaults`.
        Using `set_defaults` allows for more predictable default behavior, where parent defaults
        propagate to descendants unless a descendant itself overrides the default.

2. Similarly, define configs within the Job that consumes the configs.
* Avoid defining configs at a module unless the module itself requires the config.
    For example, instead of defining "command" at the parent and propagating it to each child,
    define "command" at the leaf config that executes the command.
    This allows for better composition, because different child configs within the same parent job
    can execute different commands.
* Minimize the number of "pass-through" configs, i.e. configs that are directly `set` on a
    child config.
    If a value needs to be "passed-through", consider using flag values to propagate (using
    `set_defaults` as discussed above), since flag values can be read by descendants that are
    multiple layers deep without requiring each layer to forward its configs.
"""

import logging
import time
import traceback
from typing import Any

from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import REQUIRED, Required, config_class


# TODO(markblee): Consider replacing `Job` with a decorator. This is more flexible than subclassing:
# 1. It allows jobs to omit max_tries/retry_interval configs if retries are not necessary;
# 2. It allows the job to subclass from other base classes, not just `Job`.
# 3. It allows jobs to implement interfaces other than `_execute`.
class Job(FlagConfigurable):
    """Base Job definition.

    Job's main API method is `execute`, which sets up the environment according to `bundler`,
    runs the `_execute` method, and retries if necessary.

    Subclasses of `Job` further specify the platform (e.g., TPUs on GCP) on which the job should
    run, as well as the implementation of `_execute` (e.g., what commands to run).

    The implementation of `execute` should be idempotent---invoking `execute` multiple times
    should be equivalent to invoking it once.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures Job.

        Attributes:
            max_tries: Max attempts to execute the Job.
            retry_interval: Retry interval in seconds.
        """

        max_tries: Required[int] = REQUIRED
        retry_interval: Required[float] = REQUIRED

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines absl flags to be read by `from_flags()`.

        Avoid assigning default values in this method. Instead, defaults can be assigned in
        `set_defaults`.

        See also ``On configuration`` in file docstring.
        """
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_integer("max_tries", None, "Max attempts to execute the job.", **common_kwargs)
        flags.DEFINE_integer(
            "retry_interval",
            None,
            "Interval in seconds between attempts.",
            **common_kwargs,
        )

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
