# Copyright © 2024 Apple Inc.

"""A library to measure e2e metrics like goodput."""

import enum
import importlib
from typing import Optional, TypeVar

from absl import flags, logging

from axlearn.common.config import REQUIRED, Configurable, Required, config_class


class Event(enum.Enum):
    """Event to be recorded.

    Attributes:
        START_JOB: Start of job.
        END_JOB: End of job.
        START_STEP: Start of a training step. Should be recorded with `step` as a positional arg.
    """

    START_JOB = "START_JOB"
    END_JOB = "END_JOB"
    START_STEP = "START_STEP"


class Recorder(Configurable):
    """The base interface for collecting e2e metrics."""

    @config_class
    class Config(Configurable.Config):
        """Configures Recorder.

        Attributes:
            name: Name of the recorder.
        """

        name: Required[str] = REQUIRED

    @classmethod
    def from_flags(cls, fv: Optional[flags.FlagValues]) -> "Recorder":
        """Converts flags to a recorder."""
        raise NotImplementedError(cls)

    def record(self, event: Event, *args, **kwargs):
        """Records an event with the given name."""
        raise NotImplementedError(type(self))


_recorders: dict[str, type] = {}
_T = TypeVar("_T")


def register_recorder(name: str):
    def fn(cls: _T) -> _T:
        """Registers a recorder class for `get_recorder_config`."""
        if name in _recorders:
            raise ValueError(f"Recorder {name} is already registered.")
        _recorders[name] = cls
        return cls

    return fn


def define_flags(**kwargs):
    """Common measurement flags."""

    flags.DEFINE_string(
        "recorder_type",
        None,
        "The recorder type. It can be a recorder name, e.g. `my_recorder`, or "
        "a module paired with a recorder name, e.g. `my.module:my_recorder`.",
        **kwargs,
    )
    flags.DEFINE_multi_string(
        "recorder_spec",
        [],
        "Recorder spec provided as key=value. "
        "Refer to each recorders's `from_flags` method docstring for details.",
        **kwargs,
    )


global_recorder: Optional[Recorder] = None


def initialize(fv: flags.FlagValues):
    """Initializes the recorder from flags."""
    global global_recorder
    if not fv.recorder_type:
        logging.info("No recorder type specified, skipping initialize().")
        return
    if global_recorder is None:
        # Infer module from recorder_type.
        parts = fv.recorder_type.split(":", 1)
        if len(parts) > 1:
            logging.info("Registering recorders in %s", parts[0])
            importlib.import_module(parts[0])
        if recorder_class := _recorders.get(parts[-1], None):
            global_recorder = recorder_class.from_flags(fv=fv)
        else:
            raise NotImplementedError(
                f"Unknown recorder type: {fv.recorder_type}. "
                f"Supported types are: {sorted(list(_recorders.keys()))}\n"
                "You can also specify a specific module to identify the recorder "
                "(e.g., `my.module:my_recorder`)."
            )
        logging.info("Initialized global recorder: %s", global_recorder)
    else:
        logging.warning(
            "Recorder %s is already initialized, ignoring initialize().",
            global_recorder,
        )


def record_event(event: Event):
    """Records a global event."""
    if global_recorder is None:
        logging.log_first_n(logging.INFO, "No recorder configured, ignoring events.", 1)
    else:
        global_recorder.record(event)
