# Copyright © 2024 Apple Inc.

"""A library to measure e2e metrics like goodput."""

import importlib
from typing import Optional

from absl import flags, logging

from axlearn.common.measurement_base import (
    Event,
    EventType,
    Recorder,
    _recorders,
    define_flags,
    register_recorder,
)

__all__ = [
    "Event",
    "EventType",
    "Recorder",
    "define_flags",
    "register_recorder",
    "global_recorder",
    "initialize",
    "record_event",
    "start_monitoring",
]

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
    """A global utility to record an event via the `global_recorder`.

    Note:
        Do not call this function from within a
        `recorder.record_event()` context manager. Prefer using the
        `recorder.record_event()` in call-sites over this utility.
    """
    if global_recorder is None:
        logging.log_first_n(logging.INFO, "No recorder configured, ignoring events.", 1)
    else:
        global_recorder.record(event)


def start_monitoring():
    """A global utility to start monitoring metrics via the `global_recorder`.

    Note:
        Do not call this function from within a
        `recorder.maybe_monitor_all()` context manager. Prefer using the
        `recorder.maybe_monitor_all()` in call-sites over this utility.
    """
    if global_recorder is None:
        logging.log_first_n(
            logging.INFO, "Since recorder is not set up, monitoring cannot be started.", 1
        )
    else:
        global_recorder.start_monitoring()
        logging.info(
            "Starting monitoring of events using global recorder's monitor: %s", global_recorder
        )
