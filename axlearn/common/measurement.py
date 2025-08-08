# Copyright © 2024 Apple Inc.

"""A library to measure e2e metrics like goodput."""

import contextlib
import enum
import importlib
from typing import Optional, TypeVar

from absl import flags, logging

from axlearn.common.config import REQUIRED, Configurable, Required, config_class


class Event(enum.Enum):
    """Event to be recorded (Legacy).

    Attributes:
        START_JOB: Start of job.
        END_JOB: End of job.
        START_STEP: Start of a training step. Should be recorded with `step` as a positional arg.
        START_ACCELERATOR_INIT: Start of accelerator mesh initialization.
        END_ACCELERATOR_INIT: End of accelerator mesh initialization.
        START_TRAINING_PREPARATION: Start of training preparation.
        END_TRAINING_PREPARATION: End of training preparation.
        START_DATA_LOADING: Start of data loading.
        END_DATA_LOADING: End of data loading.
        START_CUSTOM_BADPUT_EVENT: Start of custom badput event.
        END_CUSTOM_BADPUT_EVENT: End of custom badput event.
    """

    START_JOB = "START_JOB"
    END_JOB = "END_JOB"
    START_STEP = "START_STEP"
    START_ACCELERATOR_INIT = "START_ACCELERATOR_INIT"
    END_ACCELERATOR_INIT = "END_ACCELERATOR_INIT"
    START_TRAINING_PREPARATION = "START_TRAINING_PREPARATION"
    END_TRAINING_PREPARATION = "END_TRAINING_PREPARATION"
    START_DATA_LOADING = "START_DATA_LOADING"
    END_DATA_LOADING = "END_DATA_LOADING"
    START_CUSTOM_BADPUT_EVENT = "START_CUSTOM_BADPUT_EVENT"
    END_CUSTOM_BADPUT_EVENT = "END_CUSTOM_BADPUT_EVENT"


class EventType(enum.Enum):
    """Event to be recorded.

    Attributes:
        JOB: Start and end of the job.
        STEP: Start of a training step. Should be recorded with `step` as a positional arg.
        ACCELERATOR_INIT: Start and end of accelerator mesh initialization.
        TRAINING_PREPARATION: Start and end of training preparation.
        DATA_LOADING: Start and end of data loading.
        CUSTOM_BADPUT_EVENT: Start and end of custom badput events.
    """

    JOB = "job"
    STEP = "step"
    ACCELERATOR_INIT = "tpu_init"
    TRAINING_PREPARATION = "training_preparation"
    DATA_LOADING = "data_loading"
    CUSTOM_BADPUT_EVENT = "custom_badput_event"


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
        """Records a single, instantaneous event.

        Note:
            This method is maintained for backward compatibility.
            New child recorder implementations should prioritize implementing the
            `record_event` context manager instead.
        """
        raise NotImplementedError(type(self))

    def start_monitoring(self, **kwargs):
        """Starts computing and uploading metrics in the background.

        Note:
            This method is maintained for backward compatibility. New child
            recorders should prioritize implementing the `maybe_monitor_all`
            context manager, which provides a clearer lifecycle for monitoring.
        """
        raise NotImplementedError(type(self))

    @contextlib.contextmanager
    def record_event(self, event: EventType, *args, **kwargs):
        """A context manager to record the start and end of an event.

        This is the preferred method for recording events. Child classes
        should implement this context manager to handle timed operations.

        Example:
            with recorder.record_event(EventType.ACCELERATOR_INIT):
                # Device initialization.
        """
        # pylint: disable=unnecessary-pass
        # pylint: disable=unused-argument
        try:
            yield
        finally:
            pass

    @contextlib.contextmanager
    def maybe_monitor_all(self):
        """Context manager to start and stop computing and monitoring metrics.

        This is the preferred method for monitoring events. Child classes
        should implement this context manager to manage all types of monitoring.

        Example:
            with recorder.maybe_monitor_all():
                # Train
        """
        yield


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
