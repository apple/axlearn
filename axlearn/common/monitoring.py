# Copyright © 2024 Apple Inc.

"""Asynchronously compute and monitor metrics like goodput and badput."""

import importlib
from typing import Optional, TypeVar

from absl import flags, logging

from axlearn.common.config import REQUIRED, Configurable, Required, config_class


class Monitor(Configurable):
    """The base interface for computing and monitoring metrics."""

    @config_class
    class Config(Configurable.Config):
        """Configures any type of Monitor.

        Attributes:
            name: Name of the monitor (example: GoodputMonitor).
            upload_dir: Storage directory where metrics are uploaded.
            upload_interval: Time interval (seconds) at which to query and upload metrics.
        """

        name: Required[str] = REQUIRED
        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED

    @classmethod
    def from_flags(cls, fv: Optional[flags.FlagValues]) -> "Monitor":
        """Converts flags to a monitor."""
        raise NotImplementedError(cls)

    def start_monitoring(self, **kwargs):
        """Starts computing and uploading metrics at some configured interval in the background."""
        raise NotImplementedError(type(self))


_monitors: dict[str, type] = {}
_T = TypeVar("_T")


def register_monitor(name: str):
    def fn(cls: _T) -> _T:
        """Registers a monitor into a dict of global monitors with reference to its class type."""
        if name in _monitors:
            raise ValueError(f"Monitor {name} is already registered.")
        _monitors[name] = cls
        return cls

    return fn


def define_flags(**kwargs):
    """Common monitoring flags."""

    flags.DEFINE_string(
        "monitor_type",
        None,
        "The monitor type. It can be a monitor name, e.g. `GoodputMonitor`, or "
        "a module paired with a monitor name, e.g. `my.module:my_monitor`.",
        **kwargs,
    )
    flags.DEFINE_multi_string(
        "monitor_spec",
        [],
        "Monitor spec provided as key=value. "
        "Refer to each monitor's `from_flags` method docstring for details.",
        **kwargs,
    )


global_monitor: Optional[Monitor] = None


def initialize(fv: flags.FlagValues):
    """Initializes the monitor from flags."""
    global global_monitor
    if not fv.monitor_type:
        logging.info("No monitor type specified, skipping monitoring initialize().")
        return
    if global_monitor is None:
        # Infer module from monitor_type.
        parts = fv.monitor_type.split(":", 1)
        if len(parts) > 1:
            logging.info("Registering monitors in %s", parts[0])
            importlib.import_module(parts[0])
        if monitor_class := _monitors.get(parts[-1], None):
            # This will instantiate a specific monitor of monitor_type if supported.
            global_monitor = monitor_class.from_flags(fv=fv)
        else:
            raise NotImplementedError(
                f"Monitor type: {fv.monitor_type} is not supported. "
                f"Supported types are: {sorted(list(_monitors.keys()))}\n"
                "You can also specify a specific module to identify the monitor "
                "(e.g., `my.module:my_monitor`)."
            )
        logging.info("Initialized global monitor: %s", global_monitor)
    else:
        logging.warning(
            "Monitor %s is already initialized, ignoring monitoring initialize().",
            global_monitor,
        )


def start_monitoring():
    """Begins monitoring events as per global monitor functionality."""
    if global_monitor is None:
        logging.log_first_n(logging.INFO, "No Monitor configured, no events will be monitored.", 1)
    else:
        global_monitor.start_monitoring()
        logging.info("Starting monitoring of events using global monitor: %s", global_monitor)
