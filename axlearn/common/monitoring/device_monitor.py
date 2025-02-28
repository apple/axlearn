# Copyright © 2024 Apple Inc.

"""Device monitor module, to collect and report system metrics."""
import contextlib
import threading
from typing import Literal

from absl import logging

from axlearn.common.config import Configurable, config_class, maybe_instantiate
from axlearn.common.utils import DeviceUsage as Usage


class DeviceMonitorClient(Configurable):
    """Base Client for fetching metrics from devices."""

    @config_class
    class Config(Configurable.Config):
        """Configures DeviceMonitorClient."""

        # TODO(kelvin-zou): Add support for GPU and Trainium.
        platform: Literal["tpu", "gpu", "trainium"] = "tpu"

    def __init__(self, cfg: Config):
        """Initialize the DeviceMonitorClient."""
        super().__init__(cfg)
        cfg = self.config

    def collect_metrics(self) -> list[Usage]:
        """Collect metrics from the device, it should be empty."""
        return []

    def is_host_idle(self, usages: list[Usage]) -> bool:
        """Check if the devices on the host are idle, always return False."""
        # Make sure the usages are empty.
        assert usages == []
        return False


class DeviceMonitor(Configurable):
    """Device Monitor to collect and report system metrics.
    It also checks if the devices on the host are idle.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures DeviceMonitor.

        Fields:
            monitor_client: Config for the monitor client.
            check_interval_in_sec: The interval to check the system metrics, in seconds.
                0 to disable.
            log_every_n: The interval to log the system metrics in info log.
                Logs may be used for further anomaly detection.

        """

        monitor_client: DeviceMonitorClient.Config = DeviceMonitorClient.default_config()
        check_interval_in_sec: float = 60
        log_every_n: int = 10

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._monitor_client = maybe_instantiate(cfg.monitor_client)
        self._idle = False
        self._monitor_thread = None
        self._monitor_stopping = None

    @contextlib.contextmanager
    def start_monitoring(self):
        """Start the monitor."""
        self._start_monitoring()
        try:
            yield
        finally:
            self._stop_monitor()

    def is_host_idle(self) -> bool:
        """Check if the TPU device on the host are idle."""
        return self._idle

    def _check_host_and_log_metrics(self) -> bool:
        """Check if the devices on the host are idle."""
        cfg: DeviceMonitor.Config = self.config
        metrics: list[Usage] = self._monitor_client.collect_metrics()
        logging.log_every_n(
            logging.INFO, "%s metrics: %s", cfg.log_every_n, cfg.monitor_client.platform, metrics
        )
        return self._monitor_client.is_host_idle(metrics)

    def _start_monitoring(self):
        """Start the monitor."""
        if self.config.check_interval_in_sec > 0:
            self._monitor_stopping = threading.Event()
            self._monitor_thread = threading.Thread(
                name="device_monitor",
                target=self._monitor_loop,
            )
            self._monitor_thread.start()
            logging.info("_monitor_thread started.")

    def _stop_monitor(self):
        """Stops the monitor."""
        logging.info("Waiting for watchdog_thread to finish")
        if self._monitor_thread is not None:
            self._monitor_stopping.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            logging.info("_monitor_thread finished.")

    def _monitor_loop(self):
        while True:
            # Update the idle status.
            self._idle = self._check_host_and_log_metrics()
            if self._monitor_stopping.wait(timeout=self.config.check_interval_in_sec):
                break
        logging.info("monitor loop exit.")
