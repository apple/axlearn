# Copyright © 2024 Apple Inc.

"""Measurement utils for GCP.

    Example:

    # Enable Goodput when launching an AXLearn training job
    axlearn gcp launch run --instance_type=tpu-v5litepod-16 \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile \
        -- python3 -m my_training_job \
        --recorder_type=axlearn.cloud.gcp.measurement:goodput \
        --recorder_spec=name=my-run-with-goodput \
        --recorder_spec=upload_dir=my-output-directory/summaries \
        --recorder_spec=upload_interval=30 \
        --recorder_spec=rolling_window_size=86400,259200,432000

"""

import contextlib
import os
from typing import Optional, Sequence

import jax
from absl import flags, logging
from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring as goodput_monitoring

from axlearn.cloud.common.utils import parse_kv_flags
from axlearn.common import measurement
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config


@measurement.register_recorder("goodput")
class GoodputRecorder(measurement.Recorder):
    """Records overall training goodput."""

    @config_class
    class Config(measurement.Recorder.Config):
        """Configures GoodputRecorder.

        Attributes:
            upload_dir: Directory to store metrics for the monitor.
            upload_interval: Time interval (seconds) for monitoring uploads.
            rolling_window_size: A sequence of integers defining the rolling window sizes in
                seconds.
            jax_backend: Jax backend type to infer Pathways environment.
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED
        rolling_window_size: Sequence[int] = []
        jax_backend: Optional[str] = None

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputRecorder can
        additionally take in following Tensorboard configs in the recorder_spec:
        - upload_dir: The directory to write Tensorboard data to.
        - upload_interval: The time interval in seconds at which to query and upload data
            to Tensorboard.
        - rolling_window_size: Comma-separated list of integers representing rolling window
            sizes in seconds.
        - jax_backend: The type of jax backend.
        """
        cfg: measurement.Recorder.Config = cls.default_config()
        parsed_flags = parse_kv_flags(fv.recorder_spec, delimiter="=")
        if "upload_interval" in parsed_flags:
            parsed_flags["upload_interval"] = int(parsed_flags["upload_interval"])
        if "rolling_window_size" in parsed_flags and isinstance(
            parsed_flags["rolling_window_size"], str
        ):
            parsed_flags["rolling_window_size"] = [
                int(x) for x in parsed_flags["rolling_window_size"].split(",")
            ]
        return maybe_set_config(cfg, **parsed_flags).instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        self._recorder: Optional[goodput.GoodputRecorder] = None
        self._monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._rolling_window_monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._job_name = cfg.name
        self._logger_name = f"goodput_logger_{cfg.name}"

    @contextlib.contextmanager
    def record_event(self, event: measurement.Event, *args, **kwargs):
        """Records a goodput event using a context manager."""
        # Lazily instantiate the recorder if it hasn't been already.
        if self._recorder is None:
            if jax.process_index() == 0:
                logging.info("Lazily instantiating goodput recorder.")
            self._recorder = goodput.GoodputRecorder(
                job_name=self._job_name,
                logger_name=self._logger_name,
                logging_enabled=(jax.process_index() == 0),
            )

        start_method_name = f"record_{event.value}_start_time"
        end_method_name = f"record_{event.value}_end_time"

        record_event_start = getattr(self._recorder, start_method_name, None)
        record_event_end = getattr(self._recorder, end_method_name, None)

        try:
            if record_event_start:
                record_event_start(*args, **kwargs)
        except RuntimeError as e:
            logging.warning(
                "Failed to record start of event %s. Error: %s", event.value, e, exc_info=True
            )

        try:
            yield
        finally:
            try:
                if record_event_end:
                    record_event_end(*args, **kwargs)
            except RuntimeError as e:
                logging.warning(
                    "Failed to record end of event %s. Error: %s", event.value, e, exc_info=True
                )

    @contextlib.contextmanager
    def maybe_monitor_goodput(self, *args, **kwargs):
        """Monitor cumulative goodput if enabled.

        Instantiate ml-goodput-measurement's GoodputMonitor to asynchronously calculate
        Goodput, Badput, Step & Disruption Information at the upload_interval to the
        specified TensorBoard directory and Google Cloud Monitoring.
        Note: This function requires initialization of distributed JAX before it is called.
        If there are internal GCP errors from querying and uploading data, these will be
        logged without affecting the workload. GoodputMonitor logs will provide further
        information if data is not being uploaded correctly.

        Default behavior is to push metrics to Google Cloud Monitoring.
        This behavior can be overridden by configuring `goodput_monitoring.GCPOptions`
        """
        if jax.process_index() != 0:
            yield
            return
        try:
            if self._monitor is None:
                self._monitor = goodput_monitoring.GoodputMonitor(
                    job_name=self._job_name,
                    logger_name=self._logger_name,
                    tensorboard_dir=self.config.upload_dir,
                    upload_interval=self.config.upload_interval,
                    monitoring_enabled=True,
                    pathway_enabled=self.config.jax_backend == "proxy",
                    include_badput_breakdown=True,
                )

            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard & GCM in the background!")
            yield
        finally:
            if self._monitor:
                self._monitor.stop_goodput_uploader()
                logging.info("Flushed final metrics and safe exited from Goodput monitoring.")

    @contextlib.contextmanager
    def maybe_monitor_rolling_window_goodput(self):
        """Monitor rolling window goodput if enabled."""
        if not self.config.rolling_window_size or jax.process_index() != 0:
            yield
            return
        try:
            if self._rolling_window_monitor is None:
                rolling_window_tensorboard_dir = os.path.join(
                    self.config.upload_dir, f"rolling_window_{self.config.name}"
                )
                self._rolling_window_monitor = goodput_monitoring.GoodputMonitor(
                    job_name=self._job_name,
                    logger_name=self._logger_name,
                    tensorboard_dir=rolling_window_tensorboard_dir,
                    upload_interval=self.config.upload_interval,
                    monitoring_enabled=True,
                    pathway_enabled=self.config.jax_backend == "proxy",
                    include_badput_breakdown=True,
                )
            self._rolling_window_monitor.start_rolling_window_goodput_uploader(
                self.config.rolling_window_size
            )
            logging.info("Started Rolling Window Goodput monitoring in the background!")
            yield
        finally:
            if self._rolling_window_monitor:
                self._rolling_window_monitor.stop_rolling_window_goodput_uploader()
                logging.info(
                    "Flushed final metrics and safe exited from Rolling Window Goodput monitoring."
                )


def create_goodput_recorder(cfg: GoodputRecorder.Config):
    """Factory method to create GoodputRecorder."""
    return GoodputRecorder(cfg)
