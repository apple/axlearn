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
        --recorder_spec=step_deviation_interval_seconds=30

"""

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
            step_deviation_interval_seconds: Time interval (seconds) for step deviation metrics
            uploads. -1 to disable step deviation uploads.
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED
        step_deviation_interval_seconds: int = 30  # Default to 30 seconds

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputRecorder can
        additionally take in following Tensorboard configs in the recorder_spec:
         - upload_dir: The directory to write Tensorboard data to.
         - upload_interval: The time interval in seconds at which to query and upload data
           to Tensorboard.
        - step_deviation_interval_seconds: Time interval (seconds) for step deviation metrics
        uploads. Set to less than or equal to 0 to disable step deviation uploads.
        """
        cfg: measurement.Recorder.Config = cls.default_config()
        cfg = maybe_set_config(cfg, **parse_kv_flags(fv.recorder_spec, delimiter="="))
        return cfg.instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg: GoodputRecorder.Config = self.config
        self._recorder = None
        self._monitor = None

    def record(self, event: measurement.Event, *args, **kwargs):
        # Lazily instantiate the recorder. This avoids invoking jax before setup is complete.
        if self._recorder is None:
            cfg: GoodputRecorder.Config = self.config
            self._recorder = goodput.GoodputRecorder(
                job_name=cfg.name,
                logger_name=f"goodput_logger_{cfg.name}",
                logging_enabled=(jax.process_index() == 0),
            )

        if event == measurement.Event.START_JOB:
            self._recorder.record_job_start_time(*args, **kwargs)
        elif event == measurement.Event.END_JOB:
            self._recorder.record_job_end_time(*args, **kwargs)
        elif event == measurement.Event.START_STEP:
            self._recorder.record_step_start_time(*args, **kwargs)
        elif event == measurement.Event.START_ACCELERATOR_INIT:
            self._recorder.record_tpu_init_start_time(*args, **kwargs)
        elif event == measurement.Event.END_ACCELERATOR_INIT:
            self._recorder.record_tpu_init_end_time(*args, **kwargs)
        elif event == measurement.Event.START_TRAINING_PREPARATION:
            self._recorder.record_training_preparation_start_time(*args, **kwargs)
        elif event == measurement.Event.END_TRAINING_PREPARATION:
            self._recorder.record_training_preparation_end_time(*args, **kwargs)
        elif event == measurement.Event.START_DATA_LOADING:
            self._recorder.record_data_loading_start_time(*args, **kwargs)
        elif event == measurement.Event.END_DATA_LOADING:
            self._recorder.record_data_loading_end_time(*args, **kwargs)
        else:
            logging.log_first_n(
                logging.WARNING,
                "Ignoring unknown event %s",
                1,
                event,
            )

    def start_monitoring(self, *args, **kwargs):
        """Starts Monitoring of Goodput.

        Instantiate ml-goodput-measurement's GoodputMonitor to asynchronously calculate
        Goodput and Badput at the upload_interval and upload to the specified TensorBoard
        directory.
        Note: This function requires initialization of distributed JAX before it is called.
        If there are internal GCP errors from querying and uploading data, these will be
        logged without affecting the workload. GoodputMonitor logs will provide further
        information if data is not being uploaded correctly.

        Default behavior is to push metrics to Google Cloud Monitoring.
        This behavior can be overridden by configuring `goodput_monitoring.GCPOptions`
        """
        cfg: GoodputRecorder.Config = self.config
        include_step_deviation = True
        if jax.process_index() == 0:
            if self._monitor is None:
                if int(cfg.step_deviation_interval_seconds) <= 0:
                    include_step_deviation = False

                gcp_options = goodput_monitoring.GCPOptions(
                    enable_gcp_goodput_metrics=True,
                    enable_gcp_step_deviation_metrics=include_step_deviation,
                )
                self._monitor = goodput_monitoring.GoodputMonitor(
                    job_name=cfg.name,
                    logger_name=f"goodput_logger_{cfg.name}",
                    tensorboard_dir=cfg.upload_dir,
                    upload_interval=int(cfg.upload_interval),
                    monitoring_enabled=True,
                    include_badput_breakdown=True,
                    include_step_deviation=include_step_deviation,
                    step_deviation_interval_seconds=int(cfg.step_deviation_interval_seconds),
                    gcp_options=gcp_options,
                )

            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard & GCM in the background!")
            if include_step_deviation:
                self._monitor.start_step_deviation_uploader(*args, **kwargs)
                logging.info(
                    "Started Step Deviation upload to Tensorboard & GCM in the background!"
                )
