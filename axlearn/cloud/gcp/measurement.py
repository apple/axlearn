# Copyright © 2024 Apple Inc.

"""Measurement utils for GCP."""

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
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputRecorder can
        additionally take in following Tensorboard configs in the recorder_spec:
         - upload_dir: The directory to write Tensorboard data to.
         - upload_interval: The time interval in seconds at which to query and upload data
           to Tensorboard.
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
        """
        Instantiate ml-goodput-measurement's GoodputMonitor to asynchronously calculate
        Goodput and Badput at the upload_interval and upload to the specified TensorBoard
        directory.
        Note: This function requires initialization of distributed JAX before it is called.
        """
        if self._monitor is None:
            cfg: GoodputRecorder.Config = self.config
            self._monitor = goodput_monitoring.GoodputMonitor(
                job_name=cfg.name,
                logger_name=f"goodput_logger_{cfg.name}",
                tensorboard_dir=cfg.upload_dir,
                upload_interval=int(cfg.upload_interval),
                monitoring_enabled=(jax.process_index() == 0),
                include_badput_breakdown=True,
            )

        if self._monitor:
            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard in the background!")
        else:
            logging.log_first_n(
                logging.WARNING,
                "Goodput upload could not be started. Please check GoodputMonitor logs.",
                1,
            )
