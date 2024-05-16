# Copyright © 2024 Apple Inc.

"""Measurement utils for GCP."""

import jax
from absl import flags, logging
from ml_goodput_measurement import goodput

from axlearn.cloud.common.utils import parse_kv_flags
from axlearn.common import measurement
from axlearn.common.config import maybe_set_config


@measurement.register_recorder("goodput")
class GoodputRecorder(measurement.Recorder):
    """Records overall training goodput."""

    Config = measurement.Recorder.Config

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values.
        """
        cfg: measurement.Recorder.Config = cls.default_config()
        cfg = maybe_set_config(cfg, **parse_kv_flags(fv.recorder_spec, delimiter="="))
        return cfg.instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg: GoodputRecorder.Config = self.config
        self._recorder = None

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
        else:
            logging.log_first_n(
                logging.WARNING,
                "Ignoring unknown event %s",
                1,
                event,
            )
