# Copyright © 2024 Apple Inc.

"""Goodput & Badput computation and monitoring utils for GCP."""

import jax
from absl import flags, logging
from ml_goodput_measurement import monitoring as goodput_monitoring

from axlearn.cloud.common.utils import parse_kv_flags
from axlearn.common import monitoring
from axlearn.common.config import maybe_set_config


@monitoring.register_monitor("GoodputMonitor")
class GoodputMonitor(monitoring.Monitor):
    """Computes and uploads overall training goodput and optionally badput."""

    Config = monitoring.Monitor.Config

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputMonitor":
        """Converts flags to a GoodputMonitor.

        `fv.monitor_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputMonitor can
        additionally take in following Tensorboard configs in the monitor_spec:
         - upload_dir: The directory to write Tensorboard data to.
         - upload_interval: The time interval in seconds at which to query and upload data
           to Tensorboard.
        """
        cfg: monitoring.Monitor.Config = cls.default_config()
        cfg = maybe_set_config(cfg, **parse_kv_flags(fv.monitor_spec, delimiter="="))
        return cfg.instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg: GoodputMonitor.Config = self.config
        self._monitor = None

    def start_monitoring(self, *args, **kwargs):
        # Instantiate ml-goodput-measurement's GoodputMonitor
        # to asynchronously calculate goodput and badput at
        # the upload_interval and upload to the specified
        # tensorboard directory.
        if self._monitor is None:
            cfg: GoodputMonitor.Config = self.config
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
