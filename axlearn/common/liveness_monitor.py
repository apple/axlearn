# Copyright © 2023 Apple Inc.

"""Liveness monitors for measuring trainer heartbeat."""

import threading
import time

from absl import logging
from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.io_wrapper import GetLogdirSubdirectories

from axlearn.common.config import REQUIRED, Configurable, Required, config_class


# TODO(markblee): Consider adding a /healthz-like endpoint for detecting stuck/down machines.
class LivenessMonitor(Configurable):
    """Monitors whether a job is stuck."""

    def started(self):
        """Whether monitor has started."""
        raise NotImplementedError(type(self))

    def reset(self):
        """Starts/resets the grace period."""
        raise NotImplementedError(type(self))

    def ping(self) -> bool:
        """Returns True iff job is alive."""
        raise NotImplementedError(type(self))


class TfSummaryMonitor(LivenessMonitor):
    """Monitors a TensorFlow summary dir."""

    @config_class
    class Config(LivenessMonitor.Config):
        # Directory to look for summaries. Will search subdirectories for events.
        summary_dir: Required[str] = REQUIRED
        # Maximum seconds after grace period without a heartbeat, before we declare the job stuck.
        max_timeout_seconds: Required[float] = REQUIRED
        # Allow for a period of time where we skip healthchecks.
        # This allows some time for training to start writing summaries.
        max_start_seconds: Required[float] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Mapping from summary dir name to EventAccumulator.
        self._accums = {}
        self._accums_mut = threading.Lock()
        self._latest = None
        self._start_time = None

    def started(self):
        """Whether monitor has started."""
        return self._start_time is not None

    def reset(self):
        """Starts/resets the grace period."""
        self._latest = time.time()
        self._start_time = self._latest
        logging.info("Started summary monitor at %s", self._start_time)

    def _iter(self):
        cfg = self.config
        with self._accums_mut:
            for summary_dir in GetLogdirSubdirectories(cfg.summary_dir):
                # TODO(markblee): Investigate why acc.most_recent_step is always -1.
                if summary_dir not in self._accums:
                    self._accums[summary_dir] = EventAccumulator(summary_dir)
            accums = list(self._accums.values())

        for acc in accums:
            yield from acc._generator.Load()  # pylint: disable=protected-access

    def ping(self) -> bool:
        """Returns True iff summaries are still updating."""
        cfg = self.config

        if not self.started():
            raise RuntimeError("Monitoring hasn't started yet.")

        # Look for the latest wall time in the summary dir.
        # Note that iter maintains its place each Load().
        try:
            for event in self._iter():
                self._latest = max(self._latest, event.wall_time)
        except DirectoryDeletedError:
            logging.warning("Summary dir is empty -- unable to infer health.")

        now = time.time()
        delta = now - self._latest
        logging.info(
            "Now: %s; Latest summary wall time: %s; Seconds since latest: %s",
            now,
            self._latest,
            delta,
        )

        if delta > cfg.max_timeout_seconds:
            # Check grace period.
            if time.time() - self._start_time <= cfg.max_start_seconds:
                logging.info("Still in grace period, skipping healthcheck.")
                return True
            logging.error("Healthcheck failed (exceeded max timeout %s)", cfg.max_timeout_seconds)
            return False

        return True
