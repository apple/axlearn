# Copyright © 2024 Apple Inc.

"""Base runner interface."""

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.gcp.job import GCPJob


class BaseRunnerJob(GCPJob):
    """Base runner job interface."""

    Config = GCPJob.Config

    def __init__(self, cfg: Config, *, bundler: Bundler):
        super().__init__(cfg)
        self._bundler = bundler
