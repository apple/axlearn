# Copyright © 2025 Apple Inc.

"""Utilities to validate Jobs."""

from axlearn.cloud.common.types import JobSpec
from axlearn.common.config import Configurable


class JobValidator(Configurable):
    """A job validator interface."""

    def validate(self, job: JobSpec):
        """Raises ValidationError with reason if jobspec is invalid."""
        raise NotImplementedError(type(self))
