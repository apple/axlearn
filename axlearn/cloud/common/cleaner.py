# Copyright © 2023 Apple Inc.

"""Utilities to clean resources."""

from typing import Dict, Sequence

from axlearn.cloud.common.types import ResourceMap
from axlearn.common.config import Configurable


class Cleaner(Configurable):
    """A basic cleaner interface."""

    # pylint: disable-next=unused-argument,no-self-use
    def sweep(self, jobs: Dict[str, ResourceMap]) -> Sequence[str]:
        """Removes resources in a non-blocking manner."""
        raise NotImplementedError(type(self))
