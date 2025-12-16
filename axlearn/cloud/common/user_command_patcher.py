# Copyright © 2025 Apple Inc.

"""Command patcher interface for job processing."""

from typing import Any, Dict

from axlearn.common.config import Configurable


class UserCommandPatcher(Configurable):
    """A command patcher interface for modifying a job's user command."""

    def patch(self, command: str, **kwargs: Dict[str, Any]) -> str:
        """Patches the command string.

        Args:
            command: The original user command string.

        Returns:
            The patched command string.
        """
        raise NotImplementedError(type(self))
