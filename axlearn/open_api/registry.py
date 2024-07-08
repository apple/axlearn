# Copyright © 2024 Apple Inc.

# Some of the code in this file is adapted from:
#
# vllm-project/vllm:
# Copyright 2023 The vLLM team. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Registers all available clients and uses lazy import via client name."""
import importlib
import logging
from typing import Dict, List, Optional, Type

from axlearn.open_api.common import BaseClient

# The dict of {client_name: (module, class)}.
_OPEN_API_CLIENTS_MODULE_CLASS = {
    "openai": ("openai", "OpenAIClient"),
    "gemini": ("gemini", "GeminiClient"),
    "anthropic": ("anthropic", "AnthropicClient"),
}

# The dict of {client_name: class}.
_OPEN_API_CLIENTS_CLASS: Dict[str, Type[BaseClient]] = {}


class ClientRegistry:
    """A registry of open api clients."""

    @staticmethod
    def load_client_cls(client_name: str) -> Optional[Type[BaseClient]]:
        """Loads an open api client."""
        if client_name in _OPEN_API_CLIENTS_CLASS:
            return _OPEN_API_CLIENTS_CLASS[client_name]
        if client_name not in _OPEN_API_CLIENTS_MODULE_CLASS:
            return None
        module_name, client_cls_name = _OPEN_API_CLIENTS_MODULE_CLASS[client_name]
        module = importlib.import_module(f"axlearn.open_api.{module_name}")
        return getattr(module, client_cls_name, None)

    @staticmethod
    def get_supported_clients() -> List[str]:
        """Gets supported clients."""
        return list(_OPEN_API_CLIENTS_MODULE_CLASS.keys()) + list(_OPEN_API_CLIENTS_CLASS.keys())

    @staticmethod
    def register(client_name: str, client_cls: Type[BaseClient]):
        """Registers a new client.

        Args:
           client_name: A string of client name.
           client_cls: A class of Open API client in the type of BaseClient.
        """
        if client_name in _OPEN_API_CLIENTS_CLASS:
            logging.warning(
                "Client %s will be overwritten by the new client class %s.",
                client_name,
                client_cls.__name__,
            )
        _OPEN_API_CLIENTS_CLASS[client_name] = client_cls
