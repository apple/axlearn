# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# vllm-project/vllm:
# Copyright 2023 The vLLM team. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Registers all available clients and uses lazy import via client name."""
import importlib
import logging
from typing import Optional

from axlearn.open_api import eval_set
from axlearn.open_api.common import BaseClient, EvalSet, MetricFn

# The dict of {client_name: (module, class)}.
_OPEN_API_CLIENTS_MODULE_CLASS = {
    "openai": ("openai", "OpenAIClient"),
    "gemini": ("gemini", "GeminiClient"),
    "anthropic": ("anthropic", "AnthropicClient"),
}

# The dict of {client_name: class}.
_OPEN_API_CLIENTS_CLASS: dict[str, type[BaseClient]] = {}


class ClientRegistry:
    """A registry of open api clients."""

    @staticmethod
    def load_client_cls(client_name: str) -> Optional[type[BaseClient]]:
        """Loads an open api client."""
        if client_name in _OPEN_API_CLIENTS_CLASS:
            return _OPEN_API_CLIENTS_CLASS[client_name]
        if client_name not in _OPEN_API_CLIENTS_MODULE_CLASS:
            return None
        module_name, client_cls_name = _OPEN_API_CLIENTS_MODULE_CLASS[client_name]
        module = importlib.import_module(f"axlearn.open_api.{module_name}")
        return getattr(module, client_cls_name, None)

    @staticmethod
    def get_supported_clients() -> list[str]:
        """Gets supported clients."""
        return list(_OPEN_API_CLIENTS_MODULE_CLASS.keys()) + list(_OPEN_API_CLIENTS_CLASS.keys())

    @staticmethod
    def register(client_name: str, client_cls: type[BaseClient]):
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


# The dict of {metric_name: (module, func)}.
_METRIC_MODULE_FUNC = {
    "tool_use_plan": ("tool_use_plan", "metric_fn"),
    "tool_use_execution": ("tool_use_execution", "metric_fn"),
    "math": ("math", "metric_fn"),
    "code_kaggle": ("code_kaggle", "metric_fn"),
    "code_kaggle_retry": ("code_kaggle", "retry_metric_fn"),
    "code_kaggle_oracle": ("code_kaggle", "oracle_metric_fn"),
    "code_contests": ("code_contests", "code_execution_metric_fn"),
    "code_contests_understand": ("code_contests", "understand_metric_fn"),
    "code_contests_plan": ("code_contests", "plan_metric_fn"),
    "code_contests_retry": ("code_contests", "retry_metric_fn"),
}

# The dict of {metric_name: func}.
_METRIC_FUNC: dict[str, MetricFn] = {}


class MetricRegistry:
    """A registry of metrics."""

    @staticmethod
    def load_metric(metric_name: str) -> Optional[MetricFn]:
        """Loads a metric compute function."""
        if metric_name in _METRIC_FUNC:
            return _METRIC_FUNC[metric_name]
        if metric_name not in _METRIC_MODULE_FUNC:
            return None
        module_name, metric_func_name = _METRIC_MODULE_FUNC[metric_name]
        module = importlib.import_module(f"axlearn.open_api.metrics.{module_name}")
        return getattr(module, metric_func_name, None)

    @staticmethod
    def get_supported_metrics() -> list[str]:
        """Gets supported metrics."""
        return list(_METRIC_FUNC.keys()) + list(_METRIC_MODULE_FUNC.keys())

    @staticmethod
    def register(metric_name: str, metric_func: MetricFn):
        """Registers a new metric.

        Args:
           metric_name: A string of metric name.
           metric_func: A function to compute metric.
        """
        if metric_name in _METRIC_FUNC:
            logging.warning(
                "Metric %s will be overwritten by the new metric function.",
                metric_name,
            )
        _METRIC_FUNC[metric_name] = metric_func


# The dict of {eval_set_name: (module, class)}.
_EVAL_SET_MODULE_CLASS = {
    "mmau_tool_use_plan": ("mmau", "ToolUsePlan"),
    "mmau_tool_use_execution": ("mmau", "ToolUseExecution"),
    "mmau_math": ("mmau", "Math"),
    "mmau_code_contests": ("mmau", "CodeContests"),
    "mmau": ("mmau", "MMAU"),
}

# The dict of {eval_set_name: class}.
_EVAL_SET_CLASS: dict[str, type[EvalSet]] = {}


class EvalSetRegistry:
    """A registry of eval sets."""

    @staticmethod
    def load(name: str) -> Optional[type[EvalSet]]:
        """Loads a eval set."""
        if name in _EVAL_SET_CLASS:
            return _EVAL_SET_CLASS[name]
        if name not in _EVAL_SET_MODULE_CLASS:
            return None
        module_name, metric_func_name = _EVAL_SET_MODULE_CLASS[name]
        module = importlib.import_module(f"{eval_set.__name__}.{module_name}")
        return getattr(module, metric_func_name, None)

    @staticmethod
    def get_supported() -> list[str]:
        """Gets supported."""
        return list(_EVAL_SET_CLASS.keys()) + list(_EVAL_SET_MODULE_CLASS.keys())

    @staticmethod
    def register(name: str, eval_set_type: type[EvalSet]):
        """Registers a new eval set.

        Args:
           name: A string of eval set name.
           eval_set_type: A customized eval set.
        """
        if name in _EVAL_SET_CLASS:
            logging.warning(
                "Eval set %s will be overwritten by the new eval set loader.",
                name,
            )
        _EVAL_SET_CLASS[name] = eval_set_type
