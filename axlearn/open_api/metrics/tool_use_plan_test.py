# Copyright © 2024 Apple Inc.

"""Unit tests for tool_use_plan.py."""
import json
import unittest
from unittest.mock import MagicMock

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# pylint: disable=wrong-import-position
from axlearn.open_api.common import EvalGeneratorType
from axlearn.open_api.metrics.tool_use_plan import metric_fn

# pylint: enable=wrong-import-position


class TestMetricFn(unittest.TestCase):
    """Unit tests for metric_fn in tool_use_plan.py."""

    def setUp(self):
        self.responses = [
            {"target_plan_number": 1, "response": json.dumps({"content": "**Chosen Plan:** 1"})},
            {"target_plan_number": 2, "response": json.dumps({"content": "**Chosen Plan:** 2"})},
        ]

        self.generator = MagicMock()
        self.generator.config = MagicMock()
        self.generator.config.client.klass = MagicMock()

        self.grader_generator = MagicMock()

    def test_metric_fn_success(self):
        self.generator.config.client.klass.parse_generation.side_effect = [
            [MagicMock(content="**Chosen Plan:** 1")],
            [MagicMock(content="**Chosen Plan:** 2")],
        ]

        metrics = metric_fn(
            responses=self.responses,
            generators={
                EvalGeneratorType.RESPONSE: self.generator,
                EvalGeneratorType.GRADER: self.grader_generator,
            },
            debug=False,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["instruction_following_error"], 0.0)
        self.assertEqual(metrics["number_of_examples"], 2)

    def test_metric_fn_mismatch(self):
        self.generator.config.client.klass.parse_generation.side_effect = [
            [MagicMock(content="**Chosen Plan:** 1")],
            [MagicMock(content="**Chosen Plan:** 3")],
        ]

        metrics = metric_fn(
            responses=self.responses,
            generators={
                EvalGeneratorType.RESPONSE: self.generator,
                EvalGeneratorType.GRADER: self.grader_generator,
            },
            debug=False,
        )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["instruction_following_error"], 0.0)
        self.assertEqual(metrics["number_of_examples"], 2)

    def test_metric_fn_with_debug(self):
        self.generator.config.client.klass.parse_generation = MagicMock(
            return_value=[MagicMock(content="**Chosen Plan:** 3")]
        )

        with self.assertLogs(level="DEBUG") as log:
            metrics = metric_fn(
                responses=self.responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                    EvalGeneratorType.GRADER: self.grader_generator,
                },
                debug=True,
            )

        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["instruction_following_error"], 0.0)
        self.assertEqual(metrics["number_of_examples"], 2)
        self.assertIn("deliverable_id", log.output[0])

    def test_metric_fn_invalid_response(self):
        self.generator.config.client.klass.parse_generation = MagicMock(
            return_value=[MagicMock(content="**Chosen Plan:** X")]
        )

        metrics = metric_fn(
            responses=self.responses,
            generators={
                EvalGeneratorType.RESPONSE: self.generator,
                EvalGeneratorType.GRADER: self.grader_generator,
            },
            debug=False,
        )

        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["instruction_following_error"], 1.0)
        self.assertEqual(metrics["number_of_examples"], 2)
