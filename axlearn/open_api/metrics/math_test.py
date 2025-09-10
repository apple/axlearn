# Copyright © 2024 Apple Inc.

"""Unit test for math.py."""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# pylint: disable=wrong-import-position
from axlearn.open_api.common import EvalGeneratorType
from axlearn.open_api.metrics.math import _get_judgement_from_generation, metric_fn

# pylint: enable=wrong-import-position


class TestMathEvaluator(unittest.IsolatedAsyncioTestCase):
    """Unit test for metric_fn in math.py."""

    def setUp(self):
        self.generator = AsyncMock()
        self.generator.config = MagicMock()
        self.generator.config.client.klass = MagicMock()

        self.messages = [
            [MagicMock(content="#ANSWER# math solution is x = 4", role="assistant")],
            [MagicMock(content="#ANSWER# math solution is x = 3", role="assistant")],
            [MagicMock(content="#ANSWER# math solution is x = 4", role="assistant")],
        ]
        self.judgement_messages = [
            [MagicMock(content="```python correct ```", role="assistant")],
            [MagicMock(content="```python incorrect ```", role="assistant")],
            [MagicMock(content="```python wrong ```", role="assistant")],
        ]
        self.responses = []
        for idx, message in enumerate(self.messages):
            self.responses.append(
                {
                    "response": json.dumps(
                        {
                            "choices": [
                                {"message": {"role": "assistant", "content": message[0].content}}
                            ]
                        }
                    ),
                    "raw_problem": "Solve 2x + 3 = 11",
                    "gt_answer": "x = 4",
                    "id": f"test{idx+1}",
                }
            )

    def test_get_judgement_from_generation(self):
        test_input = "```python correct ```"
        expected_output = "correct"
        self.assertEqual(_get_judgement_from_generation(test_input), expected_output)

        test_input = "```python incorrect ```"
        expected_output = "incorrect"
        self.assertEqual(_get_judgement_from_generation(test_input), expected_output)

    def test_compute_metrics(self):
        self.generator.config.client.klass.parse_generation.side_effect = (
            self.messages + self.judgement_messages
        )
        self.generator.async_generate_from_requests.return_value = [
            {
                "response": json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "```python correct ```", "role": "assistant"}}
                        ]
                    }
                ),
                "raw_problem": "Solve 2x + 3 = 11",
                "gt_answer": "x = 4",
                "id": "test1",
            },
            {
                "response": json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "```python incorrect ```", "role": "assistant"}}
                        ]
                    }
                ),
                "raw_problem": "Solve 2x + 3 = 11",
                "gt_answer": "x = 4",
                "id": "test3",
            },
            {
                "response": json.dumps(
                    {"choices": [{"message": {"content": "```wrong```", "role": "assistant"}}]}
                ),
                "raw_problem": "Solve 2x + 3 = 11",
                "gt_answer": "x = 4",
                "id": "test4",
            },
        ]
        metrics = metric_fn(
            responses=self.responses,
            generators={
                EvalGeneratorType.RESPONSE: self.generator,
                EvalGeneratorType.GRADER: self.generator,
            },
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["total_valid_answers"], 3)
        self.assertEqual(metrics["correct_judgments"], 1)
        self.assertEqual(metrics["incorrect_judgments"], 1)
        self.assertEqual(metrics["invalid_judgments"], 1)
