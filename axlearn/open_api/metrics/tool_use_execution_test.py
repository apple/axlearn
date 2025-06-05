# Copyright © 2024 Apple Inc.
"""Unit test for tool_use_execution.py."""

import json
from typing import Union
from unittest.mock import MagicMock, Mock, patch

import pytest
from absl.testing import parameterized

from axlearn.open_api import mock_utils

with mock_utils.mock_openai_package():
    # pylint: disable=wrong-import-position
    from axlearn.open_api.common import EvalGeneratorType
    from axlearn.open_api.metrics.tool_use_execution import metric_fn

# pylint: enable=wrong-import-position


@mock_utils.safe_mocks(mock_utils.mock_openai_package)
class TestToolUseExecution(parameterized.TestCase):
    """Unit tests for tool_use_execution."""

    def setUp(self):
        self.generator = MagicMock()
        self.generator.config = MagicMock()
        self.generator.config.client.klass = MagicMock()

        self.grader_generator = MagicMock()

    def test_empty_responses(self):
        """Test compute_metrics with an empty list of responses."""
        responses = []
        metrics = metric_fn(
            responses=responses,
            generators={
                EvalGeneratorType.RESPONSE: self.generator,
                EvalGeneratorType.GRADER: self.generator,
            },
        )
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["number_of_examples"], 0)

    def test_responses_without_target_message(self):
        """Tests with responses that lack target_message field."""
        responses = [
            {
                "response": json.dumps({"content": "Test message"}),
            }
        ]
        with self.assertRaises(ValueError):
            metric_fn(
                responses=responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                    EvalGeneratorType.GRADER: self.generator,
                },
            )

    @parameterized.parameters(
        # Date normalization.
        dict(
            target_message_match_rules=[{"arguments": {"date": {"date_match": True}}}],
            target_arguments=json.dumps({"date": "4/2/2024"}),
            accuracy=1,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
        ),
        # Date normalization but incorrect argument.
        dict(
            target_message_match_rules=[{"arguments": {"date": {"date_match": True}}}],
            target_arguments=json.dumps({"date": "4/2024"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
        ),
        # No date normalization.
        dict(
            target_message_match_rules=[{"arguments": {"date": {"date_match": False}}}],
            target_arguments=json.dumps({"date": "4/2/2024"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
        ),
        # Mutiple choices.
        dict(
            target_message_match_rules=[
                {"arguments": {"date": {"multi_choices": ["4/2/2024", "April 2, 2024"]}}}
            ],
            target_arguments=json.dumps({"date": "4/2/2024"}),
            accuracy=1,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
        ),
        # Mutiple choices.
        dict(
            target_message_match_rules=[
                {"arguments": {"date": {"multi_choices": ["4/2/2023", "April 2, 2023"]}}}
            ],
            target_arguments=json.dumps({"date": "4/2/2024"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
        ),
        # Regex match.
        dict(
            target_message_match_rules=[
                {"arguments": {"location": {"regex_match": r"^cupertino"}}}
            ],
            target_arguments=json.dumps({"location": "cupertino."}),
            accuracy=1,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments=json.dumps({"location": "Cupertino, CA"}),
        ),
        # Regex match.
        dict(
            target_message_match_rules=[{"arguments": {"location": {"regex_match": r"^san jose"}}}],
            target_arguments=json.dumps({"location": "cupertino."}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments=json.dumps({"location": "Cupertino, CA"}),
        ),
        # Partial match.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino", "unit": "celcius"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
            pred_arguments=json.dumps({"location": "cupertino"}),
        ),
        # String match.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino"}),
            accuracy=1,
            func_name_accuracy=1,
            strict_accuracy=1,
            lenient_accuracy=1,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments=json.dumps({"location": "cupertino"}),
        ),
        # Punctuation normalization match.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino"}),
            accuracy=1,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=1,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments=json.dumps({"location": "cupertino."}),
        ),
        # Invalid arguments json format.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino"}),
            accuracy=0,
            func_name_accuracy=0,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=0,
            number_of_expected_tool_calls=1,
            pred_arguments='{"location": "cupertino"',
        ),
        # Lenient match.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=1,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments='{"location": "my cupertino"}',
        ),
        # BOW match.
        dict(
            target_message_match_rules=None,
            target_arguments=json.dumps({"location": "cupertino"}),
            accuracy=0,
            func_name_accuracy=1,
            strict_accuracy=0,
            lenient_accuracy=0,
            bow_accuracy=1,
            number_of_expected_tool_calls=1,
            pred_arguments='{"location": "cupertino CA"}',
        ),
    )
    @pytest.mark.skip(reason="Flaky in CI. TODO(gyin94): Fix and re-enable.")
    def test_match_rules(
        self,
        target_arguments,
        accuracy,
        func_name_accuracy,
        strict_accuracy,
        lenient_accuracy,
        bow_accuracy,
        number_of_expected_tool_calls,
        target_message_match_rules=None,
        pred_arguments=None,
    ):
        """Tests where responses match the targets"""
        pred_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "arguments": pred_arguments or json.dumps({"date": "April 2, 2024"}),
                        "name": "get_weather",
                    },
                }
            ],
        }
        target_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "arguments": target_arguments,
                        "name": "get_weather",
                    },
                }
            ],
        }
        responses = [
            {
                "response": json.dumps({"choices": [{"message": pred_message}]}),
                "target_message": target_message,
            }
        ]
        if target_message_match_rules:
            responses[0].update({"target_message_match_rules": target_message_match_rules})
        mock_target_message = Mock(**target_message)
        mock_target_message.model_dump.return_value = target_message
        mock_pred_message = Mock(**pred_message)
        mock_pred_message.model_dump.return_value = pred_message
        self.generator.config.client.klass.parse_generation.return_value = [mock_pred_message]

        with patch(
            "axlearn.open_api.openai.OpenAIClient.format_message", return_value=mock_target_message
        ):
            metrics = metric_fn(
                responses=responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                    EvalGeneratorType.GRADER: self.generator,
                },
            )
            self.assertEqual(metrics["accuracy"], accuracy)
            self.assertEqual(metrics["func_name_accuracy"], func_name_accuracy)
            self.assertEqual(metrics["strict_accuracy"], strict_accuracy)
            self.assertEqual(metrics["lenient_accuracy"], lenient_accuracy)
            self.assertEqual(metrics["bow_accuracy"], bow_accuracy)
            self.assertEqual(
                metrics["number_of_expected_tool_calls"], number_of_expected_tool_calls
            )
            self.assertEqual(metrics["number_of_func_call_intents_ground_truth"], 1)
            self.assertEqual(metrics["number_of_func_call_intents_pred"], 1)
            self.assertEqual(metrics["func_intent_recall"], 1)
            self.assertEqual(metrics["func_intent_precision"], 1)

    def test_empty_pred(self):
        pred_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": None,
        }
        target_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "arguments": {"location": "cupertino"},
                        "name": "get_weather",
                    },
                }
            ],
        }
        responses = [
            {
                "response": json.dumps({"choices": [{"message": pred_message}]}),
                "target_message": target_message,
            }
        ]
        mock_target_message = Mock(**target_message)
        mock_target_message.model_dump.return_value = target_message
        mock_pred_message = Mock(**pred_message)
        mock_pred_message.model_dump.return_value = pred_message
        self.generator.config.client.klass.parse_generation.return_value = [mock_pred_message]

        with patch(
            "axlearn.open_api.openai.OpenAIClient.format_message", return_value=mock_target_message
        ):
            metrics = metric_fn(
                responses=responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                    EvalGeneratorType.GRADER: self.generator,
                },
            )
            self.assertEqual(metrics["accuracy"], 0.0)
            self.assertEqual(metrics["number_of_func_call_intents_ground_truth"], 1)
            self.assertEqual(metrics["number_of_func_call_intents_pred"], 0)
            self.assertEqual(metrics["func_intent_recall"], 0)
            self.assertEqual(metrics["func_intent_precision"], 0)

    @parameterized.parameters(
        dict(
            pred_tool_calls=None,
            target_tool_calls=[],
            expected_number_of_func_call_intents_pred=0,
            expected_number_of_func_call_intents_ground_truth=0,
            expected_func_intent_recall=0.0,
            expected_func_intent_precision=0.0,
        ),
        dict(
            pred_tool_calls=[],
            target_tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                }
            ],
            expected_number_of_func_call_intents_pred=0,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_func_intent_recall=0.0,
            expected_func_intent_precision=0.0,
        ),
        dict(
            pred_tool_calls=[
                {"id": "call_0", "type": "function", "function": {"name": "get", "arguments": {}}}
            ],
            target_tool_calls=[],
            expected_number_of_func_call_intents_pred=1,
            expected_number_of_func_call_intents_ground_truth=0,
            expected_func_intent_recall=0.0,
            expected_func_intent_precision=0.0,
        ),
        dict(
            pred_tool_calls=[
                {"id": "call_0", "type": "function", "function": {"name": "get", "arguments": {}}}
            ],
            target_tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                }
            ],
            expected_number_of_func_call_intents_pred=1,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_func_intent_recall=1.0,
            expected_func_intent_precision=1.0,
        ),
    )
    def test_intents(
        self,
        pred_tool_calls,
        target_tool_calls,
        expected_number_of_func_call_intents_pred,
        expected_number_of_func_call_intents_ground_truth,
        expected_func_intent_recall,
        expected_func_intent_precision,
    ):
        pred_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": pred_tool_calls,
        }
        target_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": target_tool_calls,
        }
        responses = [
            {
                "response": json.dumps({"choices": [{"message": pred_message}]}),
                "target_message": target_message,
            }
        ]
        mock_target_message = Mock(**target_message)
        mock_target_message.model_dump.return_value = target_message
        mock_pred_message = Mock(**pred_message)
        mock_pred_message.model_dump.return_value = pred_message
        self.generator.config.client.klass.parse_generation.return_value = [mock_pred_message]

        with patch(
            "axlearn.open_api.openai.OpenAIClient.format_message", return_value=mock_target_message
        ):
            metrics = metric_fn(
                responses=responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                    EvalGeneratorType.GRADER: self.generator,
                },
            )
            self.assertEqual(
                metrics["number_of_func_call_intents_ground_truth"],
                expected_number_of_func_call_intents_ground_truth,
            )
            self.assertEqual(
                metrics["number_of_func_call_intents_pred"],
                expected_number_of_func_call_intents_pred,
            )
            self.assertEqual(metrics["func_intent_recall"], expected_func_intent_recall)
            self.assertEqual(metrics["func_intent_precision"], expected_func_intent_precision)

    @parameterized.parameters(
        # Simple match.
        dict(
            pred_target_pair=[
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
            ],
            expected_accuracy=1.0,
            expected_number_of_examples=1,
            expected_func_name_accuracy=1.0,
            expected_strict_accuracy=1.0,
            expected_lenient_accuracy=1.0,
            expected_bow_accuracy=1.0,
            expected_number_of_expected_tool_calls=1,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_number_of_func_call_intents_pred=1,
            expected_func_intent_recall=1.0,
            expected_func_intent_precision=1.0,
        ),
        # Test function intent, ground truth no tool_calls.
        dict(
            pred_target_pair=[
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
                """Get weather for Boston~""",
            ],
            expected_accuracy=0.0,
            expected_number_of_examples=1,
            expected_func_name_accuracy=0.0,
            expected_strict_accuracy=0.0,
            expected_lenient_accuracy=0.0,
            expected_bow_accuracy=0.0,
            expected_number_of_expected_tool_calls=0,
            expected_number_of_func_call_intents_ground_truth=0,
            expected_number_of_func_call_intents_pred=1,
            expected_func_intent_recall=0.0,
            expected_func_intent_precision=0.0,
        ),
        # Test function intent, pred no tool_calls.
        dict(
            pred_target_pair=[
                """Get weather for Boston~""",
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
            ],
            expected_accuracy=0.0,
            expected_number_of_examples=1,
            expected_func_name_accuracy=0.0,
            expected_strict_accuracy=0.0,
            expected_lenient_accuracy=0.0,
            expected_bow_accuracy=0.0,
            expected_number_of_expected_tool_calls=1,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_number_of_func_call_intents_pred=0,
            expected_func_intent_recall=0.0,
            expected_func_intent_precision=0.0,
        ),
        # Tool call name mis-match.
        dict(
            pred_target_pair=[
                [{"name": "get_weather", "arguments": {"location": "Boston, MA"}}],
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
            ],
            expected_accuracy=0.0,
            expected_number_of_examples=1,
            expected_func_name_accuracy=0.0,
            expected_strict_accuracy=0.0,
            expected_lenient_accuracy=0.0,
            expected_bow_accuracy=0.0,
            expected_number_of_expected_tool_calls=1,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_number_of_func_call_intents_pred=1,
            expected_func_intent_recall=1.0,
            expected_func_intent_precision=1.0,
        ),
        # Tool call BOW.
        dict(
            pred_target_pair=[
                [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}],
                [{"name": "get_current_weather", "arguments": {"location": "Boston"}}],
            ],
            expected_accuracy=0.0,
            expected_number_of_examples=1,
            expected_func_name_accuracy=1.0,
            expected_strict_accuracy=0.0,
            expected_lenient_accuracy=1.0,  # 'ma' is part of _STOP_WORDS and removed.
            expected_bow_accuracy=1.0,
            expected_number_of_expected_tool_calls=1,
            expected_number_of_func_call_intents_ground_truth=1,
            expected_number_of_func_call_intents_pred=1,
            expected_func_intent_recall=1.0,
            expected_func_intent_precision=1.0,
        ),
    )
    def test_all_metrics(
        self,
        pred_target_pair,
        expected_accuracy,
        expected_number_of_examples,
        expected_func_name_accuracy,
        expected_strict_accuracy,
        expected_lenient_accuracy,
        expected_bow_accuracy,
        expected_number_of_expected_tool_calls,
        expected_number_of_func_call_intents_ground_truth,
        expected_number_of_func_call_intents_pred,
        expected_func_intent_recall,
        expected_func_intent_precision,
    ):
        """Tests all metrics."""

        def _make_message(content_tool_call_info: Union[list, str]) -> dict:
            if isinstance(content_tool_call_info, str):
                content = content_tool_call_info
                tool_calls = None
            elif isinstance(content_tool_call_info, list):
                content = ""
                tool_calls = []
                for tool_call_info in content_tool_call_info:
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": tool_call_info,
                        }
                    )
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }

        responses = []
        pred_message = _make_message(pred_target_pair[0])
        target_message = _make_message(pred_target_pair[1])
        responses = [
            {
                "response": json.dumps({"choices": [{"message": pred_message}]}),
                "target_message": target_message,
            }
        ]
        mock_target_message = Mock(**target_message)
        mock_target_message.model_dump.return_value = target_message
        mock_pred_message = Mock(**pred_message)
        mock_pred_message.model_dump.return_value = pred_message
        self.generator.config.client.klass.parse_generation.return_value = [mock_pred_message]

        with patch(
            "axlearn.open_api.openai.OpenAIClient.format_message", return_value=mock_target_message
        ):
            metrics = metric_fn(
                responses=responses,
                generators={
                    EvalGeneratorType.RESPONSE: self.generator,
                },
            )
            expected_metrics = {
                "accuracy": expected_accuracy,
                "number_of_examples": expected_number_of_examples,
                "number_of_parsing_errors": 0,
                "number_of_generation_errors": 0,
                "func_name_accuracy": expected_func_name_accuracy,
                "strict_accuracy": expected_strict_accuracy,
                "lenient_accuracy": expected_lenient_accuracy,
                "bow_accuracy": expected_bow_accuracy,
                "number_of_expected_tool_calls": expected_number_of_expected_tool_calls,
                "number_of_func_call_intents_ground_truth": expected_number_of_func_call_intents_ground_truth,  # pylint: disable=line-too-long
                "number_of_func_call_intents_pred": expected_number_of_func_call_intents_pred,
                "func_intent_recall": expected_func_intent_recall,
                "func_intent_precision": expected_func_intent_precision,
            }

            self.assertEqual(metrics, expected_metrics)
