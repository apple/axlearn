# Copyright © 2024 Apple Inc.
"""Unit test for tool_use_execution.py."""

from absl.testing import parameterized

from axlearn.open_api.metrics.tool_use_execution_utils import (
    ArgumentMatchType,
    _string_lenient_transform,
    check_arguments,
)


class TestToolUseExecutionUtils(parameterized.TestCase):
    """Unit tests for tool_use_execution_utils."""

    @parameterized.parameters(
        dict(pred={}, target={}),
        dict(pred={"a": "foo"}, target={"a": "foo"}),
        dict(pred={"a": "foo", "b": "bar"}, target={"b": "bar", "a": "foo"}),
    )
    def test_all_positive_matches(self, pred, target):
        self.assertTrue(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.STRICT,
            )
        )
        self.assertTrue(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT,
            )
        )
        self.assertTrue(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT_BAG_OF_WORD,
            )
        )

    @parameterized.parameters(
        dict(
            pred={"a": "you foo"}, target={"a": "foo"}, lenient_bow=True, lenient=True, strict=False
        ),
        dict(
            pred={"a": "you foo"}, target={"a": "foo"}, lenient_bow=True, lenient=True, strict=False
        ),
        dict(
            pred={"a": "you bar foo"},
            target={"a": "foo"},
            lenient_bow=True,
            lenient=False,
            strict=False,
        ),
    )
    def test_all_matches(self, pred, target, lenient_bow, lenient, strict):
        self.assertEqual(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT_BAG_OF_WORD,
            ),
            lenient_bow,
        )
        self.assertEqual(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT,
            ),
            lenient,
        )
        self.assertEqual(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.STRICT,
            ),
            strict,
        )

    @parameterized.parameters(
        dict(pred={"a": "foo"}, target={}),
        dict(pred={}, target={"a": "foo"}),
        dict(pred={"a": "foo"}, target={"b": "foo"}),
        dict(pred={"a": "foo"}, target={"b": "bar"}),
        dict(pred={"a": "foo", "b": "bar"}, target={"a": "foo"}),
        dict(pred={"a": "foo"}, target={"a": "foo", "b": "bar"}),
    )
    def test_all_negative_matches(self, pred, target):
        self.assertFalse(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.STRICT,
            )
        )
        self.assertFalse(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT,
            )
        )
        self.assertFalse(
            check_arguments(
                pred_args=pred,
                target_args=target,
                match_type=ArgumentMatchType.LENIENT_BAG_OF_WORD,
            )
        )

    @parameterized.parameters(
        dict(words="", result=""),
        dict(words="you", result=""),
        dict(words="you you", result=""),
        dict(words="you foo", result="foo"),
        dict(words="you foo you", result="foo"),
        dict(words="you foo you you", result="foo"),
        dict(words="you foo bar you", result="foo bar"),
        dict(words="you foo bar you you", result="foo bar"),
        dict(words="you you foo", result="foo"),
        dict(words="you you foo you", result="foo"),
        dict(words="you you foo you you", result="foo"),
        dict(words="you you foo bar you", result="foo bar"),
        dict(words="you you foo bar you you", result="foo bar"),
        dict(words="you you foo you bar you", result="foo you bar"),
        dict(words="you you foo you bar you you", result="foo you bar"),
        dict(words="you foo you bar you", result="foo you bar"),
        dict(words="you foo you bar you you", result="foo you bar"),
        dict(words="foo you bar you", result="foo you bar"),
        dict(words="foo you bar you you", result="foo you bar"),
        dict(words="foo you bar", result="foo you bar"),
        dict(words="foo you bar you", result="foo you bar"),
    )
    def test_string_lenient_transform(self, words, result):
        self.assertEqual(_string_lenient_transform(words), result)
