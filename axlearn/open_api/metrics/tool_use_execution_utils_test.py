# Copyright © 2024 Apple Inc.
"""Unit test for tool_use_execution.py."""

from absl.testing import parameterized

from axlearn.open_api.metrics.tool_use_execution_utils import check_arguments


class TestToolUseExecutionUtils(parameterized.TestCase):
    """Unit tests for tool_use_execution_utils."""

    @parameterized.parameters(
        dict(pred={}, target={}),
        dict(pred={"a": "foo"}, target={"a": "foo"}),
        dict(pred={"a": "foo", "b": "bar"}, target={"b": "bar", "a": "foo"}),
    )
    def test_all_positive_matches(self, pred, target):
        self.assertTrue(check_arguments(pred, target, True, True))
        self.assertTrue(check_arguments(pred, target, True, False))
        self.assertTrue(check_arguments(pred, target, False, True))
        self.assertTrue(check_arguments(pred, target, False, False))

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
        self.assertEqual(check_arguments(pred, target, True, True), lenient_bow)
        self.assertEqual(check_arguments(pred, target, True, False), lenient)
        self.assertEqual(check_arguments(pred, target, False, False), strict)

    @parameterized.parameters(
        dict(pred={"a": "foo"}, target={}),
        dict(pred={}, target={"a": "foo"}),
        dict(pred={"a": "foo"}, target={"b": "foo"}),
        dict(pred={"a": "foo"}, target={"b": "bar"}),
        dict(pred={"a": "foo", "b": "bar"}, target={"a": "foo"}),
        dict(pred={"a": "foo"}, target={"a": "foo", "b": "bar"}),
    )
    def test_all_negative_matches(self, pred, target):
        self.assertFalse(check_arguments(pred, target, True, True))
        self.assertFalse(check_arguments(pred, target, True, False))
        self.assertFalse(check_arguments(pred, target, False, True))
        self.assertFalse(check_arguments(pred, target, False, False))
