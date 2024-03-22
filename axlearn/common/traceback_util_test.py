# Copyright © 2023 Apple Inc.

"""Tests for traceback_util.py."""
# pylint: disable=protected-access,used-before-assignment,broad-exception-caught,broad-exception-raised

import unittest.mock

from absl.testing import absltest

from axlearn.common import traceback_util
from axlearn.common.test_utils import TestCase
from axlearn.common.traceback_util import _walk_annotated_tb, annotate_stack, no_stack_summary


class TracebackUtilTest(TestCase):
    def test_annotate_stack(self):
        @annotate_stack(test=1)
        @annotate_stack(test=42)
        @annotate_stack(test2="asdf")
        def f():
            raise Exception()

        annotate_stack(test3=7)(f)  # Should have no effect.

        try:
            f()
        except Exception as e:
            aux = [
                aux
                for frame, aux in _walk_annotated_tb(e.__traceback__)
                if frame.f_code.co_name == "f"
            ][0]
        self.assertEqual(aux, dict(test=42, test2="asdf"))

    def test_walk_annotated_tb(self):
        def f():
            g()

        @annotate_stack(test=1)
        @annotate_stack(test2="asdf")
        def g():
            h()

        def h():
            raise Exception()

        annotate_stack(test3=7)(f)  # Should have no effect.

        try:
            f()
        except Exception as e:
            results = [
                (frame.f_code.co_name, aux) for frame, aux in _walk_annotated_tb(e.__traceback__)
            ]
        self.assertSequenceEqual(
            results,
            [
                ("test_walk_annotated_tb", {}),
                ("f", {}),
                ("g", dict(test=1, test2="asdf")),
                ("h", {}),
            ],
        )

    def test_in_context_exception(self):
        @traceback_util.wrap
        def f():
            g()

        @no_stack_summary
        def g():
            h()

        def h():
            raise Exception()

        annotate_stack(test3=7)(f)  # Should have no effect.

        try:
            f()
        except Exception as e:
            self.assertTrue(hasattr(e, "_stack_summary"))
            in_context_exception = e._stack_summary  # pytype: disable=attribute-error

        summary = in_context_exception._format_summary()
        self.assertTrue("in f\n" in summary)
        self.assertTrue("in g\n" not in summary)
        self.assertTrue("in h\n" in summary)
        self.assertTrue("in_context_exception_wrapper" not in summary)

    def test_excepthook_patch(self):
        mock = unittest.mock.MagicMock()
        e = Exception("my_exception")
        args = (type(e), e, e.__traceback__, mock)
        traceback_util._excepthook(*args)
        mock.assert_called_with(*args[:-1])

        e._stack_summary = ValueError("another_exception")
        traceback_util._excepthook(*args)
        mock.assert_called_with(
            type(e._stack_summary), e._stack_summary, e._stack_summary.__traceback__
        )


if __name__ == "__main__":
    absltest.main()
