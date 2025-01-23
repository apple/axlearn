# Copyright © 2024 Apple Inc.

"""Tests launch utils."""

import contextlib
from typing import Optional
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.common.launch import _init_context
from axlearn.common.test_utils import TestCase


class TestInitContext(TestCase):
    """Tests _init_context."""

    @parameterized.parameters(
        dict(value=["my.module.path"], expect={"my.module.path": None}),
        dict(
            value=["my.module.path:k1=v1,k2=v2"],
            expect={"my.module.path": "k1=v1,k2=v2"},
        ),
        dict(
            value=["my.module.path:k1:v1"],
            expect={"my.module.path": "k1:v1"},
        ),
        dict(
            value=["my.module.path:k1:v1", "my.other.module:k2:v2,k3:v3"],
            expect={
                "my.module.path": "k1:v1",
                "my.other.module": "k2:v2,k3:v3",
            },
        ),
    )
    def test_init_context(self, value, expect: dict[str, Optional[str]]):
        fv = flags.FlagValues()
        flags.DEFINE_multi_string("init_module", value, "", flag_values=fv)
        fv.mark_as_parsed()

        with mock.patch("importlib.import_module") as mock_import:
            with _init_context(fv):
                for i, k in enumerate(expect):
                    self.assertEqual(k, mock_import.call_args_list[i][0][0])

        side_effect = []
        actual_specs = []
        for _ in range(len(value)):

            @contextlib.contextmanager
            def mock_setup(actual):
                actual_specs.append(actual)
                yield

            mock_module = mock.Mock(**{"setup.side_effect": mock_setup})
            side_effect.append(mock_module)

        with mock.patch("importlib.import_module", side_effect=side_effect):
            with _init_context(fv):
                self.assertEqual(list(expect.values()), actual_specs)
