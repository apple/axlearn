# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for generator.py."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from absl import flags

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# isort: off
# pylint: disable=wrong-import-position
from axlearn.open_api import common
from axlearn.open_api.common import Evaluator, Generator
from axlearn.open_api.evaluator import evaluate_from_file

# pylint: enable=wrong-import-position
# isort: one


class TestEvaluateFromFile(unittest.IsolatedAsyncioTestCase):
    """Unit test for evaluate_from_file."""

    def setUp(self):
        self.mock_responses = [
            {"response": "response1"},
            {"response": "response2"},
            {"response": "response3"},
        ]

        # Create a temporary file and write the mock responses to it.
        # pylint: disable-next=consider-using-with
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        json.dump(self.mock_responses, self.temp_file)
        # Ensure data is written to file.
        self.temp_file.flush()
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        # Close and remove the temporary file
        self.temp_file.close()
        os.remove(self.temp_file_path)

    @patch(
        f"{common.__name__}.Evaluator.evaluate",
        new_callable=MagicMock,
    )
    async def test_evaluate_from_file(self, mock_evaluate):
        fv = flags.FlagValues()
        Generator.define_flags(fv)
        Evaluator.define_flags(fv)
        fv.set_default("model", "test")
        fv.set_default("check_vllm_readiness", False)
        fv.set_default("metric_name", "tool_use_plan")
        fv.set_default("input_file", self.temp_file_path)
        fv.mark_as_parsed()

        # Call the method under test.
        evaluate_from_file(fv=fv)
        mock_evaluate.assert_called_once()

    @patch(
        f"{common.__name__}.Evaluator.evaluate",
        new_callable=MagicMock,
    )
    # pylint: disable-next=unused-argument
    async def test_evaluate_from_file_no_metric(self, mock_evaluate):
        fv = flags.FlagValues()
        Generator.define_flags(fv)
        Evaluator.define_flags(fv)
        fv.set_default("model", "test")
        fv.set_default("check_vllm_readiness", False)
        fv.set_default("input_file", self.temp_file_path)
        fv.mark_as_parsed()

        # Test that ValueError is raised.
        with self.assertRaises(ValueError):
            # Call the method under test.
            evaluate_from_file(fv=fv)
