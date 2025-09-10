# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit test for code_kaggle.py.

TODO(gyin): Add more unit tests for code kaggle core functions.
"""

import json
import unittest
from unittest.mock import Mock, patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()


# pylint: disable=wrong-import-position
from axlearn.open_api.metrics import code_kaggle
from axlearn.open_api.metrics.code_kaggle import BaseClient, _post_process_responses

# pylint: enable=wrong-import-position


class TestPostProcessResponses(unittest.TestCase):
    """Unit tests for _post_process_responses."""

    @patch(f"{code_kaggle.__name__}._post_process_cells")
    def test_post_process_responses(self, mock_post_process_cells):
        responses = [
            {
                "response": json.dumps({"key": "value"}),
                "notebook": {"cells": []},
                "messages": [{"role": "user", "content": "Some user content"}],
            }
        ]
        client_cls = Mock(spec=BaseClient)
        mock_code_response = Mock()
        mock_code_response.content = '{"key": "value"}'
        client_cls.parse_generation.return_value = [mock_code_response]

        mock_post_process_cells.side_effect = lambda x: x

        _post_process_responses(responses, client_cls=client_cls)

        self.assertIn("solutions", responses[0])
        self.assertTrue(client_cls.parse_generation.called)
        self.assertTrue(mock_post_process_cells.called)
        self.assertEqual(len(responses[0]["solutions"]), 1)
        self.assertIn("messages", responses[0]["solutions"][0])
        self.assertIn("notebook", responses[0]["solutions"][0])
        self.assertIn("execution_success", responses[0]["solutions"][0])
        self.assertIn("executed_notebook", responses[0]["solutions"][0])
        self.assertIn("last_cell", responses[0]["solutions"][0])
        self.assertIn("answers", responses[0]["solutions"][0])

    @patch(f"{code_kaggle.__name__}.logging.error")
    def test_post_process_responses_no_notebook(self, mock_logging_error):
        responses = [
            {
                "response": json.dumps({"key": "value"}),
                "messages": [{"role": "user", "content": "Some user content"}],
            }
        ]
        client_cls = Mock(spec=BaseClient)

        _post_process_responses(responses, client_cls=client_cls)

        mock_logging_error.assert_called_with("No notebook in the response.")
        self.assertIn("solutions", responses[0])
        self.assertEqual(responses[0]["solutions"], {})

    @patch(f"{code_kaggle.__name__}.logging.error")
    def test_post_process_responses_no_code_responses(self, mock_logging_error):
        responses = [
            {
                "response": json.dumps({"key": "value"}),
                "notebook": {"cells": []},
                "messages": [{"role": "user", "content": "Some user content"}],
            }
        ]
        client_cls = Mock(spec=BaseClient)
        client_cls.parse_generation.return_value = []

        _post_process_responses(responses, client_cls=client_cls)

        mock_logging_error.assert_called_with("No code responses.")
        self.assertIn("solutions", responses[0])
        self.assertEqual(responses[0]["solutions"], {})
