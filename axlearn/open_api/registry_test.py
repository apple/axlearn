# Copyright © 2024 Apple Inc.

# pylint: disable=protected-access
"""Unit tests for common.py."""
import unittest
from unittest.mock import MagicMock, patch

from axlearn.open_api.mock_utils import mock_openai_package

mock_openai_package()

# pylint: disable=wrong-import-position

from axlearn.open_api.common import BaseClient
from axlearn.open_api.registry import (
    _OPEN_API_CLIENTS_CLASS,
    _OPEN_API_CLIENTS_MODULE_CLASS,
    ClientRegistry,
)

# pylint: enable=wrong-import-position

_module_root = "axlearn"


class MockClient(BaseClient):
    """A mock of open api client."""

    pass


class TestClientRegistry(unittest.TestCase):
    """Unit tests for ClientRegistry."""

    def setUp(self):
        # Clear the client registry before each test
        _OPEN_API_CLIENTS_CLASS.clear()

    @patch(f"{_module_root}.open_api.registry.importlib.import_module")
    def test_load_client_cls_existing_client(self, mock_import_module):
        # Mock the import and getattr functions
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        mock_module.OpenAIClient = MockClient

        _OPEN_API_CLIENTS_MODULE_CLASS["mockclient"] = ("mock_module", "OpenAIClient")

        client_cls = ClientRegistry.load_client_cls("mockclient")
        self.assertEqual(client_cls, MockClient)

    def test_load_client_cls_registered_client(self):
        _OPEN_API_CLIENTS_CLASS["mockclient"] = MockClient

        client_cls = ClientRegistry.load_client_cls("mockclient")
        self.assertEqual(client_cls, MockClient)

    def test_load_client_cls_non_existing_client(self):
        client_cls = ClientRegistry.load_client_cls("nonexistent")
        self.assertIsNone(client_cls)

    def test_get_supported_clients(self):
        _OPEN_API_CLIENTS_MODULE_CLASS["client1"] = ("module1", "Client1")
        _OPEN_API_CLIENTS_CLASS["client2"] = MockClient

        supported_clients = ClientRegistry.get_supported_clients()
        self.assertIn("client1", supported_clients)
        self.assertIn("client2", supported_clients)

    def test_register_new_client(self):
        ClientRegistry.register("mockclient", MockClient)

        self.assertIn("mockclient", _OPEN_API_CLIENTS_CLASS)
        self.assertEqual(_OPEN_API_CLIENTS_CLASS["mockclient"], MockClient)

    @patch(f"{_module_root}.open_api.registry.logging.warning")
    def test_register_existing_client(self, mock_warning):
        _OPEN_API_CLIENTS_CLASS["mockclient"] = BaseClient

        ClientRegistry.register("mockclient", MockClient)

        mock_warning.assert_called_once()
        self.assertEqual(_OPEN_API_CLIENTS_CLASS["mockclient"], MockClient)
