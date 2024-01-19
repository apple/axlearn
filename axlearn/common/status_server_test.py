# Copyright © 2024 Apple Inc.

"""Tests for HTTP Status server."""

import socket
import sys
import time
import urllib.request

import pytest

from axlearn.common import status_server
from axlearn.common.test_utils import TestCase


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Finds a free port and then binds to it.
        return s.getsockname()[1]


class StatusServerTest(TestCase):
    """Tests for Status Server."""

    def test_server(self):
        port = get_free_port()
        server = status_server.StatusHTTPServer(port)
        server.start()
        time.sleep(1)  # Let server start

        url = f"http://localhost:{port}"
        with urllib.request.urlopen(url) as response:
            data = response.read()
            content = data.decode("utf-8")

        self.assertIn("MainThread", content)
        server.stop()

    def test_exit_when_main_crashes(self):
        with pytest.raises(SystemExit) as exit_info:
            port = get_free_port()
            server = status_server.StatusHTTPServer(port)
            server.start()
            time.sleep(1)  # Let server start.
            sys.exit(-1)

        self.assertEqual(exit_info.type, SystemExit)
        self.assertEqual(exit_info.value.code, -1)
