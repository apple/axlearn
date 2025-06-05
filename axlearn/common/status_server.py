# Copyright © 2024 Apple Inc.

"""HTTP Status server."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from absl import logging

from axlearn.common.utils import thread_stack_traces


def _status_html():
    """Generates status page HTML."""
    stacks = thread_stack_traces()
    content = ""
    for stack_lines in stacks:
        stack = "\n".join(stack_lines)
        content += f"<pre>{stack}</pre>"
    html = (
        """<html><head><title>python stack traces</title></head>"""
        f"""<body>{content}</body></html>"""
    )
    return html


class RequestHandler(BaseHTTPRequestHandler):
    """Simple HTML Request Handler."""

    def do_GET(self):  # pylint: disable=invalid-name
        """Handles a GET request."""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(_status_html().encode("utf-8"))


class StatusHTTPServer:
    """Status HTTP Server."""

    def __init__(self, port: int):
        self.port = port
        self.httpd = None

    def _start_blocking(self):
        """Starts status server (blocking)."""
        server_address = ("", self.port)
        self.httpd = HTTPServer(server_address, RequestHandler)
        logging.info("Starting status http server on port %s ...", self.port)
        self.httpd.serve_forever()

    def start(self):
        """Starts status server."""
        self.thread = threading.Thread(target=self._start_blocking)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stops status server."""
        if self.httpd:
            self.httpd.shutdown()
            self.thread.join()
