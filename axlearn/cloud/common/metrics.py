"""Prometheus metrics for Bastion runners."""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest, multiprocess

BASTION_DOWNLOAD_JOB_ALL_SECONDS = Histogram(
    "bastion_download_job_all_seconds",
    "Time in seconds consumed in each scheduling loop to download all the job specs",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),  # 1s to 1hr
)

JOB_TIME_TO_RUNNING_SECONDS = Histogram(
    "bastion_job_time_to_running_seconds",
    "Time in seconds from runner starts a job until until it reaches Running state",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),  # 1s to 1hr
)

JOB_RUN_LATENCY_GAUGE = Gauge(
    "bastion_job_run_latency_seconds",
    "Time in seconds from runner starts a job until until it reaches Running state",
    ["job_name"],
)

JOB_WAIT_FOR_BUILD_SECONDS = Gauge(
    "bastion_job_wait_for_build_seconds",
    "time in seconds until the job is built successfully",
    ["job_name"],
)

JOB_SUSPENDED_DURATION_SECONDS = Histogram(
    "bastion_job_suspended_duration_seconds",
    "Time in seconds the jobset stayed in SUSPENDED state",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),  # 1s to 1hr
)


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves aggregated metrics from multiprocess mode."""

    def do_GET(self):  # pylint: disable=invalid-name
        if self.path == "/metrics":
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            metrics = generate_latest(registry)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(metrics)
        else:
            self.send_error(404)

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        # Suppress default logging
        pass


class MetricsServer:
    """Singleton class to manage Prometheus metrics server.

    Supports multiprocess mode where metrics from parent and child processes
    are aggregated and exposed via a single endpoint.
    """

    _instance = None
    _lock = threading.Lock()
    _server_started = False
    _multiprocess_dir = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def start(cls, port: int = 8000):
        """Start Prometheus metrics HTTP server on the specified port.

        Args:
            port: Port number to start the metrics server on. Defaults to 8000.
            use_multiprocess: If True, use multiprocess mode which aggregates metrics
                            from child processes. Defaults to False for backward compatibility.
        """
        with cls._lock:
            if not cls._server_started:
                try:
                    # Use custom handler for multiprocess aggregation
                    server = HTTPServer(("", port), _MetricsHandler)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    logging.info(
                        "Prometheus metrics server (multiprocess mode) started on port %s", port
                    )
                    cls._server_started = True
                except OSError as e:
                    # Port might already be in use
                    logging.warning("Failed to start metrics server on port %s: %s", port, e)

    @classmethod
    def is_started(cls) -> bool:
        """Check if the metrics server has been started."""
        return cls._server_started

    @classmethod
    def get_multiprocess_dir(cls) -> str:
        """Get the multiprocess directory path if configured."""
        return cls._multiprocess_dir


def record_job_time_to_running(time_seconds: float):
    """Record the time it took for a job to reach RUNNING state.

    Args:
        job_name: Name of the job
        time_seconds: Time in seconds from first seen until RUNNING state
    """
    JOB_TIME_TO_RUNNING_SECONDS.observe(time_seconds)


def record_job_run_latency(job_name: str, time_seconds: float):
    JOB_RUN_LATENCY_GAUGE.labels(job_name).set(time_seconds)


def record_job_wait_for_build(job_name: str, time_seconds: float):
    """Record the time it took to wait for the cloud build to finish

    This does not necessarily mean the whole cloud build time for a job,
    as the build starts when the client submits the job. This only measures
    how long the runner waits for the build to complete before launching
    the job.
    """
    JOB_WAIT_FOR_BUILD_SECONDS.labels(job_name).set(time_seconds)


def record_job_suspended_duration(time_seconds: float):
    """Record the time a job spent in SUSPENDED state.

    Args:
        job_name: Name of the job
        time_seconds: Time in seconds the job was in SUSPENDED state
    """
    JOB_SUSPENDED_DURATION_SECONDS.observe(time_seconds)


def cleanup(pid=None):
    if pid:
        logging.info("Cleaning up prometheus metrics DB file for %s", pid)
        multiprocess.mark_process_dead(pid)
