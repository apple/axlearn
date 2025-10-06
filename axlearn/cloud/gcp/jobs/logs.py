# Copyright © 2024 Apple Inc.

"""View logs for a job via Cloud Logging.

Example:

    # At the moment, name is assumed to be a job submitted via GKE.
    axlearn gcp logs --name=...

"""

import urllib.parse

from absl import app, flags, logging

from axlearn.cloud.gcp.config import default_project
from axlearn.cloud.gcp.utils import catch_auth

FLAGS = flags.FLAGS


def _private_flags(flag_values: flags.FlagValues = FLAGS):
    flags.DEFINE_string("name", None, "Job name.", flag_values=flag_values)
    flags.DEFINE_alias("job_name", "name", flag_values=flag_values)
    flags.DEFINE_integer("worker", 0, "Worker ID.", flag_values=flag_values)
    flags.DEFINE_integer("replica", 0, "Replica ID.", flag_values=flag_values)
    flags.DEFINE_integer(
        "num_lines", 1000, "Number of log lines.", short_name="n", flag_values=flag_values
    )
    flags.DEFINE_string(
        "start_time",
        None,
        "Start time of the job query, in the format of yyyy-mm-dd or yyyy-mm-ddTH:M:SZ.",
        flag_values=flag_values,
    )
    flags.DEFINE_string(
        "end_time",
        None,
        "End time of the job query, in the format of yyyy-mm-dd or yyyy-mm-ddTH:M:SZ.",
        flag_values=flag_values,
    )
    flags.DEFINE_boolean(
        "earliest",
        False,
        "Print the earliest logs between start_time and end_time. Default is to print the latest.",
        flag_values=flag_values,
    )
    flags.register_validator(
        "num_lines",
        lambda value: value > 0,
        message="--num_lines must be greater than 0",
        flag_values=flag_values,
    )


# Kept as a standalone fn for easier testing/mocking.
def _logging_client():
    try:
        # pylint: disable-next=import-error,import-outside-toplevel
        from google.cloud import logging as cloud_logging  # pytype: disable=import-error
    except ImportError as e:
        raise app.UsageError("Required to view logs: pip install google-cloud-logging") from e

    return cloud_logging.Client(project=default_project())


# TODO(markblee): Support non-GKE jobs.
@catch_auth
def main(_, *, flag_values: flags.FlagValues = FLAGS):
    client = _logging_client()

    # pylint: disable-next=import-error,import-outside-toplevel
    from google.cloud import logging as cloud_logging  # pytype: disable=import-error

    # Filter results by property == value.
    query_eq_filters = {
        "resource.labels.namespace_name": "default",
        "resource.labels.container_name": flag_values.name,
        "labels.k8s-pod/batch_kubernetes_io/job-completion-index": flag_values.worker,
        "labels.k8s-pod/jobset_sigs_k8s_io/job-index": flag_values.replica,
    }
    filters = [f"{k}={v}" for k, v in query_eq_filters.items()]

    # Using start and end time filters can significantly reduce query time for older jobs, and avoid
    # timeouts.
    if flag_values.start_time:
        filters.append(f'timestamp>="{flag_values.start_time}"')
    if flag_values.end_time:
        filters.append(f'timestamp<="{flag_values.end_time}"')

    entries = client.list_entries(
        filter_=" AND ".join(filters),
        # Use descending so we read out the most recent logs.
        order_by=cloud_logging.ASCENDING if flag_values.earliest else cloud_logging.DESCENDING,
        max_results=flag_values.num_lines,
    )
    entries = list(entries)
    if not entries:
        logging.info(
            "No logs found. Consider specifying a time range with "
            "--start_time and --end_time= to narrow your query with "
            "yyyy-mm-dd or yyyy-mm-ddTH:M:SZ."
        )
        return

    # Print the most recent logs in sorted order (i.e., reverse timestamp order).
    for entry in reversed(entries):
        message = entry.payload
        if isinstance(message, dict):
            message = message.get("message", "")
        print(entry.timestamp, message)

    start_time = entries[-1].timestamp
    end_time = entries[0].timestamp

    query_args = "".join(
        urllib.parse.urlencode([(k, f"{v}\n")]) for k, v in query_eq_filters.items()
    )
    # Use ISO format.
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://console.cloud.google.com/logs/query;query={query_args};startTime={start_time}"
    if end_time:
        end_time = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{url};endTime={end_time}"

    print(f"\nCloud logging url: {url}")


if __name__ == "__main__":
    _private_flags()
    app.run(main)
