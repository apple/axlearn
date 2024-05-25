# Copyright © 2024 Apple Inc.

"""A script to compute goodput and upload to Cloud Monitoring.

This can be run as a daemon for each training job for which `GoodputRecorder` is configured.

Example:

    python3 -m axlearn.experiments.calculate_goodput --job_name=my-test-job

"""

import time
from datetime import datetime

from absl import app, flags, logging
from googleapiclient import discovery, errors
from ml_goodput_measurement import goodput

from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import get_credentials

FLAGS = flags.FLAGS
_METRIC_NAME = "goodput"


def _private_flags():
    flags.DEFINE_string("project", None, "GCP project.")
    flags.DEFINE_string("zone", None, "GCP zone.")
    flags.DEFINE_string("job_name", None, "Name of job.", required=True)


def _monitoring_resource() -> discovery.Resource:
    return discovery.build(
        "monitoring", "v3", credentials=get_credentials(), cache_discovery=False
    ).projects()


def create_custom_metric(*, project: str, metric_name: str):
    """Creates a custom metric if it doesn't already exist."""
    resource = _monitoring_resource().metricDescriptors()
    metric_id = f"custom.googleapis.com/{metric_name}"

    try:
        metric = resource.get(name=f"projects/{project}/metricDescriptors/{metric_id}").execute()
    except errors.HttpError as e:
        if e.status_code != 404:
            raise
        metric = None

    if metric is None:
        logging.info("Metric %s does not exist, creating it...", metric_name)
        metric = resource.create(
            name=f"projects/{project}",
            body={
                "name": metric_name,
                "type": metric_id,
                "description": f"{metric_name.capitalize()} metric.",
                "displayName": metric_name.capitalize(),
                "metricKind": "GAUGE",
                "valueType": "DOUBLE",
            },
        ).execute()

    logging.info("Using %s metric: %s", metric_name, metric)


def write_time_series_metric(
    *,
    project: str,
    metric_name: str,
    value: float,
    resource_labels: dict,
    metric_labels: dict,
    end_time: float,
):
    """Writes a custom time-series metric value."""
    resource = _monitoring_resource().timeSeries()
    utc_end_time = datetime.utcfromtimestamp(end_time)
    resource.create(
        name=f"projects/{project}",
        body={
            "timeSeries": [
                {
                    "metric": {
                        "type": f"custom.googleapis.com/{metric_name}",
                        "labels": {
                            metric_name: str(value),
                            "event_time": utc_end_time.strftime("%d %b %Y %H:%M:%S UTC"),
                            **metric_labels,
                        },
                    },
                    "resource": {
                        "labels": {
                            # The namespace/node_id labels are mandatory.
                            "namespace": "namespace",
                            "node_id": "node_id",
                            **resource_labels,
                        },
                        "type": "generic_node",
                    },
                    "points": [
                        {
                            "interval": {"endTime": utc_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")},
                            "value": {"doubleValue": value},
                        },
                    ],
                }
            ]
        },
    ).execute()


def main(_):
    project, zone = gcp_settings("project"), gcp_settings("zone")
    create_custom_metric(project=project, metric_name=_METRIC_NAME)

    goodput_calculator = goodput.GoodputCalculator(
        job_name=FLAGS.job_name,
        logger_name=f"goodput_logger_{FLAGS.job_name}",
    )

    start_time = time.time()
    current_goodput = goodput_calculator.get_job_goodput()
    end_time = time.time()

    print(f"Job goodput: {current_goodput:.4f}%")
    print(f"Fetch time: {end_time - start_time:.2f} seconds")

    # TODO(markblee): Change back to polling in an interval when updating to the latest
    # ml_goodput_measurement version.
    write_time_series_metric(
        project=project,
        metric_name=_METRIC_NAME,
        value=current_goodput,
        resource_labels={"location": zone},
        metric_labels={"job_name": FLAGS.job_name},
        end_time=end_time,
    )


if __name__ == "__main__":
    _private_flags()
    app.run(main)
