from ml_goodput_measurement import goodput
import time
from datetime import datetime

import os
import time

from google.api import metric_pb2
from google.cloud import compute_v1
from google.cloud import monitoring_v3

# TODO: fill out your project ID
PROJECT_ID = "PROJECT ID"
ZONE = "us-central2-b"

# Recommend to run this script on a separate machine than your ML training machine
# Demo used the newer version of the package: ml_goodput_measurement==0.0.8

def main():
    # specify the run name you want to query GoodPut from
    run_name='test'
    goodput_logger_name = f'goodput_{run_name}'
    metric_name = f'goodput_metric_{run_name}'
    project_name = f"projects/{PROJECT_ID}"

    # TODO: if this doesn't already exist
    set_up_metric(metric_name,project_name)

    # Create a GoodPut Calculator object
    goodput_calculator = goodput.GoodputCalculator(job_name=run_name, logger_name=goodput_logger_name)

    # Pull GoodPut metric every 30 seconds and write it to custom metric
    while True:
        before_gp = datetime.now()
        current_goodput = goodput_calculator.get_job_goodput()
        after_gp = datetime.now()
        time_diff = after_gp - before_gp
        print(f"=========> Current job goodput: {current_goodput:.4f}%")
        print(f'Fetch time: {time_diff.total_seconds():.2f} seconds')

        write_time_series_step(metric_name, run_name, current_goodput)
        time.sleep(30)

def set_up_metric(metric_name,project_name):
    # Create a custom metric for Goodput
    client = get_metrics_service_client()

    descriptor = metric_pb2.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/" + metric_name
    descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
    descriptor.description = "Goodput of the job."

    descriptor = client.create_metric_descriptor(
        name=project_name, metric_descriptor=descriptor
    )
    print("Created custom metric: {}.".format(descriptor.name))


def get_metrics_service_client():
  """Returns Cloud Monitoring API client."""
  return monitoring_v3.MetricServiceClient()


def get_query_service_client():
  """Returns Cloud Monitoring Query Service client."""
  return monitoring_v3.QueryServiceClient()


def get_compute_instances_client():
  """Returns Cloud Compute Instances client."""
  return compute_v1.InstancesClient()


def get_instance_id():
  client = get_compute_instances_client()
  instance_name = os.uname().nodename
  instance = client.get(project=PROJECT_ID, zone=ZONE, instance=instance_name)
  return instance.id


def write_time_series_step(metric_name, job_name, goodput_value):
  """Emits a Goodput data point when a query is made.

  Args:
    metric_name: name of the metric
    job_name: training step
    goodput_value: Current Goodput of job
  """
  client = get_metrics_service_client()
  project_name = f"projects/{PROJECT_ID}"

  seconds_since_epoch_utc = time.time()
  nanos_since_epoch_utc = int(
      (seconds_since_epoch_utc - int(seconds_since_epoch_utc)) * 10**9
  )
  interval = monitoring_v3.types.TimeInterval({
      "end_time": {
          "seconds": int(seconds_since_epoch_utc),
          "nanos": nanos_since_epoch_utc,
      }
  })

  event_time = time.strftime(
      "%d %b %Y %H:%M:%S UTC", time.gmtime(seconds_since_epoch_utc)
  )


  series = monitoring_v3.types.TimeSeries()
  series.metric.type = "custom.googleapis.com/" + metric_name
  series.resource.type = "generic_node"
  series.resource.labels["location"] = "us-central2-b"
  series.resource.labels["namespace"] = "namespace"
  series.resource.labels["node_id"] = "node_id"
  series.metric.labels["goodput"] = str(goodput_value)
  series.metric.labels["job_name"] = job_name
  series.metric.labels["event_time"] = event_time
  series.points = [
      monitoring_v3.types.Point(
          interval=interval,
          value=monitoring_v3.types.TypedValue(double_value=goodput_value),
      )
  ]

  client.create_time_series(name=project_name, time_series=[series])


if __name__ == "__main__":
    main()
