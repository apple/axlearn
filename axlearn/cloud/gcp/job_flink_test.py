# Copyright © 2025 Apple Inc.

"""Unit tests of job_flink.py."""

import json
import logging
from typing import Optional

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, job_flink, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.test_utils import TestCase

# These two json text-literals are auto generated on test failure as a warning log.
# Please copy the updated version here and verify if the update is expected.
expected_flink_deployment_json = """
{
  "apiVersion": "flink.apache.org/v1beta1",
  "kind": "FlinkDeployment",
  "metadata": {
    "namespace": "default",
    "name": "fake-name-flink-cluster"
  },
  "spec": {
    "image": "flink:1.18",
    "flinkVersion": "v1_18",
    "serviceAccount": "sa",
    "mode": "standalone",
    "podTemplate": {
      "spec": {
        "initContainers": [
          {
            "name": "output-uploader",
            "image": "google/cloud-sdk:alpine",
            "restartPolicy": "Always",
            "command": [
              "/bin/sh",
              "-c"
            ],
            "args": [
              "while true; do gsutil -m rsync -r /opt/flink/log fake-output-dir/output/$HOSTNAME/; sleep 60; done"
            ],
            "resources": {
              "requests": {
                "cpu": "100m",
                "memory": "128Mi"
              },
              "limits": {
                "cpu": "500m",
                "memory": "256Mi"
              }
            },
            "volumeMounts": [
              {
                "mountPath": "/opt/flink/log",
                "name": "flink-logs"
              }
            ]
          }
        ],
        "containers": [
          {
            "name": "flink-main-container",
            "volumeMounts": [
              {
                "mountPath": "/opt/flink/log",
                "name": "flink-logs"
              }
            ]
          }
        ],
        "volumes": [
          {
            "name": "flink-logs",
            "emptyDir": {}
          }
        ]
      }
    },
    "flinkConfiguration": {
      "taskmanager.numberOfTaskSlots": "4",
      "taskmanager.memory.task.off-heap.size": "16g",
      "taskmanager.network.bind-host": "0.0.0.0",
      "rest.address": "0.0.0.0",
      "execution.checkpointing.interval": "10m",
      "execution.checkpointing.mode": "EXACTLY_ONCE",
      "restart-strategy.type": "fixed-delay",
      "restart-strategy.fixed-delay.attempts": "3",
      "restart-strategy.fixed-delay.delay": "10m"
    },
    "jobManager": {
      "resource": {
        "memory": "2g",
        "cpu": 1
      }
    },
    "taskManager": {
      "replicas": 2,
      "resource": {
        "cpu": 83,
        "memory": "179Gi"
      },
      "podTemplate": {
        "spec": {
          "nodeSelector": {
            "pre-provisioner-id": "fake-name",
            "cloud.google.com/gke-accelerator-count": "4",
            "cloud.google.com/gke-tpu-accelerator": "tpu-v5p-slice",
            "cloud.google.com/gke-tpu-topology": "2x2x1",
            "cloud.google.com/gke-location-hint": "fake-location-hint"
          },
          "tolerations": [
            {
              "key": "google.com/tpu",
              "operator": "Equal",
              "value": "present",
              "effect": "NoSchedule"
            }
          ],
          "initContainers": [
            {
              "name": "output-uploader",
              "image": "google/cloud-sdk:alpine",
              "restartPolicy": "Always",
              "command": [
                "/bin/sh",
                "-c"
              ],
              "args": [
                "while true; do gsutil -m rsync -r /opt/flink/log fake-output-dir/output/$HOSTNAME/; sleep 60; done"
              ],
              "resources": {
                "requests": {
                  "cpu": "100m",
                  "memory": "128Mi"
                },
                "limits": {
                  "cpu": "500m",
                  "memory": "256Mi"
                }
              },
              "volumeMounts": [
                {
                  "mountPath": "/opt/flink/log",
                  "name": "flink-logs"
                }
              ]
            }
          ],
          "containers": [
            {
              "name": "python-harness",
              "volumeMounts": [
                {
                  "mountPath": "/opt/flink/log",
                  "name": "flink-logs"
                }
              ],
              "image": "settings-repo/test-image:fake-name",
              "args": [
                "-worker_pool"
              ],
              "env": [
                {
                  "name": "BEAM_EXTERNAL_HOST",
                  "value": "0.0.0.0"
                },
                {
                  "name": "JAX_PLATFORMS",
                  "value": "tpu"
                },
                {
                  "name": "NODE_IP",
                  "valueFrom": {
                    "fieldRef": {
                      "apiVersion": "v1",
                      "fieldPath": "status.hostIP"
                    }
                  }
                },
                {
                  "name": "NODE_NAME",
                  "valueFrom": {
                    "fieldRef": {
                      "apiVersion": "v1",
                      "fieldPath": "spec.nodeName"
                    }
                  }
                },
                {
                  "name": "TPU_WORKER_HOSTNAMES",
                  "value": "localhost"
                },
                {
                  "name": "TPU_WORKER_ID",
                  "value": "0"
                },
                {
                  "name": "GCS_RESOLVE_REFRESH_SECS",
                  "value": "600"
                },
                {
                  "name": "TPU_TYPE",
                  "value": "v5p-16"
                },
                {
                  "name": "NUM_TPU_SLICES",
                  "value": "1"
                },
                {
                  "name": "XLA_FLAGS",
                  "value": "--xla_dump_to=/output/fake-name/xla"
                },
                {
                  "name": "TF_CPP_MIN_LOG_LEVEL",
                  "value": "0"
                },
                {
                  "name": "TPU_STDERR_LOG_LEVEL",
                  "value": "0"
                },
                {
                  "name": "TPU_MIN_LOG_LEVEL",
                  "value": "0"
                },
                {
                  "name": "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS",
                  "value": "60"
                },
                {
                  "name": "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES",
                  "value": "256"
                },
                {
                  "name": "LD_PRELOAD",
                  "value": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
                }
              ],
              "resources": {
                "limits": {
                  "google.com/tpu": 4,
                  "memory": "448Gi"
                },
                "requests": {
                  "memory": "179Gi"
                }
              },
              "ports": [
                {
                  "containerPort": 50000,
                  "name": "harness-port",
                  "protocol": "TCP"
                }
              ]
            },
            {
              "name": "flink-main-container",
              "volumeMounts": [
                {
                  "mountPath": "/opt/flink/log",
                  "name": "flink-logs"
                }
              ]
            }
          ],
          "volumes": [
            {
              "name": "flink-logs",
              "emptyDir": {}
            }
          ]
        }
      }
    }
  }
}
"""

expected_jobsubmission_json = """
{
  "apiVersion": "batch/v1",
  "kind": "Job",
  "metadata": {
    "name": "fake-name"
  },
  "spec": {
    "backoffLimit": 0,
    "template": {
      "metadata": {
        "labels": {
          "app": "fake-name",
          "app_type": "beam_pipline_submitter"
        }
      },
      "spec": {
        "serviceAccountName": "sa",
        "volumes": [
          {
            "name": "shared-output",
            "emptyDir": {}
          }
        ],
        "terminationGracePeriodSeconds": 100,
        "initContainers": [
          {
            "name": "output-uploader",
            "image": "google/cloud-sdk:alpine",
            "restartPolicy": "Always",
            "command": [
              "/bin/sh",
              "-c"
            ],
            "args": [
              "while true; do gsutil -m rsync -r /output fake-output-dir/output/$HOSTNAME/; sleep 60; done"
            ],
            "resources": {
              "requests": {
                "cpu": "100m",
                "memory": "128Mi"
              },
              "limits": {
                "cpu": "500m",
                "memory": "256Mi"
              }
            },
            "volumeMounts": [
              {
                "name": "shared-output",
                "mountPath": "/output"
              }
            ]
          }
        ],
        "containers": [
          {
            "name": "fake-name",
            "env": [
              {
                "name": "PYTHONUNBUFFERED",
                "value": "1"
              }
            ],
            "image": "settings-repo/test-image:fake-name",
            "volumeMounts": [
              {
                "name": "shared-output",
                "mountPath": "/output"
              }
            ],
            "command": [
              "/bin/sh",
              "-c"
            ],
            "args": [
              ""
            ]
          }
        ],
        "restartPolicy": "Never"
      }
    }
  }
}
"""


def _get_expected_job_submission_command(parallelism):
    return (
        "python -m fake --command "
        "--flink_master=1.2.3.4:8081 "
        f"--parallelism={parallelism} "
        "--artifacts_dir=fake-output-dir/artifacts_dir "
        "--flink_version=1.18 "
        "--runner=FlinkRunner "
        "--environment_type=EXTERNAL "
        "--environment_config=localhost:50000 2>&1 | tee /output/beam_pipline_log"
    )


class FlinkTPUGKEJobTest(TestCase):
    """Tests GKEJob with Flink."""

    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        with (
            mock_gcp_settings(
                [jobset_utils.__name__, bundler.__name__],
                settings=self._settings,
            ),
        ):
            return super().run(result)

    def _job_config(
        self,
        bundler_cls: type[Bundler],
        command: str = "python -m fake --command",
        location_hint: Optional[str] = None,
        **kwargs,
    ) -> tuple[job_flink.FlinkTPUGKEJob.Config, Bundler.Config]:
        self._settings["location_hint"] = location_hint
        fv = flags.FlagValues()
        cfg = job_flink.FlinkTPUGKEJob.default_config().set(
            builder=jobset_utils.TPUReplicatedJob.default_config()
        )
        define_flags(cfg, fv)
        fv.set_default("name", "fake-name")
        fv.set_default("instance_type", "tpu-v5p-16")
        fv.set_default("output_dir", "fake-output-dir")
        for key, value in kwargs.items():
            if value is not None:
                setattr(fv, key, value)
        fv.mark_as_parsed()
        cfg = from_flags(cfg, fv, command=command)
        bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
        return cfg, bundler_cfg

    @parameterized.product(
        reservation=[None, "test"],
        service_account=[None, "sa"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        enable_pre_provisioner=[None, False, True],
        location_hint=["fake-location-hint", None],
        flink_threads_per_worker=[1, 2, 4],
    )
    def test_get_flinkdeployment(
        self,
        reservation,
        service_account,
        enable_pre_provisioner,
        location_hint,
        bundler_cls,
        flink_threads_per_worker,
    ):
        cfg, bundler_cfg = self._job_config(
            bundler_cls,
            location_hint=location_hint,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
            flink_threads_per_worker=flink_threads_per_worker,
        )
        flink_job: job_flink.FlinkTPUGKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable=protected-access
        system = flink_job._get_system_info()
        flink_deployment = flink_job._build_flink_deployment(system)
        expected_flink_deployment = json.loads(expected_flink_deployment_json)
        expected_flink_deployment["spec"]["serviceAccount"] = (
            service_account if service_account else self._settings["k8s_service_account"]
        )
        expected_flink_deployment["spec"]["flinkConfiguration"][
            "taskmanager.numberOfTaskSlots"
        ] = str(flink_threads_per_worker)
        if not location_hint:
            del expected_flink_deployment["spec"]["taskManager"]["podTemplate"]["spec"][
                "nodeSelector"
            ]["cloud.google.com/gke-location-hint"]
        try:
            self.assertDictEqual(expected_flink_deployment, flink_deployment)
        except AssertionError:
            logging.warning(
                "The actual flink_deployment is as follow in json format,"
                "please diff it with expected_flink_deployment_json"
            )
            logging.warning(json.dumps(flink_deployment, indent=2))
            raise

    @parameterized.product(
        reservation=[None, "test"],
        service_account=[None, "sa"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        enable_pre_provisioner=[None, False, True],
        flink_threads_per_worker=[1, 2, 4],
    )
    def test_get_job_submission_deployment(
        self,
        reservation,
        service_account,
        enable_pre_provisioner,
        bundler_cls,
        flink_threads_per_worker,
    ):
        cfg, bundler_cfg = self._job_config(
            bundler_cls,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
            flink_threads_per_worker=flink_threads_per_worker,
        )
        flink_job: job_flink.FlinkTPUGKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable=protected-access
        system = flink_job._get_system_info()
        job_submission = flink_job._build_job_submission_deployment("1.2.3.4", system)
        expected_job_submission = json.loads(expected_jobsubmission_json)
        expected_job_submission["spec"]["template"]["spec"]["serviceAccountName"] = (
            service_account if service_account else "settings-account"
        )
        expected_parallelism = flink_job._get_num_of_tpu_nodes(system) * flink_threads_per_worker
        expected_job_submission["spec"]["template"]["spec"]["containers"][0]["args"][
            0
        ] = _get_expected_job_submission_command(expected_parallelism)
        try:
            self.assertDictEqual(expected_job_submission, job_submission)
        except AssertionError:
            logging.warning(
                "The actual job_submission is as follow in json format,"
                "please diff it with expected_jobsubmission_json"
            )
            logging.warning(json.dumps(job_submission, indent=2))
            raise
