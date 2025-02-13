# Copyright © 2025 Apple Inc.

"""Tests Jobset utilities."""

import contextlib
import io
import math
import os
from datetime import datetime
from typing import Optional
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import (
    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR,
    BASTION_JOB_VERSION_ENV_VAR,
    deserialize_jobspec,
    new_jobspec,
    serialize_jobspec,
)
from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.types import JobMetadata
from axlearn.cloud.gcp import bundler, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.jobset_utils import (
    _MEMORY_REQUEST_PERCENTAGE,
    _METADATA_GOOGLE_INTERNAL_IP,
    BASTION_JOB_VERSION_LABEL,
    AcceleratorConfig,
    GCSFuseMount,
    HostMount,
)
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.test_utils import TestCase


def _create_serialized_job_spec(job_priority: int, user_id: str):
    test_spec = new_jobspec(
        name="test_job",
        command="test command",
        metadata=JobMetadata(
            user_id=user_id,
            project_id="test_project",
            # Make sure str timestamp isn't truncated even when some numbers are 0.
            creation_time=datetime(1900, 1, 1, 0, 0, 0, 0),
            resources={"test": 8},
            priority=job_priority,
        ),
    )
    serialized_jobspec = io.StringIO()
    serialize_jobspec(test_spec, serialized_jobspec)
    return serialized_jobspec.getvalue()


def mock_settings():
    return {
        "project": "settings-project",
        "zone": "settings-zone",
        "ttl_bucket": "settings-ttl-bucket",
        "gke_cluster": "settings-cluster",
        "gke_reservation": "settings-reservation",
        "k8s_service_account": "settings-account",
        "docker_repo": "settings-repo",
        "default_dockerfile": "settings-dockerfile",
        "location_hint": "settings-location-hint",
    }


class TPUReplicatedJobTest(TestCase):
    """Tests TPUReplicatedJob."""

    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: type[Bundler],
        reservation: Optional[str] = None,
        service_account: Optional[str] = None,
        enable_pre_provisioner: Optional[bool] = None,
        host_mount_spec: Optional[list[str]] = None,
        priority_class: Optional[str] = None,
        gcsfuse_mount_spec: Optional[str] = None,
    ):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__], mock_settings()):
            fv = flags.FlagValues()
            jobset_utils.TPUReplicatedJob.define_flags(fv)
            if reservation:
                fv.set_default("reservation", reservation)
            if service_account:
                fv.set_default("service_account", service_account)
            if host_mount_spec:
                fv.set_default("host_mount_spec", host_mount_spec)
            if gcsfuse_mount_spec:
                fv.set_default("gcsfuse_mount_spec", gcsfuse_mount_spec)
            fv.mark_as_parsed()
            cfg = jobset_utils.TPUReplicatedJob.from_flags(fv)
            cfg.accelerator = AcceleratorConfig().set(instance_type="tpu-v4-8")
            cfg.enable_pre_provisioner = enable_pre_provisioner
            cfg.priority_class = priority_class
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.product(
        [
            dict(env={}, reservation=None, expect_reserved=False),
            dict(env={"BASTION_TIER": "0"}, reservation=None, expect_reserved=False),
            dict(
                env={
                    "BASTION_TIER": "0",
                    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR: _create_serialized_job_spec(1, "user-1"),
                    BASTION_JOB_VERSION_ENV_VAR: "1",
                },
                reservation="test-reservation",
                expect_reserved=True,
            ),
            dict(
                env={"BASTION_TIER": "1", BASTION_JOB_VERSION_ENV_VAR: "2"},
                reservation="test-reservation",
                expect_reserved=False,
            ),
            dict(
                env={_BASTION_SERIALIZED_JOBSPEC_ENV_VAR: _create_serialized_job_spec(5, "user-2")},
                reservation="test-reservation",
                expect_reserved=False,
            ),
        ],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        enable_ici_resiliency=[True, False, None],
        enable_pre_provisioner=[None, True, False],
        location_hint=["test-location-hint", None],
        enable_tpu_smart_repair=[True, False],
        host_mount_spec=[["name=host-mount,host_path=/tmp,mount_path=/host-tmp"], None],
        gcsfuse_mount_spec=[
            ["mount_path=/tmp/gcsfuse", "gcs_path=/tmp/gcs_path", "shared_memory=5Gi"],
            None,
        ],
        priority_class=[None, "such-high-priority"],
    )
    def test_build_pod(
        self,
        bundler_cls: type[Bundler],
        expect_reserved: bool,
        enable_ici_resiliency: bool,
        env: dict,
        reservation: Optional[str] = None,
        enable_pre_provisioner: Optional[bool] = None,
        location_hint: Optional[str] = None,
        enable_tpu_smart_repair: bool = False,
        host_mount_spec: Optional[list[str]] = None,
        gcsfuse_mount_spec: Optional[list[str]] = None,
        priority_class: Optional[str] = None,
    ):
        with (
            mock.patch.dict("os.environ", env),
            self._job_config(
                bundler_cls,
                host_mount_spec=host_mount_spec,
                gcsfuse_mount_spec=gcsfuse_mount_spec,
                priority_class=priority_class,
            ) as (cfg, bundler_cfg),
        ):
            gke_job: jobset_utils.TPUReplicatedJob = cfg.set(
                reservation=reservation,
                enable_tpu_ici_resiliency=enable_ici_resiliency,
                enable_pre_provisioner=enable_pre_provisioner,
                location_hint=location_hint,
                name="test",
                enable_tpu_smart_repair=enable_tpu_smart_repair,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())
            # pylint: disable-next=protected-access
            pod = gke_job._build_pod()
            pod_spec = pod["spec"]
            node_selector = pod_spec["nodeSelector"]
            annotations = pod["metadata"]["annotations"]
            labels = pod["metadata"]["labels"]
            host_alias = pod["spec"]["hostAliases"]

            self.assertEqual(1, len(host_alias))
            self.assertEqual(
                dict(
                    ip=_METADATA_GOOGLE_INTERNAL_IP,
                    hostnames=["metadata", "metadata.google.internal"],
                ),
                host_alias[0],
            )

            # The reservation should be used only if scheduled as tier 0.
            if expect_reserved:
                self.assertEqual(
                    reservation, node_selector.get("cloud.google.com/reservation-name", None)
                )
                self.assertNotIn("cloud.google.com/gke-spot", node_selector)
                self.assertEqual([], pod_spec.get("tolerations", []))
                self.assertEqual("reserved", labels.get("bastion-tier", None))
            else:
                self.assertEqual("true", node_selector.get("cloud.google.com/gke-spot", None))
                self.assertNotIn("cloud.google.com/reservation-name", node_selector)
                tolerations = {
                    kv["key"]: (kv["value"], kv["effect"]) for kv in pod_spec.get("tolerations", [])
                }
                self.assertEqual(
                    ("true", "NoSchedule"), tolerations.get("cloud.google.com/gke-spot", None)
                )
                self.assertEqual("spot", labels.get("bastion-tier", None))

            self.assertEqual(len(pod_spec["containers"]), 1)

            # Verify worker container specs
            container = pod_spec["containers"][0]
            # Check command.
            self.assertIn("test_command", container["command"])

            if host_mount_spec:
                for v in pod_spec["volumes"]:
                    if v["name"] == "host-mount":
                        self.assertEqual(v["hostPath"], {"path": "/tmp", "type": "Directory"})
                        break
                else:
                    self.fail("host-mount not found!")

                for v in container["volumeMounts"]:
                    if v["name"] == "host-mount":
                        self.assertEqual(v["mountPath"], "/host-tmp")
                        self.assertEqual(v["readOnly"], False)
                        break
                else:
                    self.fail("host-mount not found!")

            if gcsfuse_mount_spec:
                self.assertIn("shared-memory", [v["name"] for v in pod_spec["volumes"]])
                for v in pod_spec["volumes"]:
                    if v["name"] == "shared-memory":
                        self.assertIn("sizeLimit", v["emptyDir"])
                        size_limit_request = [x for x in gcsfuse_mount_spec if "shared_memory" in x]
                        self.assertLessEqual(len(size_limit_request), 1)
                        if size_limit_request:
                            size_limit_request = size_limit_request[0].split("=")[1]
                            self.assertEqual(v["emptyDir"]["sizeLimit"], size_limit_request)
            else:
                self.assertNotIn("shared-memory", [v["name"] for v in pod_spec["volumes"]])

            self.assertEqual(container["imagePullPolicy"], "Always")

            # Check memory request.
            resources = container["resources"]
            self.assertIn("limits", resources)
            tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
            tpu_characteristics = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[tpu_type]
            memory_in_gi = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS.get(
                tpu_characteristics.gce_machine_type, None
            )
            if memory_in_gi is not None:
                self.assertEqual(resources["limits"]["memory"], f"{memory_in_gi}Gi")
                self.assertEqual(
                    resources["requests"]["memory"],
                    f"{math.floor(memory_in_gi * _MEMORY_REQUEST_PERCENTAGE)}Gi",
                )
            self.assertIn("google.com/tpu", resources["limits"])

            container_env = container["env"]
            container_env = {kv["name"]: kv for kv in container_env}

            if enable_ici_resiliency is not None:
                expected = "true" if enable_ici_resiliency else "false"
                self.assertEqual(
                    expected,
                    node_selector.get("cloud.google.com/gke-tpu-ici-resiliency", None),
                )
                self.assertEqual(
                    expected,
                    container_env["ENABLE_ICI_RESILIENCY"]["value"],
                )
            else:
                self.assertNotIn("cloud.google.com/gke-tpu-ici-resiliency", node_selector)
                self.assertNotIn("ENABLE_ICI_RESILIENCY", container_env)

            # Verify NODE_IP in container env.
            self.assertEqual(
                "status.hostIP",
                container_env["NODE_IP"]["valueFrom"]["fieldRef"]["fieldPath"],
            )

            # Verify uploader container specs
            self.assertEqual(len(pod_spec["initContainers"]), 1)

            uploader_container = pod_spec["initContainers"][0]
            self.assertEqual(uploader_container["name"], "output-uploader")
            self.assertEqual(uploader_container["image"], "google/cloud-sdk:alpine")
            self.assertEqual(uploader_container["restartPolicy"], "Always")
            self.assertIn("volumeMounts", uploader_container)

            volume_mounts = uploader_container["volumeMounts"]
            shared_output_mount = next(
                (vm for vm in volume_mounts if vm["name"] == "shared-output"), None
            )
            self.assertIsNotNone(shared_output_mount)
            self.assertEqual(shared_output_mount["mountPath"], "/output")

            command = uploader_container["command"]
            self.assertEqual(command, ["/bin/sh", "-c"])
            sync_command = uploader_container["args"][0]
            self.assertIn("gsutil -m rsync -r /output", sync_command)
            self.assertIn("$HOSTNAME", sync_command)
            self.assertIn("sleep", sync_command)

            if enable_pre_provisioner:
                self.assertIn(PRE_PROVISIONER_LABEL, node_selector)
                self.assertEqual(
                    "true",
                    annotations.get(
                        "tpu-provisioner.cloud.google.com/disable-autoprovisioning", None
                    ),
                )
            else:
                self.assertNotIn(PRE_PROVISIONER_LABEL, node_selector)
                self.assertEqual(
                    "false",
                    annotations.get(
                        "tpu-provisioner.cloud.google.com/disable-autoprovisioning", None
                    ),
                )

            self.assertEqual(location_hint, node_selector.get("cloud.google.com/gke-location-hint"))

            if _BASTION_SERIALIZED_JOBSPEC_ENV_VAR in env:
                spec = deserialize_jobspec(
                    io.StringIO(os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR))
                )

                self.assertEqual(str(spec.metadata.priority), labels.get("job-priority", None))
                self.assertEqual(
                    str(spec.metadata.priority), node_selector.get("job-priority", None)
                )
                self.assertEqual(spec.metadata.user_id, labels.get("user-id", None))
            else:
                self.assertNotIn("job-priority", labels)
                self.assertNotIn("job-priority", node_selector)
                self.assertNotIn("user-id", labels)

            if BASTION_JOB_VERSION_ENV_VAR in env:
                job_version = env.get(BASTION_JOB_VERSION_ENV_VAR)
                self.assertEqual(job_version, labels.get(BASTION_JOB_VERSION_LABEL, None))
            else:
                self.assertNotIn(BASTION_JOB_VERSION_LABEL, labels)

            if enable_tpu_smart_repair:
                self.assertIn(
                    "cloud.google.com/gke-tpu-auto-restart",
                    annotations.get("tpu-provisioner.cloud.google.com/copy-labels", {}),
                )
                self.assertEqual("true", labels.get("cloud.google.com/gke-tpu-auto-restart", None))
            else:
                self.assertNotIn(
                    "cloud.google.com/gke-tpu-auto-restart",
                    annotations.get("tpu-provisioner.cloud.google.com/copy-labels", {}),
                )
                self.assertNotIn("cloud.google.com/gke-tpu-auto-restart", labels)

            if priority_class is None:
                self.assertNotIn("priorityClassName", pod_spec)
            else:
                self.assertEqual(pod_spec.get("priorityClassName", None), priority_class)

    def test_mount_dataclass(self):
        # pylint: disable=missing-kwoa
        # pytype: disable=missing-parameter
        with self.assertRaises(TypeError):
            m = GCSFuseMount()

        m = GCSFuseMount(gcs_path="test")
        self.assertEqual(m.name, "gcs-fuse-csi-ephemeral")
        with self.assertRaises(TypeError):
            m = HostMount(mount_path="test")

        m = HostMount(mount_path="test", name="test", host_path="test")
        # pytype: enable=missing-parameter
        self.assertEqual(m.read_only, False)


class A3ReplicatedJobTest(TestCase):
    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: type[Bundler],
        num_replicas: int,
        service_account: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__], mock_settings()):
            fv = flags.FlagValues()
            jobset_utils.A3ReplicatedJob.define_flags(fv)
            if service_account:
                fv.set_default("service_account", service_account)
            fv.mark_as_parsed()
            cfg: jobset_utils.A3ReplicatedJob.Config = jobset_utils.A3ReplicatedJob.from_flags(fv)
            cfg.accelerator = AcceleratorConfig().set(
                instance_type="gpu-a3-highgpu-8g-256",
                num_replicas=num_replicas,
            )
            cfg.command = "test-command"
            cfg.env_vars = env_vars if env_vars is not None else {}
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.product(
        env_vars=[dict(), dict(XLA_FLAGS="--should-overwrite-all")],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        num_replicas=[1, 32],
    )
    def test_build_pod(
        self,
        bundler_cls,
        num_replicas: int,
        env_vars: Optional[dict] = None,
    ):
        with self._job_config(bundler_cls, env_vars=env_vars, num_replicas=num_replicas) as (
            cfg,
            bundler_cfg,
        ):
            gke_job: jobset_utils.A3ReplicatedJob = cfg.set(name="test").instantiate(
                bundler=bundler_cfg.instantiate()
            )
            # pylint: disable-next=protected-access
            pod = gke_job._build_pod()
            pod_spec = pod["spec"]

            self.assertEqual(len(pod_spec["containers"]), 2)
            containers = {container["name"]: container for container in pod_spec["containers"]}
            self.assertIn("tcpx-daemon", containers)
            main_container = containers["test"]
            main_container_env = main_container["env"]
            main_container_env_vars = {env["name"]: env for env in main_container_env}
            self.assertEqual(main_container["resources"]["limits"]["nvidia.com/gpu"], "8")
            self.assertEqual(main_container_env_vars["NUM_PROCESSES"]["value"], f"{num_replicas}")
            # Verify that default XLA flags can be overwritten by user.
            if env_vars and env_vars.get("XLA_FLAGS"):
                self.assertEqual(
                    main_container_env_vars["XLA_FLAGS"]["value"], env_vars["XLA_FLAGS"]
                )
