# Copyright © 2025 Apple Inc.

"""Tests Jobset utilities."""

import contextlib
import io
import json
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
from axlearn.cloud.common.pod_mutator import PodMutator
from axlearn.cloud.common.types import JobMetadata
from axlearn.cloud.common.utils import AcceleratorConfig, define_flags, from_flags
from axlearn.cloud.gcp import bundler, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.jobset_utils import (
    _ANNOTATION_ADDITIONAL_NODE_NETWORKS,
    _ANNOTATION_NODE_SERVICE_ACCOUNT,
    _MEMORY_REQUEST_PERCENTAGE,
    _METADATA_GOOGLE_INTERNAL_IP,
    _PERSISTENT_DISK_SIZE_MAX_GIB,
    BASTION_JOB_VERSION_LABEL,
    BaseReplicatedJob,
    CompositeReplicatedJob,
    EphemeralDiskMount,
    GCSFuseMount,
    HostMount,
    TPUReplicatedJob,
    _LoadBalancer,
)
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.tpu import get_default_env
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class
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


class MockUserCommandPatcher(jobset_utils.UserCommandPatcher):
    def patch(self, command: str, **kwargs: dict) -> str:
        return f"{command} && ./postfix_command.sh"


class TPUReplicatedJobTest(TestCase):
    """Tests TPUReplicatedJob."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], **kwargs):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            jobset_utils.TPUReplicatedJob.define_flags(fv)
            fv.set_default("name", "test-name")
            fv.set_default("instance_type", "tpu-v4-8")
            fv.set_default("topology", None)
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = jobset_utils.TPUReplicatedJob.from_flags(fv)
            cfg.project = jobset_utils.gcp_settings("project", required=True, fv=fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    def test_env_override(self):
        # Tests that env flags can override defaults.
        with self._job_config(
            env=["key1:value1", "TPU_TYPE:dummy"], bundler_cls=ArtifactRegistryBundler
        ) as (cfg, _):
            env = get_default_env(tpu_type="test", num_tpu_slices=1, job_name=cfg.name)
            self.assertIn("TPU_TYPE", env)
            self.assertNotIn("key1", env)
            self.assertEqual(env["TPU_TYPE"], "test")

            # Test that the env is updated accordingly.
            for k, v in env.items():
                if k == "TPU_TYPE":
                    self.assertEqual(cfg.env_vars[k], "dummy")  # Should reflect updated value.
                else:
                    self.assertEqual(cfg.env_vars[k], v)  # Should reflect original value.
            # The newly added env should be present.
            self.assertIn("key1", cfg.env_vars)
            self.assertEqual(cfg.env_vars["key1"], "value1")

    def test_validate_jobset_name(self):
        with (
            self.assertRaisesRegex(ValueError, "invalid"),
            self._job_config(bundler_cls=ArtifactRegistryBundler) as (cfg, _),
        ):
            cfg.set(name="invalid_underscore_name", command="", output_dir="")
            cfg.instantiate(bundler=mock.create_autospec(Bundler))

    @parameterized.product(
        [
            dict(
                env={},
                reservation=None,
                reservation_project=None,
                expect_reserved=False,
                expect_spot_selector=True,
            ),
            dict(
                env={"BASTION_TIER": "0"},
                reservation=None,
                reservation_project=None,
                expect_reserved=False,
                expect_spot_selector=True,
            ),
            dict(
                env={
                    "BASTION_TIER": "0",
                    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR: _create_serialized_job_spec(1, "user-1"),
                    BASTION_JOB_VERSION_ENV_VAR: "1",
                },
                reservation="test-reservation",
                reservation_project=None,
                expect_reserved=True,
                expect_spot_selector=False,
            ),
            dict(
                env={
                    "BASTION_TIER": "0",
                    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR: _create_serialized_job_spec(1, "user-1"),
                    BASTION_JOB_VERSION_ENV_VAR: "1",
                },
                reservation="test-reservation",
                reservation_project="test-reservation-project",
                expect_reserved=True,
                expect_spot_selector=False,
            ),
            dict(
                env={"BASTION_TIER": "1", BASTION_JOB_VERSION_ENV_VAR: "2"},
                reservation="test-reservation",
                reservation_project=None,
                expect_reserved=False,
                expect_spot_selector=True,
            ),
            dict(
                env={_BASTION_SERIALIZED_JOBSPEC_ENV_VAR: _create_serialized_job_spec(5, "user-2")},
                reservation="test-reservation",
                reservation_project=None,
                expect_reserved=False,
                expect_spot_selector=True,
            ),
            dict(
                env={"BASTION_TIER": "disabled"},
                reservation=None,
                reservation_project=None,
                expect_reserved=False,
                expect_spot_selector=False,
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
        additional_node_networks=[None, "network-1:subnet-1,network-2:subnet-2"],
        image_id=[None, "my-image-id"],
        user_command_patcher=[None, MockUserCommandPatcher.default_config()],
    )
    # TODO: Try to reduce positional arguments
    # pylint: disable-next=too-many-positional-arguments
    def test_build_pod(
        self,
        bundler_cls: type[Bundler],
        expect_reserved: bool,
        expect_spot_selector: bool,
        enable_ici_resiliency: bool,
        env: dict,
        reservation: Optional[str] = None,
        reservation_project: Optional[str] = None,
        enable_pre_provisioner: Optional[bool] = None,
        location_hint: Optional[str] = None,
        enable_tpu_smart_repair: bool = False,
        host_mount_spec: Optional[list[str]] = None,
        gcsfuse_mount_spec: Optional[list[str]] = None,
        priority_class: Optional[str] = None,
        additional_node_networks: Optional[str] = None,
        image_id: Optional[str] = None,
        user_command_patcher: Optional[jobset_utils.UserCommandPatcher] = None,
    ):
        with (
            mock.patch("os.environ", env),
            self._job_config(
                bundler_cls,
                host_mount_spec=host_mount_spec,
                gcsfuse_mount_spec=gcsfuse_mount_spec,
                priority_class=priority_class,
                image_id=image_id,
            ) as (cfg, bundler_cfg),
        ):
            gke_job: jobset_utils.TPUReplicatedJob = cfg.set(
                reservation=reservation,
                reservation_project=reservation_project,
                enable_tpu_ici_resiliency=enable_ici_resiliency,
                enable_pre_provisioner=enable_pre_provisioner,
                location_hint=location_hint,
                name="test",
                enable_tpu_smart_repair=enable_tpu_smart_repair,
                command="test_command",
                output_dir="FAKE",
                additional_node_networks=additional_node_networks,
                user_command_patcher=user_command_patcher,
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
                if reservation_project is not None:
                    self.assertEqual(
                        reservation_project,
                        node_selector.get("cloud.google.com/reservation-project", None),
                    )
                else:
                    self.assertNotIn("cloud.google.com/reservation-project", node_selector)
                self.assertNotIn("cloud.google.com/gke-spot", node_selector)
                self.assertEqual([], pod_spec.get("tolerations", []))
                self.assertEqual("reserved", labels.get("bastion-tier", None))
            else:
                if expect_spot_selector:
                    self.assertEqual("true", node_selector.get("cloud.google.com/gke-spot", None))
                    self.assertEqual("spot", labels.get("bastion-tier", None))
                    tolerations = {
                        kv["key"]: (kv["value"], kv["effect"])
                        for kv in pod_spec.get("tolerations", [])
                    }
                    self.assertEqual(
                        ("true", "NoSchedule"), tolerations.get("cloud.google.com/gke-spot", None)
                    )
                else:
                    self.assertNotIn("cloud.google.com/gke-spot", node_selector)
                    self.assertNotIn("provisioner-nodepool-id", node_selector)
                self.assertNotIn("cloud.google.com/reservation-name", node_selector)

            self.assertEqual(len(pod_spec["containers"]), 1)

            # Verify worker container specs
            container = pod_spec["containers"][0]
            # Check command.
            if user_command_patcher is None:
                self.assertIn("test_command", container["command"])
            else:
                user_command = container["command"][-1]
                self.assertTrue("test_command" in user_command)
                self.assertTrue("&& ./postfix_command.sh" in user_command)
            if image_id:
                self.assertEqual(image_id, container["image"])
            else:
                self.assertIn("test-image", container["image"])

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

            # shared-memory volume is present when gcsfuse or shared_memory_size_gb is set.
            volume_names = [v["name"] for v in pod_spec["volumes"]]
            if gcsfuse_mount_spec or cfg.shared_memory_size_gb is not None:
                self.assertIn("shared-memory", volume_names)
                for v in pod_spec["volumes"]:
                    if v["name"] == "shared-memory":
                        self.assertEqual(v["emptyDir"]["medium"], "Memory")
                        if gcsfuse_mount_spec:
                            # When gcsfuse is enabled, sizeLimit comes from gcsfuse shared_memory.
                            size_limit_request = [
                                x for x in gcsfuse_mount_spec if "shared_memory" in x
                            ]
                            if size_limit_request:
                                expected = size_limit_request[0].split("=")[1]
                            else:
                                expected = "1Gi"  # GCSFuseMount default
                            self.assertEqual(v["emptyDir"]["sizeLimit"], expected)
                        else:
                            # Without gcsfuse, uses shared_memory_size_gb.
                            self.assertEqual(
                                v["emptyDir"]["sizeLimit"], f"{cfg.shared_memory_size_gb}Gi"
                            )
            else:
                self.assertNotIn("shared-memory", volume_names)

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

            # Verify NUM_REPLICAS in container env.
            self.assertEqual(
                "metadata.annotations['jobset.sigs.k8s.io/replicatedjob-replicas']",
                container_env["NUM_REPLICAS"]["valueFrom"]["fieldRef"]["fieldPath"],
            )

            # Verify REPLICA_ID in container env.
            self.assertEqual(
                "metadata.annotations['jobset.sigs.k8s.io/job-index']",
                container_env["REPLICA_ID"]["valueFrom"]["fieldRef"]["fieldPath"],
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
                self.assertEqual(spec.metadata.project_id, labels.get("project-id", None))
                self.assertEqual(
                    str(gke_job.config.accelerator.num_replicas),
                    labels.get("num-replicas", None),
                )
            else:
                self.assertNotIn("job-priority", labels)
                self.assertNotIn("job-priority", node_selector)
                self.assertNotIn("user-id", labels)
                self.assertNotIn("project-id", labels)

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

            if additional_node_networks:
                self.assertEqual(
                    additional_node_networks,
                    annotations.get(_ANNOTATION_ADDITIONAL_NODE_NETWORKS),
                )
                self.assertEqual(
                    f"{cfg.service_account}@{cfg.project}.iam.gserviceaccount.com",
                    annotations.get(_ANNOTATION_NODE_SERVICE_ACCOUNT),
                )
                self.assertTrue(pod_spec.get("hostNetwork", False))
                self.assertEqual(pod_spec.get("dnsPolicy"), "ClusterFirstWithHostNet")
            else:
                self.assertNotIn(_ANNOTATION_ADDITIONAL_NODE_NETWORKS, annotations)
                self.assertNotIn(_ANNOTATION_NODE_SERVICE_ACCOUNT, annotations)
                self.assertNotIn("hostNetwork", pod_spec)
                self.assertNotIn("dnsPolicy", pod_spec)

    def test_replicated_job(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            replicated_job: jobset_utils.TPUReplicatedJob = cfg.set(
                name="test",
                command="test_command",
                output_dir="FAKE",
                job_name="replicatedJob",
            ).instantiate(bundler=bundler_cfg.instantiate())

            job_spec = replicated_job()[0]["template"]
            annotations = job_spec["metadata"]["annotations"]
            # Annotation to create load balancer.
            self.assertEqual(
                "test-replicatedJob-service",
                annotations.get("axlearn/replicatedjob-load-balancer-service-name", {}),
            )
            self.assertEqual(
                "9000",
                annotations.get("axlearn/replicatedjob-load-balancer-target-port", {}),
            )
            self.assertEqual(
                "80",
                annotations.get("axlearn/replicatedjob-load-balancer-port", {}),
            )

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

    def test_load_balancer_class(self):
        lb1 = _LoadBalancer(jobset_name="jobset1", replicated_job_name="job1")
        self.assertEqual(lb1.service_name, "jobset1-job1-service")
        self.assertEqual(lb1.target_port, 9000)
        self.assertEqual(lb1.port, 80)
        lb2 = _LoadBalancer(
            jobset_name="jobset1", replicated_job_name="job2", target_port=8080, port=443
        )
        self.assertEqual(lb2.service_name, "jobset1-job2-service")
        self.assertEqual(lb2.target_port, 8080)
        self.assertEqual(lb2.port, 443)

    @parameterized.parameters(
        dict(
            instance_type="v5p-16",
            topology="2x2x2",
            expected=ValueError("custom topology is only available for v5p-128 and above."),
        ),
        dict(
            instance_type="v5p-128",
            topology="2x64",
            expected=ValueError("custom topology only supports 3d topology for v5p."),
        ),
        dict(
            instance_type="v5p-128",
            topology="2x1x64",
            expected=ValueError("There should be no 1 in each topology dimension."),
        ),
        dict(
            instance_type="v5p-128",
            topology="2x2x2",
            expected=ValueError(
                "custom topology 2x2x2 doesn't match the number of cores in instance_type v5p-128."
            ),
        ),
        dict(instance_type="v5p-128", topology="2x4x8", expected=None),
    )
    def test_verify_custom_topology_availability(self, instance_type, topology, expected):
        accelerator = AcceleratorConfig().set(instance_type=instance_type, topology=topology)
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                TPUReplicatedJob.verify_custom_topology_availability(accelerator)
        else:
            TPUReplicatedJob.verify_custom_topology_availability(accelerator)

    @parameterized.parameters(
        # v6e → hyperdisk-balanced
        dict(
            instance_type="tpu-v6e-16",
            persistent_disk_size_gb=500,
            expected_class="hyperdisk-balanced",
        ),
        # v4 → hyperdisk-balanced
        dict(
            instance_type="tpu-v4-8",
            persistent_disk_size_gb=100,
            expected_class="hyperdisk-balanced",
        ),
        # v5p → pd-balanced
        dict(instance_type="tpu-v5p-8", persistent_disk_size_gb=200, expected_class="pd-balanced"),
        # v5litepod → pd-balanced
        dict(
            instance_type="tpu-v5litepod-16",
            persistent_disk_size_gb=300,
            expected_class="pd-balanced",
        ),
    )
    def test_persistent_disk_storage_class(
        self, instance_type, persistent_disk_size_gb, expected_class
    ):
        """Tests that from_flags sets the correct storage class for ephemeral_disk."""
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            jobset_utils.TPUReplicatedJob.define_flags(fv)
            fv.set_default("name", "test-name")
            fv.set_default("instance_type", instance_type)
            fv.set_default("topology", None)
            setattr(fv, "persistent_disk_size_gb", persistent_disk_size_gb)
            fv.mark_as_parsed()
            cfg = jobset_utils.TPUReplicatedJob.from_flags(fv)
            self.assertIsNotNone(cfg.ephemeral_disk)
            self.assertEqual(cfg.ephemeral_disk.storage_class, expected_class)
            self.assertEqual(cfg.ephemeral_disk.size_gb, persistent_disk_size_gb)
            self.assertEqual(cfg.ephemeral_disk.name, "persistent-disk")
            self.assertEqual(cfg.ephemeral_disk.mount_path, "/mnt")

    def test_no_persistent_disk_when_flag_unset(self):
        """Tests that ephemeral_disk is None when --persistent_disk_size_gb is not provided."""
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            jobset_utils.TPUReplicatedJob.define_flags(fv)
            fv.set_default("name", "test-name")
            fv.set_default("instance_type", "tpu-v6e-16")
            fv.set_default("topology", None)
            fv.mark_as_parsed()
            cfg = jobset_utils.TPUReplicatedJob.from_flags(fv)
            self.assertIsNone(cfg.ephemeral_disk)

    def test_persistent_disk_size_exceeds_limit(self):
        """Tests that from_flags raises ValueError when persistent_disk_size_gb is out of range."""
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            for bad_size in (0, -1, _PERSISTENT_DISK_SIZE_MAX_GIB + 1):
                fv = flags.FlagValues()
                jobset_utils.TPUReplicatedJob.define_flags(fv)
                fv.set_default("name", "test-name")
                fv.set_default("instance_type", "tpu-v6e-16")
                fv.set_default("topology", None)
                setattr(fv, "persistent_disk_size_gb", bad_size)
                fv.mark_as_parsed()
                with self.assertRaisesRegex(ValueError, "must be between"):
                    jobset_utils.TPUReplicatedJob.from_flags(fv)

    @parameterized.parameters(
        dict(
            instance_type="tpu-v6e-16",
            persistent_disk_size_gb=500,
            expected_class="hyperdisk-balanced",
        ),
        dict(instance_type="tpu-v5p-8", persistent_disk_size_gb=200, expected_class="pd-balanced"),
    )
    def test_persistent_disk_pod_spec(self, instance_type, persistent_disk_size_gb, expected_class):
        """Tests that _build_pod and _build_container include ephemeral disk volume and mount."""
        with self._job_config(
            bundler_cls=ArtifactRegistryBundler,
            instance_type=instance_type,
            persistent_disk_size_gb=persistent_disk_size_gb,
        ) as (cfg, bundler_cfg):
            cfg.set(command="test-command", output_dir="gs://bucket/output")
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = job._build_pod()  # pylint: disable=protected-access

            # Check that the ephemeral volume is present in pod volumes.
            volumes = pod["spec"]["volumes"]
            ephemeral_vols = [v for v in volumes if v.get("name") == "persistent-disk"]
            self.assertLen(ephemeral_vols, 1)
            vol = ephemeral_vols[0]
            self.assertIn("ephemeral", vol)
            claim_spec = vol["ephemeral"]["volumeClaimTemplate"]["spec"]
            self.assertEqual(claim_spec["storageClassName"], expected_class)
            self.assertEqual(claim_spec["accessModes"], ["ReadWriteOnce"])
            self.assertEqual(
                claim_spec["resources"]["requests"]["storage"],
                f"{persistent_disk_size_gb}Gi",
            )

            # Check that the volume mount is present in the main container.
            container = job._build_container()  # pylint: disable=protected-access
            mounts = container["volumeMounts"]
            data_mounts = [m for m in mounts if m.get("name") == "persistent-disk"]
            self.assertLen(data_mounts, 1)
            self.assertEqual(data_mounts[0]["mountPath"], "/mnt")

    def test_no_persistent_disk_pod_spec(self):
        """Tests that no ephemeral disk volume or mount appears when flag is not set."""
        with self._job_config(
            bundler_cls=ArtifactRegistryBundler,
            instance_type="tpu-v6e-16",
        ) as (cfg, bundler_cfg):
            cfg.set(command="test-command", output_dir="gs://bucket/output")
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = job._build_pod()  # pylint: disable=protected-access

            volumes = pod["spec"]["volumes"]
            ephemeral_vols = [v for v in volumes if v.get("name") == "persistent-disk"]
            self.assertEmpty(ephemeral_vols)

            container = job._build_container()  # pylint: disable=protected-access
            mounts = container["volumeMounts"]
            data_mounts = [m for m in mounts if m.get("name") == "persistent-disk"]
            self.assertEmpty(data_mounts)

    def test_shared_memory_default_no_volume(self):
        """Tests that shared-memory volume is not created when shared_memory_size_gb is None."""
        with self._job_config(bundler_cls=ArtifactRegistryBundler) as (cfg, bundler_cfg):
            cfg.set(command="test-command", output_dir="gs://bucket/output")
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = job._build_pod()  # pylint: disable=protected-access
            volumes = pod["spec"]["volumes"]
            shm_vols = [v for v in volumes if v["name"] == "shared-memory"]
            self.assertEmpty(shm_vols)

    def test_shared_memory_with_size_limit(self):
        """Tests that shared-memory volume respects shared_memory_size_gb config."""
        with self._job_config(bundler_cls=ArtifactRegistryBundler) as (cfg, bundler_cfg):
            cfg.set(
                command="test-command",
                output_dir="gs://bucket/output",
                shared_memory_size_gb=500,
            )
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = job._build_pod()  # pylint: disable=protected-access
            volumes = pod["spec"]["volumes"]
            shm_vols = [v for v in volumes if v["name"] == "shared-memory"]
            self.assertLen(shm_vols, 1)
            self.assertEqual(shm_vols[0]["emptyDir"]["medium"], "Memory")
            self.assertEqual(shm_vols[0]["emptyDir"]["sizeLimit"], "500Gi")

    def test_shared_memory_volume_mount_not_present_by_default(self):
        """Tests that /dev/shm volume mount is not present when shared_memory_size_gb is None."""
        with self._job_config(bundler_cls=ArtifactRegistryBundler) as (cfg, bundler_cfg):
            cfg.set(command="test-command", output_dir="gs://bucket/output")
            self.assertIsNone(cfg.gcsfuse_mount)
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            container = job._build_container()  # pylint: disable=protected-access
            shm_mounts = [m for m in container["volumeMounts"] if m.get("name") == "shared-memory"]
            self.assertEmpty(shm_mounts)

    def test_shared_memory_size_gb_zero_means_unlimited(self):
        """Tests that shared_memory_size_gb=0 means unlimited (no sizeLimit)."""
        with self._job_config(bundler_cls=ArtifactRegistryBundler) as (cfg, bundler_cfg):
            cfg.set(
                command="test-command",
                output_dir="gs://bucket/output",
                shared_memory_size_gb=0,
            )
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = job._build_pod()  # pylint: disable=protected-access
            volumes = pod["spec"]["volumes"]
            shm_vols = [v for v in volumes if v["name"] == "shared-memory"]
            self.assertLen(shm_vols, 1)
            self.assertEqual(shm_vols[0]["emptyDir"]["medium"], "Memory")
            self.assertEqual(shm_vols[0]["emptyDir"]["sizeLimit"], "0Gi")

    def test_ephemeral_disk_mount_dataclass(self):
        """Tests EphemeralDiskMount defaults and field assignment."""
        m = EphemeralDiskMount(storage_class="hyperdisk-balanced", size_gb=500)
        self.assertEqual(m.name, "persistent-disk")
        self.assertEqual(m.mount_path, "/mnt")
        self.assertEqual(m.storage_class, "hyperdisk-balanced")
        self.assertEqual(m.size_gb, 500)
        self.assertEqual(m.read_only, False)


class CompositeReplicatedJobTest(TestCase):
    def test_composite_replicated_job(self):
        # pylint: disable=missing-class-docstring

        class DummyReplicatedJob(BaseReplicatedJob):
            @config_class
            class Config(BaseReplicatedJob.Config):
                command: Required[str] = REQUIRED

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                flags.DEFINE_string("command", None, "Command", flag_values=fv)

            def __call__(self):
                cfg: DummyReplicatedJob.Config = self.config  # pytype: disable=invalid-annotation
                return [{"name": cfg.name, "command": cfg.command}]

        cfg: CompositeReplicatedJob.Config = CompositeReplicatedJob.default_config().set(
            inner={
                "a": DummyReplicatedJob.default_config(),
                "b": DummyReplicatedJob.default_config(),
            }
        )
        fv = flags.FlagValues()
        define_flags(cfg, fv)
        for child in ("a", "b"):
            setattr(fv, f"{child}.name", child)
            setattr(fv, f"{child}.command", f"{child}_command")
        from_flags(cfg, fv)

        for child in ("a", "b"):
            # pylint: disable=unsubscriptable-object
            self.assertEqual(cfg.inner[child].name, child)
            self.assertEqual(cfg.inner[child].command, f"{child}_command")

        composite = cfg.instantiate(bundler=mock.create_autospec(Bundler))
        self.assertNestedEqual(
            [{"name": "a", "command": "a_command"}, {"name": "b", "command": "b_command"}],
            composite(),
        )


class A3HighReplicatedJobTest(TestCase):
    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: type[Bundler],
        num_replicas: int,
        env_vars: Optional[dict] = None,
    ):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            jobset_utils.A3HighReplicatedJob.define_flags(fv)
            fv.set_default("instance_type", "gpu-a3-highgpu-8g-256")
            fv.set_default("num_replicas", num_replicas)
            fv.mark_as_parsed()
            cfg: jobset_utils.A3HighReplicatedJob.Config = (
                jobset_utils.A3HighReplicatedJob.from_flags(fv)
            )
            cfg.project = jobset_utils.gcp_settings("project", required=True, fv=fv)
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
            gke_job: jobset_utils.A3HighReplicatedJob = cfg.set(name="test").instantiate(
                bundler=bundler_cfg.instantiate()
            )
            # pylint: disable-next=protected-access
            pod = gke_job._build_pod()
            pod_spec = pod["spec"]

            self.assertEqual(len(pod_spec["containers"]), 1)
            self.assertEqual(len(pod_spec["initContainers"]), 2)
            containers = {container["name"]: container for container in pod_spec["containers"]}
            init_containers = {
                init_container["name"]: init_container
                for init_container in pod_spec["initContainers"]
            }
            self.assertIn("tcpx-daemon", init_containers)
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

    def test_gpu_pod_mutators(self):
        """Tests that pod_mutators are applied in GPUReplicatedJob._build_pod()."""
        with self._job_config(ArtifactRegistryBundler, num_replicas=1) as (cfg, bundler_cfg):

            class _TestMutator(PodMutator):
                @jobset_utils.config_class
                class Config(PodMutator.Config):
                    marker: str = ""

                def mutate(self, job_spec, pod):
                    pod["metadata"]["annotations"]["test-mutator"] = self.config.marker
                    return pod

            cfg.pod_mutators = [_TestMutator.default_config().set(marker="gpu-marker")]
            job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            result = job()
            pod = result[0]["template"]["spec"]["template"]
            self.assertEqual(pod["metadata"]["annotations"]["test-mutator"], "gpu-marker")


class TopologyAssignmentTest(TestCase):
    """Tests topology assignment functionality."""

    def _tpu_child_config(
        self,
        *,
        job_name: str = "worker",
        instance_type: str = "tpu-7x-128",
        num_replicas: int = 1,
        enable_tpu_slice_auto_provisioning: bool = True,
    ):
        """Helper to create a TPUReplicatedJob config for composite job tests."""
        return TPUReplicatedJob.default_config().set(
            name=job_name,
            job_name=job_name,
            command="echo test",
            project="test-project",
            output_dir="gs://test-bucket/output",
            accelerator=AcceleratorConfig(instance_type=instance_type, num_replicas=num_replicas),
            enable_tpu_slice_auto_provisioning=enable_tpu_slice_auto_provisioning,
        )

    def test_tpu_job_get_workload_labels_with_topology(self):
        """TPUJobBuilder.get_workload_labels returns slice-autoprovisioning label."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_cfg = self._tpu_child_config(job_name="worker")
            child_cfg.topology_assignment = [["sb-1"]]
            job = child_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            self.assertEqual(
                {"tpu-provisioner.cloud.google.com/slice-autoprovisioning": "sync"},
                job.get_workload_labels(),
            )

    def test_tpu_job_get_workload_annotations_with_topology(self):
        """TPUJobBuilder.get_workload_annotations returns slice-selection annotation."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_cfg = self._tpu_child_config(job_name="worker")
            child_cfg.topology_assignment = [["sb-1"]]
            job = child_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            annotations = job.get_workload_annotations()
            self.assertIn("tpu-provisioner.cloud.google.com/slice-selection", annotations)
            selection = json.loads(annotations["tpu-provisioner.cloud.google.com/slice-selection"])
            self.assertEqual({"worker": [["sb-1"]]}, selection)

    def test_tpu_job_get_workload_labels_without_topology(self):
        """TPUJobBuilder.get_workload_labels returns empty dict when no topology."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_cfg = self._tpu_child_config(job_name="worker")
            job = child_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            self.assertEqual({}, job.get_workload_labels())
            self.assertEqual({}, job.get_workload_annotations())

    def test_tpu_job_get_workload_labels_provisioning_disabled(self):
        """TPUJobBuilder.get_workload_labels returns empty dict when provisioning disabled."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_cfg = self._tpu_child_config(
                job_name="worker", enable_tpu_slice_auto_provisioning=False
            )
            child_cfg.topology_assignment = [["sb-1"]]
            job = child_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            self.assertEqual({}, job.get_workload_labels())
            self.assertEqual({}, job.get_workload_annotations())

    def test_composite_get_workload_labels_merges_children(self):
        """CompositeReplicatedJob.get_workload_labels merges from enabled children."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_cfg = self._tpu_child_config(job_name="worker")
            child_cfg.topology_assignment = [["sb-1"]]
            composite_cfg = CompositeReplicatedJob.default_config().set(
                name="composite", inner={"a": child_cfg}
            )
            composite = composite_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            self.assertEqual(
                {"tpu-provisioner.cloud.google.com/slice-autoprovisioning": "sync"},
                composite.get_workload_labels(),
            )

    def test_composite_get_workload_annotations_merges_children(self):
        """CompositeReplicatedJob.get_workload_annotations merges from enabled children."""
        with mock_gcp_settings([jobset_utils.__name__]):
            child_a = self._tpu_child_config(job_name="a-worker")
            child_a.topology_assignment = [["sb-1"]]
            child_b = self._tpu_child_config(job_name="b-worker")
            child_b.topology_assignment = [["sb-2"]]
            composite_cfg = CompositeReplicatedJob.default_config().set(
                name="composite", inner={"a": child_a, "b": child_b}
            )
            composite = composite_cfg.instantiate(bundler=mock.create_autospec(Bundler))
            annotations = composite.get_workload_annotations()
            # Both children contribute; annotations will be merged (last wins for same key)
            self.assertIn("tpu-provisioner.cloud.google.com/slice-selection", annotations)
