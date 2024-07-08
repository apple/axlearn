# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs.

    python3 -m axlearn.cloud.gcp.job_test TPUJobTest.test_execute_from_local \
        --tpu_type=v4-8 --project=my-project --zone=my-zone

    python3 -m axlearn.cloud.gcp.job_test CPUJobTest.test_execute_from_local \
        --project=my-project --zone=my-zone

"""

import atexit
import contextlib
import math
import os
import subprocess
import sys
from typing import Optional, Type, Union
from unittest import mock

import pytest
from absl import flags, logging
from absl.testing import absltest, parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import configure_logging, generate_job_name
from axlearn.cloud.gcp import bundler, job
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler, GCSTarBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import (
    _MEMORY_REQUEST_PERCENTAGE,
    CPUJob,
    TPUQRMJob,
    _kill_ssh_agent,
    _start_ssh_agent,
)
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.tpu import create_queued_tpu, delete_queued_tpu, infer_tpu_type, qrm_resource
from axlearn.cloud.gcp.utils import common_flags, get_credentials
from axlearn.cloud.gcp.vm import create_vm, delete_vm
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.test_utils import TestCase


@contextlib.contextmanager
def mock_job(module_name: str):
    with mock.patch(f"{module_name}.get_credentials", return_value=None):
        yield


def _private_flags():
    common_flags()
    flags.DEFINE_string("tpu_type", "v4-8", "TPU type to test with")


FLAGS = flags.FLAGS


class DummyRemoteTPUJob(TPUQRMJob):
    """A dummy TPU job."""

    def _execute(self) -> Union[subprocess.CompletedProcess, subprocess.Popen]:
        """Provisions a TPU and launches a command."""
        cfg: TPUQRMJob.Config = self.config
        bundle_id = self._bundler.bundle(cfg.name)
        resource = qrm_resource(get_credentials())
        create_queued_tpu(
            cfg.name,
            resource,
            tpu_type=infer_tpu_type(cfg.accelerator.instance_type),
            bundler_type=self._bundler.TYPE,
        )
        out = self._execute_remote_cmd(
            f"{self._bundler.install_command(bundle_id)} && {cfg.command}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        delete_queued_tpu(cfg.name, resource)
        return out[0]


@pytest.mark.tpu
@pytest.mark.gs_login
class TPUJobTest(TestCase):
    """Tests TPUJob."""

    def test_execute_from_local(self):
        jobname = generate_job_name()
        resource = qrm_resource(get_credentials())
        atexit.register(delete_queued_tpu, jobname, resource)
        project = gcp_settings("project")
        zone = gcp_settings("zone")
        cfg: DummyRemoteTPUJob.Config = DummyRemoteTPUJob.default_config().set(
            name=jobname,
            project=project,
            zone=zone,
            max_tries=1,
            retry_interval=60,
            bundler=GCSTarBundler.default_config(),
            command="pip list",
        )
        cfg.accelerator.instance_type = FLAGS.instance_type
        out = cfg.instantiate().execute()
        self.assertIn("axlearn", out.stdout)


class DummyBastionJob(CPUJob):
    """A dummy CPU job."""

    @config_class
    class Config(CPUJob.Config):
        # Type of VM.
        vm_type: str
        # Disk size in GB.
        disk_size: int

    def _execute(self) -> subprocess.CompletedProcess:
        """Provisions and launches a command on a VM."""
        cfg: DummyBastionJob.Config = self.config
        self._bundler.bundle(cfg.name)
        create_vm(
            cfg.name,
            vm_type=cfg.vm_type,
            disk_size=cfg.disk_size,
            bundler_type=self._bundler.TYPE,
            credentials=get_credentials(),
        )
        return self._execute_remote_cmd(
            cfg.command,
            detached=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


@pytest.mark.gs_login
class CPUJobTest(TestCase):
    """Tests CPUJob."""

    def test_execute_from_local(self):
        jobname = generate_job_name()
        atexit.register(delete_vm, jobname, credentials=get_credentials())
        cfg = DummyBastionJob.default_config().set(
            name=jobname,
            project=gcp_settings("project"),
            zone=gcp_settings("zone"),
            max_tries=1,
            retry_interval=60,
            bundler=GCSTarBundler.default_config(),
            vm_type="n2-standard-2",
            disk_size=64,
            command=f"mkdir -p /tmp/{jobname} && ls /tmp/",
        )
        out = cfg.instantiate().execute()
        self.assertIn(jobname, out.stdout)


class UtilTest(TestCase):
    """Tests util functions."""

    def test_ssh_agent(self):
        old_environ = os.environ.copy()
        try:
            os.environ.pop("SSH_AGENT_PID", None)
            os.environ.pop("SSH_AUTH_SOCK", None)
            self.assertIsNone(os.getenv("SSH_AGENT_PID"))
            self.assertIsNone(os.getenv("SSH_AUTH_SOCK"))
            _start_ssh_agent()
            if sys.platform == "linux":
                self.assertRegex(
                    os.getenv("SSH_AUTH_SOCK", ""),
                    r"/tmp/ssh-.+/agent.(\d+)",
                )
            elif sys.platform == "darwin":
                self.assertRegex(
                    os.getenv("SSH_AUTH_SOCK", ""),
                    r"/var/folders/[\w/]+//ssh-.+/agent.(\d+)",
                )
            self.assertTrue(os.path.exists(os.getenv("SSH_AUTH_SOCK")))
            self.assertRegex(os.getenv("SSH_AGENT_PID", ""), r"\d+")
            _kill_ssh_agent()
            self.assertIsNone(os.getenv("SSH_AGENT_PID"))
            self.assertIsNone(os.getenv("SSH_AUTH_SOCK"))
        finally:
            os.environ.clear()
            os.environ.update(old_environ)


class TPUGKEJobTest(TestCase):
    @property
    def _mock_settings(self):
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

    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: Type[Bundler],
        reservation: Optional[str] = None,
        service_account: Optional[str] = None,
        enable_pre_provisioner: Optional[bool] = None,
    ):
        with mock_gcp_settings([job.__name__, bundler.__name__], self._mock_settings):
            fv = flags.FlagValues()
            job.TPUGKEJob.define_flags(fv)
            if reservation:
                fv.set_default("reservation", reservation)
            if service_account:
                fv.set_default("service_account", service_account)
            fv.mark_as_parsed()
            cfg = job.TPUGKEJob.from_flags(fv)
            cfg.bundler = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            cfg.accelerator.instance_type = "tpu-v4-8"
            cfg.enable_pre_provisioner = enable_pre_provisioner
            yield cfg

    @parameterized.product(
        reservation=[None, "test"],
        service_account=[None, "sa"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        wrap_bundler=[False, True],
        enable_pre_provisioner=[None, False, True],
    )
    def test_instantiate(
        self, reservation, service_account, enable_pre_provisioner, bundler_cls, wrap_bundler
    ):
        class WrappedBundler(Bundler):
            @config_class
            class Config(Bundler.Config):
                inner: Required[Bundler.Config] = REQUIRED

        with self._job_config(
            bundler_cls,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
        ) as cfg:
            self.assertEqual(cfg.reservation, reservation or self._mock_settings["gke_reservation"])
            self.assertEqual(
                cfg.service_account,
                service_account or self._mock_settings.get("k8s_service_account", "default"),
            )
            self.assertEqual(cfg.enable_pre_provisioner, enable_pre_provisioner)
            self.assertEqual(cfg.location_hint, self._mock_settings["location_hint"])
            # Should work with wrapped bundlers.
            if wrap_bundler:
                cfg.bundler = WrappedBundler.default_config().set(inner=cfg.bundler)
            # Should be instantiable.
            cfg.set(
                project="test-project",
                zone="test-zone",
                command="",
                max_tries=1,
                retry_interval=1,
                env_vars={"a": 1},
                name="test",
            )
            gke_job: job.TPUGKEJob = cfg.instantiate()
            self.assertEqual("v4-8", gke_job._tpu_type)  # pylint: disable=protected-access

    @parameterized.product(
        [
            dict(env={}, reservation=None, expect_reserved=False),
            dict(env={"BASTION_TIER": "0"}, reservation=None, expect_reserved=False),
            dict(env={"BASTION_TIER": "0"}, reservation="test-reservation", expect_reserved=True),
            dict(env={"BASTION_TIER": "1"}, reservation="test-reservation", expect_reserved=False),
            dict(env={}, reservation="test-reservation", expect_reserved=False),
        ],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        enable_ici_resiliency=[True, False, None],
        enable_pre_provisioner=[None, True, False],
        location_hint=["test-location-hint", None],
    )
    def test_build_pod(
        self,
        bundler_cls,
        expect_reserved: bool,
        enable_ici_resiliency: bool,
        env: Optional[dict] = None,
        reservation: Optional[str] = None,
        enable_pre_provisioner: Optional[bool] = None,
        location_hint: Optional[str] = None,
    ):
        with mock.patch.dict("os.environ", env), self._job_config(bundler_cls) as cfg:
            gke_job: job.TPUGKEJob = cfg.set(
                reservation=reservation,
                enable_tpu_ici_resiliency=enable_ici_resiliency,
                enable_pre_provisioner=enable_pre_provisioner,
                location_hint=location_hint,
                name="test",
            ).instantiate()
            # pylint: disable-next=protected-access
            pod = gke_job._build_pod()
            pod_spec = pod["spec"]
            node_selector = pod_spec["nodeSelector"]
            annotations = pod["metadata"]["annotations"]
            # The reservation should be used only if scheduled as tier 0.
            if expect_reserved:
                self.assertEqual(
                    reservation, node_selector.get("cloud.google.com/reservation-name", None)
                )
                self.assertNotIn("cloud.google.com/gke-spot", node_selector)
                self.assertEqual([], pod_spec.get("tolerations", []))
            else:
                self.assertEqual("true", node_selector.get("cloud.google.com/gke-spot", None))
                self.assertNotIn("cloud.google.com/reservation-name", node_selector)
                tolerations = {
                    kv["key"]: (kv["value"], kv["effect"]) for kv in pod_spec.get("tolerations", [])
                }
                self.assertEqual(
                    ("true", "NoSchedule"), tolerations.get("cloud.google.com/gke-spot", None)
                )

            self.assertEqual(len(pod_spec["containers"]), 1)
            container = pod_spec["containers"][0]
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
            container_env = {kv["name"]: kv["value"] for kv in container_env}
            if enable_ici_resiliency is not None:
                expected = "true" if enable_ici_resiliency else "false"
                self.assertEqual(
                    expected,
                    node_selector.get("cloud.google.com/gke-tpu-ici-resiliency", None),
                )
                self.assertEqual(
                    expected,
                    container_env.get("ENABLE_ICI_RESILIENCY"),
                )
            else:
                self.assertNotIn("cloud.google.com/gke-tpu-ici-resiliency", node_selector)
                self.assertNotIn("ENABLE_ICI_RESILIENCY", container_env)

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


class GPUGKEJobTest(TestCase):
    @property
    def _mock_settings(self):
        return {
            "project": "settings-project",
            "zone": "settings-zone",
            "ttl_bucket": "settings-ttl-bucket",
            "gke_cluster": "settings-cluster",
            "k8s_service_account": "settings-account",
            "docker_repo": "settings-repo",
            "default_dockerfile": "settings-dockerfile",
        }

    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: Type[Bundler],
        service_account: Optional[str] = None,
        queue: Optional[str] = None,
        num_replicas: Optional[int] = None,
        env_vars: Optional[dict] = None,
    ):
        with mock_gcp_settings([job.__name__, bundler.__name__], self._mock_settings):
            fv = flags.FlagValues()
            job.GPUGKEJob.define_flags(fv)
            if service_account:
                fv.set_default("service_account", service_account)
            if num_replicas:
                fv.set_default("num_replicas", num_replicas)
            fv.mark_as_parsed()
            cfg = job.GPUGKEJob.from_flags(fv)
            cfg.bundler = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            cfg.accelerator.instance_type = "gpu-a3-highgpu-8g-256"
            cfg.queue = queue
            cfg.command = "test-command"
            cfg.env_vars = env_vars if env_vars is not None else {}
            cfg.max_tries = 999
            yield cfg

    @parameterized.product(
        service_account=[None, "sa"],
        queue=[None, "queue-name"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        wrap_bundler=[False, True],
        num_replicas=[None, 1, 2],
        env_vars=[None, {"a": "b"}],
    )
    def test_instantiate(
        self, service_account, bundler_cls, wrap_bundler, num_replicas, env_vars, queue
    ):
        class WrappedBundler(Bundler):
            @config_class
            class Config(Bundler.Config):
                inner: Required[Bundler.Config] = REQUIRED

        with self._job_config(
            bundler_cls,
            service_account=service_account,
            env_vars=env_vars,
            num_replicas=num_replicas,
            queue=queue,
        ) as cfg:
            self.assertEqual(
                cfg.service_account,
                service_account or self._mock_settings.get("k8s_service_account", "default"),
            )
            # Should work with wrapped bundlers.
            if wrap_bundler:
                cfg.bundler = WrappedBundler.default_config().set(inner=cfg.bundler)
            # Should be instantiable.
            cfg.set(
                project="test-project",
                zone="test-zone",
                command="",
                max_tries=1,
                retry_interval=1,
                name="test",
            )
            gke_job: job.GPUGKEJob = cfg.instantiate()
            job_cfg: job.GPUGKEJob.Config = gke_job.config
            self.assertEqual("gpu-a3-highgpu-8g-256", job_cfg.accelerator.instance_type)
            if num_replicas is None:
                self.assertEqual(1, job_cfg.accelerator.num_replicas)
            else:
                self.assertEqual(num_replicas, job_cfg.accelerator.num_replicas)

    @parameterized.product(
        env_vars=[dict(), dict(XLA_FLAGS="--should-overwrite-all")],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        num_replicas=[None, 1, 32],
    )
    def test_build_pod(
        self,
        bundler_cls,
        env_vars: Optional[dict] = None,
        num_replicas: Optional[int] = None,
    ):
        with self._job_config(bundler_cls, env_vars=env_vars, num_replicas=num_replicas) as cfg:
            gke_job: job.GPUGKEJob = cfg.set(
                name="test",
            ).instantiate()
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
            expected_num_replicas = 1 if num_replicas is None else num_replicas
            self.assertEqual(
                main_container_env_vars["NUM_PROCESSES"]["value"], f"{expected_num_replicas}"
            )
            # Verify that default XLA flags can be overwritten by user.
            if env_vars and env_vars.get("XLA_FLAGS"):
                self.assertEqual(
                    main_container_env_vars["XLA_FLAGS"]["value"], env_vars["XLA_FLAGS"]
                )

    @parameterized.product(
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        queue=[None, "queue-name"],
    )
    def test_build_jobset(
        self,
        bundler_cls,
        queue: Optional[str] = None,
    ):
        with self._job_config(bundler_cls, queue=queue) as cfg:
            gke_job: job.GPUGKEJob = cfg.set(
                name="test",
            ).instantiate()
            # pylint: disable-next=protected-access
            jobset = gke_job._build_jobset()
            jobset_annotations = jobset["metadata"]["annotations"]
            self.assertEqual(jobset["metadata"]["name"], cfg.name)
            if queue is None:
                self.assertNotIn("kueue.x-k8s.io/queue-name", jobset_annotations)
            else:
                self.assertEqual(jobset_annotations["kueue.x-k8s.io/queue-name"], queue)


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    absltest.main()
