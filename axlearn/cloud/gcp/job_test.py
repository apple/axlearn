# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs.

    python3 -m axlearn.cloud.gcp.job_test TPUJobTest.test_execute_from_local \
        --tpu_type=v4-8 --project=my-project --zone=my-zone

    python3 -m axlearn.cloud.gcp.job_test CPUJobTest.test_execute_from_local \
        --project=my-project --zone=my-zone

"""
# pylint: disable=protected-access

import atexit
import contextlib
import os
import subprocess
import sys
from typing import Optional, Union
from unittest import mock

import pytest
from absl import flags, logging
from absl.testing import absltest, parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import configure_logging, generate_job_name
from axlearn.cloud.gcp import bundler, job, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler, GCSTarBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import CPUJob, TPUQRMJob, _kill_ssh_agent, _start_ssh_agent
from axlearn.cloud.gcp.jobset_utils_test import mock_settings
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
        with mock_gcp_settings(
            [job.__name__, jobset_utils.__name__, bundler.__name__], mock_settings()
        ):
            fv = flags.FlagValues()
            job.TPUGKEJob.define_flags(fv)
            if reservation:
                fv.set_default("reservation", reservation)
            if service_account:
                fv.set_default("service_account", service_account)
            if host_mount_spec:
                fv.set_default("host_mount_spec", host_mount_spec)
            if gcsfuse_mount_spec:
                fv.set_default("gcsfuse_mount_spec", gcsfuse_mount_spec)
            fv.mark_as_parsed()
            cfg = job.TPUGKEJob.from_flags(fv)
            cfg.bundler = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            cfg.accelerator.instance_type = "tpu-v4-8"
            cfg.enable_pre_provisioner = enable_pre_provisioner
            cfg.builder.priority_class = priority_class
            yield cfg

    @parameterized.product(
        reservation=[None, "test"],
        service_account=[None, "sa"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        wrap_bundler=[False, True],
        enable_pre_provisioner=[None, False, True],
    )
    def test_instantiate(
        self,
        reservation,
        service_account,
        enable_pre_provisioner,
        bundler_cls,
        wrap_bundler,
    ):
        class WrappedBundler(Bundler):
            @config_class
            class Config(Bundler.Config):
                inner: Required[Bundler.Config] = REQUIRED

        settings = mock_settings()
        with self._job_config(
            bundler_cls,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
        ) as cfg:
            self.assertEqual(cfg.builder.reservation, reservation or settings["gke_reservation"])
            self.assertEqual(
                cfg.service_account,
                service_account or settings.get("k8s_service_account", "default"),
            )
            self.assertEqual(cfg.enable_pre_provisioner, enable_pre_provisioner)
            self.assertEqual(cfg.builder.location_hint, settings["location_hint"])
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
                env_vars={"a": "1"},
                output_dir="FAKE",
            )
            gke_job: job.TPUGKEJob = cfg.instantiate()

            # Instantiating should propagate fields.
            final_config: job.TPUGKEJob.Config = gke_job.config
            inner_config: jobset_utils.TPUReplicatedJob.Config = gke_job._builder.config
            for key, value in final_config.items():
                if key not in ("klass", "bundler") and key in inner_config.keys():
                    self.assertEqual(value, getattr(inner_config, key), msg=key)
            self.assertEqual("v4-8", gke_job._builder._tpu_type)  # pytype: disable=attribute-error


class GPUGKEJobTest(TestCase):
    @contextlib.contextmanager
    def _job_config(
        self,
        bundler_cls: type[Bundler],
        service_account: Optional[str] = None,
        queue: Optional[str] = None,
        num_replicas: Optional[int] = None,
        env_vars: Optional[dict] = None,
    ):
        with mock_gcp_settings(
            [job.__name__, jobset_utils.__name__, bundler.__name__], mock_settings()
        ):
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

        settings = mock_settings()
        with self._job_config(
            bundler_cls,
            service_account=service_account,
            env_vars=env_vars,
            num_replicas=num_replicas,
            queue=queue,
        ) as cfg:
            self.assertEqual(
                cfg.service_account,
                service_account or settings.get("k8s_service_account", "default"),
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
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        queue=[None, "queue-name"],
    )
    def test_build_jobset(
        self,
        bundler_cls,
        queue: Optional[str] = None,
    ):
        with self._job_config(bundler_cls, queue=queue) as cfg:
            gke_job: job.GPUGKEJob = cfg.set(name="test").instantiate()
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
