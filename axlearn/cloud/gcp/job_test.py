# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs.

    python3 -m axlearn.cloud.gcp.job_test TPUJobTest.test_execute_from_local \
        --tpu_type=v4-8 --project=my-project --zone=my-zone

    python3 -m axlearn.cloud.gcp.job_test CPUJobTest.test_execute_from_local \
        --project=my-project --zone=my-zone

"""
# pylint: disable=protected-access

import contextlib
from typing import Optional
from unittest import mock

from absl import flags, logging
from absl.testing import absltest, parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import configure_logging
from axlearn.cloud.gcp import bundler, job, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.jobset_utils_test import mock_settings
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.utils import common_flags
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
