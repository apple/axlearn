# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs.

    python3 -m axlearn.cloud.gcp.job_test TPUJobTest.test_execute_from_local \
        --tpu_type=v4-8 --project=my-project --zone=my-zone

    python3 -m axlearn.cloud.gcp.job_test CPUJobTest.test_execute_from_local \
        --project=my-project --zone=my-zone

"""
# pylint: disable=protected-access

import contextlib
from typing import Optional, cast
from unittest import mock

from absl import flags, logging
from absl.testing import absltest, parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import configure_logging
from axlearn.cloud.gcp import bundler, job, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
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
        *,
        command: str,
        bundler_cls: type[Bundler],
        priority_class: Optional[str] = None,
        **kwargs,
    ):
        settings = default_mock_settings()
        with mock_gcp_settings([job.__name__, jobset_utils.__name__, bundler.__name__], settings):
            fv = flags.FlagValues()
            job.TPUGKEJob.define_flags(fv)
            for key, value in kwargs.items():
                if value is not None:
                    # Use setattr rather than set_default to set flags.
                    setattr(fv, key, value)
            fv.output_dir = "FAKE"
            fv.instance_type = "tpu-v4-8"
            fv.mark_as_parsed()
            cfg = job.TPUGKEJob.from_flags(fv, command=command)
            # Test that retries are configured on fv by default.
            self.assertIsNotNone(fv["max_tries"].default)
            self.assertIsNotNone(fv["retry_interval"].default)
            cfg.builder.priority_class = priority_class
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg, settings

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
        bundler_cls: type[Bundler],
        wrap_bundler,
    ):
        class WrappedBundler(Bundler):
            @config_class
            class Config(Bundler.Config):
                inner: Required[Bundler.Config] = REQUIRED

        with self._job_config(
            command="test-command",
            bundler_cls=bundler_cls,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (cfg, bundler_cfg, settings):
            self.assertIsInstance(cfg.builder, jobset_utils.TPUReplicatedJob.Config)
            cfg.builder = cast(jobset_utils.TPUReplicatedJob.Config, cfg.builder)

            self.assertEqual(cfg.name, cfg.builder.name)
            self.assertEqual(cfg.project, settings["project"])
            self.assertEqual(cfg.zone, settings["zone"])
            self.assertEqual(cfg.builder.reservation, reservation or settings["gke_reservation"])
            self.assertEqual(
                cfg.service_account,
                service_account or settings.get("k8s_service_account", "default"),
            )
            self.assertEqual(cfg.builder.location_hint, settings["location_hint"])
            self.assertEqual(cfg.builder.enable_pre_provisioner, enable_pre_provisioner)
            # Should work with wrapped bundlers.
            if wrap_bundler:
                bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)
            gke_job: job.TPUGKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
            self.assertEqual("v4-8", gke_job._builder._tpu_type)  # pytype: disable=attribute-error


class GPUGKEJobTest(TestCase):
    @contextlib.contextmanager
    def _job_config(self, *, command: str, bundler_cls: type[Bundler], **kwargs):
        with mock_gcp_settings(
            [job.__name__, jobset_utils.__name__, bundler.__name__], default_mock_settings()
        ):
            fv = flags.FlagValues()
            job.GPUGKEJob.define_flags(fv)
            for key, value in kwargs.items():
                if value is not None:
                    # Use setattr rather than set_default to set flags.
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = job.GPUGKEJob.from_flags(fv, command=command)
            cfg.max_tries = 1
            cfg.retry_interval = 1
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.product(
        service_account=[None, "sa"],
        queue=[None, "queue-name"],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        wrap_bundler=[False, True],
        num_replicas=[None, 1, 2],
        instance_type=["gpu-a3-highgpu-8g-256"],
    )
    def test_instantiate(
        self, *, service_account, bundler_cls, wrap_bundler, num_replicas, queue, instance_type
    ):
        class WrappedBundler(Bundler):
            @config_class
            class Config(Bundler.Config):
                inner: Required[Bundler.Config] = REQUIRED

        command = "test-command"
        settings = default_mock_settings()
        with self._job_config(
            command=command,
            bundler_cls=bundler_cls,
            instance_type="gpu-a3-highgpu-8g-256",
            service_account=service_account,
            num_replicas=num_replicas,
            queue=queue,
        ) as (cfg, bundler_cfg):
            self.assertEqual(
                cfg.service_account,
                service_account or settings.get("k8s_service_account", "default"),
            )
            # Should work with wrapped bundlers.
            if wrap_bundler:
                bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)
            # Should be instantiable.
            gke_job: job.GPUGKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
            job_cfg: job.GPUGKEJob.Config = gke_job.config

            # Command/instance_type should be read by the builder.
            self.assertEqual(command, job_cfg.builder.command)
            self.assertEqual(instance_type, job_cfg.builder.accelerator.instance_type)
            self.assertEqual(num_replicas or 1, job_cfg.builder.accelerator.num_replicas)

    @parameterized.product(
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        queue=[None, "queue-name"],
    )
    def test_build_jobset(
        self,
        bundler_cls,
        queue: Optional[str] = None,
    ):
        with self._job_config(
            command="",
            bundler_cls=bundler_cls,
            instance_type="gpu-a3-highgpu-8g-256",
            queue=queue,
        ) as (cfg, bundler_cfg):
            gke_job: job.GPUGKEJob = cfg.set(name="test").instantiate(
                bundler=bundler_cfg.instantiate()
            )
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
