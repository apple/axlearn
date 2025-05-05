# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs."""
# pylint: disable=protected-access

from typing import Optional, cast
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, job, jobset_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class TPUGKEJobTest(TestCase):
    """Tests GKEJob with TPU."""

    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        with mock_gcp_settings(
            [jobset_utils.__name__, bundler.__name__],
            settings=self._settings,
        ):
            return super().run(result)

    def _job_config(
        self,
        *,
        command: str,
        bundler_cls: type[Bundler],
        priority_class: Optional[str] = None,
        **kwargs,
    ) -> tuple[job.GKEJob.Config, Bundler.Config]:
        fv = flags.FlagValues()
        cfg = job.GKEJob.default_config().set(
            builder=jobset_utils.TPUReplicatedJob.default_config()
        )
        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                # Use setattr rather than set_default to set flags.
                setattr(fv, key, value)
        fv.name = "fake-name"
        fv.output_dir = "FAKE"
        fv.instance_type = "tpu-v4-8"
        fv.mark_as_parsed()
        from_flags(cfg, fv, command=command)
        # Test that retries are configured on fv by default.
        self.assertIsNotNone(fv["max_tries"].default)
        self.assertIsNotNone(fv["retry_interval"].default)
        cfg.builder.priority_class = priority_class
        bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
        return cfg, bundler_cfg

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

        cfg, bundler_cfg = self._job_config(
            command="test-command",
            bundler_cls=bundler_cls,
            reservation=reservation,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
        )
        self.assertIsInstance(cfg.builder, jobset_utils.TPUReplicatedJob.Config)
        cfg.builder = cast(jobset_utils.TPUReplicatedJob.Config, cfg.builder)

        self.assertEqual(cfg.name, cfg.builder.name)
        self.assertEqual(cfg.project, self._settings["project"])
        self.assertEqual(cfg.zone, self._settings["zone"])
        self.assertEqual(cfg.builder.reservation, reservation or self._settings["gke_reservation"])
        self.assertEqual(
            cfg.builder.service_account,
            service_account or self._settings.get("k8s_service_account", "default"),
        )
        self.assertEqual(cfg.builder.location_hint, self._settings["location_hint"])
        self.assertEqual(cfg.builder.enable_pre_provisioner, enable_pre_provisioner)
        # Should work with wrapped bundlers.
        if wrap_bundler:
            bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)
        gke_job = cfg.instantiate(bundler=bundler_cfg.instantiate())
        self.assertEqual("v4-8", gke_job._builder._tpu_type)  # pytype: disable=attribute-error

    def test_delete(self):
        patch_delete = mock.patch(f"{job.__name__}.delete_k8s_jobset")
        with patch_delete as mock_delete:
            cfg, _ = self._job_config(command="test-command", bundler_cls=CloudBuildBundler)
            gke_job = cfg.instantiate(bundler=mock.Mock())
            gke_job._delete()  # pylint: disable=protected-access
            mock_delete.assert_called()


class GPUGKEJobTest(TestCase):
    """Tests GKEJob with GPUs."""

    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        with mock_gcp_settings(
            [jobset_utils.__name__, bundler.__name__],
            settings=self._settings,
        ):
            return super().run(result)

    def _job_config(
        self, *, command: str, bundler_cls: type[Bundler], **kwargs
    ) -> tuple[job.GKEJob.Config, Bundler.Config]:
        fv = flags.FlagValues()
        cfg = job.GKEJob.default_config().set(
            builder=jobset_utils.A3HighReplicatedJob.default_config()
        )
        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                # Use setattr rather than set_default to set flags.
                setattr(fv, key, value)
        fv.mark_as_parsed()
        cfg = from_flags(cfg, fv, command=command)
        bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
        return cfg, bundler_cfg

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
        cfg, bundler_cfg = self._job_config(
            command=command,
            bundler_cls=bundler_cls,
            instance_type="gpu-a3-highgpu-8g-256",
            service_account=service_account,
            num_replicas=num_replicas,
            queue=queue,
        )
        self.assertEqual(
            cfg.builder.service_account,
            service_account or settings.get("k8s_service_account", "default"),
        )
        # Should work with wrapped bundlers.
        if wrap_bundler:
            bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)
        # Should be instantiable.
        gke_job: job.GKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
        job_cfg: job.GKEJob.Config = gke_job.config

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
        cfg, bundler_cfg = self._job_config(
            command="",
            bundler_cls=bundler_cls,
            instance_type="gpu-a3-highgpu-8g-256",
            queue=queue,
        )
        gke_job: job.GKEJob = cfg.set(name="test").instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable-next=protected-access
        jobset = gke_job._build_jobset()
        jobset_annotations = jobset["metadata"]["annotations"]
        self.assertEqual(jobset["metadata"]["name"], cfg.name)
        if queue is None:
            self.assertNotIn("kueue.x-k8s.io/queue-name", jobset_annotations)
        else:
            self.assertEqual(jobset_annotations["kueue.x-k8s.io/queue-name"], queue)
