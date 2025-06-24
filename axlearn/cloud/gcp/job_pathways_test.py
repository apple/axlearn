# Copyright © 2025 Apple Inc.

"""Unit tests of job_pathways.py."""
from typing import Optional, cast

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, job, jobset_utils, pathways_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.jobset_utils import TPUReplicatedJob
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class GKEPathwaysJobSetTest(TestCase):
    """Tests GKEPathwaysJobSet with TPU."""

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
        **kwargs,
    ) -> tuple[job.GKEJob.Config, Bundler.Config]:
        fv = flags.FlagValues()
        cfg = job.GKEJob.default_config().set(
            builder=pathways_utils.PathwaysReplicatedJob.default_config()
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
        self.assertIsInstance(cfg.builder, pathways_utils.PathwaysReplicatedJob.Config)
        cfg.builder = cast(pathways_utils.PathwaysReplicatedJob.Config, cfg.builder)
        inner_builder = cfg.builder.inner
        self.assertIsInstance(inner_builder, TPUReplicatedJob.Config)

        self.assertEqual(cfg.name, cfg.builder.name)
        self.assertEqual(cfg.project, self._settings["project"])
        self.assertEqual(cfg.zone, self._settings["zone"])
        self.assertEqual(
            inner_builder.reservation, reservation or self._settings["gke_reservation"]
        )
        self.assertEqual(
            inner_builder.service_account,
            service_account or self._settings.get("k8s_service_account", "default"),
        )
        self.assertEqual(inner_builder.location_hint, self._settings["location_hint"])
        self.assertEqual(inner_builder.enable_pre_provisioner, enable_pre_provisioner)
        # Should work with wrapped bundlers.
        if wrap_bundler:
            bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)

        if enable_pre_provisioner:
            with self.assertRaises(NotImplementedError):
                cfg.instantiate(bundler=bundler_cfg.instantiate())
        else:
            gke_job = cfg.instantiate(bundler=bundler_cfg.instantiate())
            # pylint: disable-next=protected-access
            self.assertEqual("v4-8", gke_job._builder._tpu_type)  # pytype: disable=attribute-error

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
            instance_type="v4-8",
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

        replicated_jobs = jobset["spec"]["replicatedJobs"]
        for replicated_job in replicated_jobs:
            self.assertIn(
                replicated_job["name"],
                [
                    pathways_utils._PATHWAYS_HEAD_REPLICATED_JOB_NAME,  # pylint: disable=protected-access
                    pathways_utils._PATHWAYS_WORKER_REPLICATED_JOB_NAME,  # pylint: disable=protected-access
                ],
            )
