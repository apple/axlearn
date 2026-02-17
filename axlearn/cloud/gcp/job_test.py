# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs."""
# pylint: disable=protected-access

import json
from typing import Optional, cast
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, job, jobset_utils, pathways_utils
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
            gke_job = cfg.instantiate(bundler=mock.create_autospec(Bundler))
            gke_job._delete()  # pylint: disable=protected-access
            mock_delete.assert_called()

    @parameterized.product(
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        labels=[None, {"env": "tpu-test"}, {"team": "research", "experiment": "training"}],
    )
    def test_build_jobset_labels_tpu(
        self,
        bundler_cls,
        labels: Optional[dict] = None,
    ):
        cfg, bundler_cfg = self._job_config(
            command="test-command",
            bundler_cls=bundler_cls,
        )
        # Set labels on the config
        cfg = cfg.set(labels=labels)
        gke_job: job.GKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable-next=protected-access
        jobset = gke_job._build_jobset()
        jobset_metadata = jobset["metadata"]
        jobset_labels = jobset_metadata.get("labels", {})

        # Test basic metadata
        self.assertEqual(jobset_metadata["name"], cfg.name)

        # Test labels
        if labels is None:
            # When labels is None, labels dict should be empty
            self.assertEqual(jobset_labels, {})
        else:
            # When labels is provided, they should be present in metadata
            for key, value in labels.items():
                self.assertIn(key, jobset_labels)
                self.assertEqual(jobset_labels[key], value)

    @parameterized.parameters(
        # Single job with correct assignments
        dict(
            jobs=[("tpu-worker", 2, "tpu7x", "4x4x8")],
            topology_assignments=[["sb-1", "sb-2"], ["sb-3", "sb-4"]],
            expected={"tpu-worker": [["sb-1", "sb-2"], ["sb-3", "sb-4"]]},
        ),
        # Multiple TPU jobs
        dict(
            jobs=[
                ("trainer", 1, "tpu7x", "4x4x8"),  # 7x-256
                ("evaluator", 1, "tpu7x", "4x4x4"),  # 7x-128
            ],
            topology_assignments=[["sb-1", "sb-2"], ["sb-3"]],
            expected={"trainer": [["sb-1", "sb-2"]], "evaluator": [["sb-3"]]},
        ),
        # Multiple TPU jobs, with multiple replicas
        dict(
            jobs=[
                ("trainer", 2, "tpu7x", "4x4x8"),  # 7x-256
                ("evaluator", 3, "tpu7x", "4x4x4"),  # 7x-128
            ],
            topology_assignments=[["sb-1", "sb-2"], ["sb-3", "sb-4"], ["sb-5"], ["sb-6"], ["sb-7"]],
            expected={
                "trainer": [["sb-1", "sb-2"], ["sb-3", "sb-4"]],
                "evaluator": [["sb-5"], ["sb-6"], ["sb-7"]],
            },
        ),
        # Multiple TPU jobs, with multiple replicas, assignment reversed
        dict(
            jobs=[
                ("trainer", 2, "tpu7x", "4x4x8"),  # 7x-256
                ("evaluator", 3, "tpu7x", "4x4x4"),  # 7x-128
            ],
            topology_assignments=[["sb-5"], ["sb-6"], ["sb-7"], ["sb-1", "sb-2"], ["sb-3", "sb-4"]],
            expected={
                "trainer": [["sb-1", "sb-2"], ["sb-3", "sb-4"]],
                "evaluator": [["sb-5"], ["sb-6"], ["sb-7"]],
            },
        ),
        # Unsupported TPU version
        dict(
            jobs=[("v5p-job", 1, "tpu-v5p-slice", "4x4x8")],
            topology_assignments=[["sb-1"]],
            expected=(ValueError, "TPU version 'v5p'.* does not support subblock super slicing"),
        ),
        # Insufficient assignments
        dict(
            jobs=[("tpu-worker", 2, "tpu7x", "4x4x8")],
            topology_assignments=[["sb-1", "sb-2"]],  # Only 1, but needs 2 replicas
            expected=(ValueError, "Could not find unused topology assignment with 2 subblock"),
        ),
        # Wrong subblock count
        dict(
            jobs=[("tpu-worker", 1, "tpu7x", "4x4x8")],  # Needs 2 subblocks
            topology_assignments=[["sb-1"]],  # Only 1 subblock
            expected=(ValueError, "Could not find unused topology assignment with 2 subblock"),
        ),
        # Mixed CPU/TPU jobs
        dict(
            jobs=[
                ("cpu-coordinator", 1, None, None),  # CPU job
                ("tpu-worker", 1, "tpu7x", "4x4x4"),  # TPU job
            ],
            topology_assignments=[["sb-1"]],
            expected={"tpu-worker": [["sb-1"]]},
        ),
        # Unknown system
        dict(
            jobs=[("invalid-job", 1, "unknown-tpu", "9x9x9")],
            topology_assignments=[["sb-1"]],
            expected=(
                ValueError,
                "Could not find system characteristics.*accelerator"
                "='unknown-tpu'.*topology='9x9x9'",
            ),
        ),
        # Skip wrong sizes
        dict(
            jobs=[("tpu-worker", 2, "tpu7x", "4x4x8")],  # Needs 2 subblocks
            topology_assignments=[
                ["sb-wrong"],  # Wrong size (1), skip
                ["sb-1", "sb-2"],  # Correct (2), use for replica 0
                ["sb-wrong-2"],  # Wrong size (1), skip
                ["sb-3", "sb-4"],  # Correct (2), use for replica 1
            ],
            expected={"tpu-worker": [["sb-1", "sb-2"], ["sb-3", "sb-4"]]},
        ),
        # Correct assignment at end
        dict(
            jobs=[("tpu-worker", 1, "tpu7x", "4x4x8")],  # Needs 2 subblocks
            topology_assignments=[
                ["sb-1"],  # Wrong size (1), skip
                ["sb-2", "sb-3", "sb-4"],  # Wrong size (3), skip
                ["sb-5"],  # Wrong size (1), skip
                ["sb-6", "sb-7"],  # Correct size (2), use this
            ],
            expected={"tpu-worker": [["sb-6", "sb-7"]]},
        ),
        # Mixed sizes multi-job
        dict(
            jobs=[
                ("small-job", 1, "tpu7x", "4x4x4"),  # 7x-128: needs 1 subblock
                ("large-job", 1, "tpu7x", "4x4x8"),  # 7x-256: needs 2 subblocks
            ],
            topology_assignments=[
                ["sb-1", "sb-2"],  # Size 2, for large-job
                ["sb-3"],  # Size 1, for small-job
                ["sb-4", "sb-5", "sb-6"],  # Size 3, unused
            ],
            expected={"small-job": [["sb-3"]], "large-job": [["sb-1", "sb-2"]]},
        ),
    )
    def test_get_tpu_replicated_job_topology_selection(
        self, jobs: list, topology_assignments: list, expected
    ):
        """Test topology selection for TPU replicated jobs.

        Args:
            jobs: List of job specs as tuples (name, replicas, gke_accelerator, topology).
                For CPU jobs, gke_accelerator and topology should be None.
            topology_assignments: List of subblock assignments.
            expected: Either a dict mapping job names to assignments (success case),
                or a tuple of (exception_class, error_regex) for error cases.
        """

        def _make_tpu_job(name: str, replicas: int, gke_accelerator: str, topology: str):
            """Helper to build a TPU replicated job spec."""
            return {
                "name": name,
                "replicas": replicas,
                "template": {
                    "spec": {
                        "template": {
                            "spec": {
                                "nodeSelector": {
                                    "cloud.google.com/gke-tpu-accelerator": gke_accelerator,
                                    "cloud.google.com/gke-tpu-topology": topology,
                                }
                            }
                        }
                    }
                },
            }

        def _make_cpu_job(name: str, replicas: int):
            """Helper to build a CPU replicated job spec."""
            return {
                "name": name,
                "replicas": replicas,
                "template": {
                    "spec": {
                        "template": {
                            "spec": {"nodeSelector": {"axlearn/nodepool_type": "workload"}}
                        }
                    }
                },
            }

        # Build replicated jobs from specs
        replicated_jobs = []
        for name, replicas, gke_accelerator, topology in jobs:
            if gke_accelerator is None:
                # CPU job
                replicated_jobs.append(_make_cpu_job(name, replicas))
            else:
                # TPU job
                replicated_jobs.append(_make_tpu_job(name, replicas, gke_accelerator, topology))

        cfg, bundler_cfg = self._job_config(
            command="test-command",
            bundler_cls=CloudBuildBundler,
        )
        gke_job: job.GKEJob = cfg.instantiate(bundler=bundler_cfg.instantiate())

        if isinstance(expected, tuple):
            # Error case
            exception_class, error_regex = expected
            # pylint: disable-next=protected-access
            with self.assertRaisesRegex(exception_class, error_regex):
                gke_job._get_tpu_replicated_job_topology_selection(
                    replicated_jobs, topology_assignments
                )
        else:
            # Success case
            # pylint: disable-next=protected-access
            result = gke_job._get_tpu_replicated_job_topology_selection(
                replicated_jobs, topology_assignments
            )
            self.assertEqual(expected, result)


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
        labels=[None, {"env": "gpu-test"}, {"team": "ml-gpu", "priority": "high"}],
    )
    def test_build_jobset(
        self,
        bundler_cls,
        queue: Optional[str] = None,
        labels: Optional[dict] = None,
    ):
        cfg, bundler_cfg = self._job_config(
            command="",
            bundler_cls=bundler_cls,
            instance_type="gpu-a3-highgpu-8g-256",
            queue=queue,
        )
        # Set labels on the config
        cfg = cfg.set(labels=labels)
        gke_job: job.GKEJob = cfg.set(name="test").instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable-next=protected-access
        jobset = gke_job._build_jobset()
        jobset_metadata = jobset["metadata"]
        jobset_annotations = jobset_metadata["annotations"]
        jobset_labels = jobset_metadata.get("labels", {})

        # Test basic metadata
        self.assertEqual(jobset_metadata["name"], cfg.name)

        # Test queue annotations
        if queue is None:
            self.assertNotIn("kueue.x-k8s.io/queue-name", jobset_annotations)
        else:
            self.assertEqual(jobset_annotations["kueue.x-k8s.io/queue-name"], queue)

        # Test labels
        if labels is None:
            # When labels is None, labels dict should be empty
            self.assertEqual(jobset_labels, {})
        else:
            # When labels is provided, they should be present in metadata
            for key, value in labels.items():
                self.assertIn(key, jobset_labels)
                self.assertEqual(jobset_labels[key], value)


class TPUGKELeaderWorkerSetTest(TestCase):
    """Tests GKELeaderWorkerSet with TPU."""

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
    ) -> tuple[job.GKELeaderWorkerSet.Config, Bundler.Config]:
        fv = flags.FlagValues()
        cfg = job.GKELeaderWorkerSet.default_config().set(
            builder=pathways_utils.PathwaysLeaderWorkerTemplate.default_config()
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
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        wrap_bundler=[False, True],
    )
    def test_instantiate(
        self,
        reservation,
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
            num_replicas=1,
        )

        self.assertIsInstance(cfg.builder, pathways_utils.PathwaysLeaderWorkerTemplate.Config)
        cfg.builder = cast(pathways_utils.PathwaysLeaderWorkerTemplate.Config, cfg.builder)

        self.assertEqual(cfg.name, cfg.builder.name)
        self.assertEqual(cfg.project, self._settings["project"])
        self.assertEqual(cfg.zone, self._settings["zone"])
        self.assertEqual(
            cfg.builder.inner.reservation, reservation or self._settings["gke_reservation"]
        )
        self.assertEqual(cfg.num_replicas, 1)
        # Should work with wrapped bundlers.
        if wrap_bundler:
            bundler_cfg = WrappedBundler.default_config().set(inner=bundler_cfg)
        gke_job = cfg.instantiate(bundler=bundler_cfg.instantiate())
        self.assertEqual("v4-8", gke_job._builder._tpu_type)

    def test_delete(self):
        patch_delete = mock.patch(f"{job.__name__}.delete_k8s_leaderworkerset")
        with patch_delete as mock_delete:
            cfg, _ = self._job_config(command="test-command", bundler_cls=CloudBuildBundler)
            gke_job = cfg.instantiate(bundler=mock.create_autospec(Bundler))
            gke_job._delete()  # pylint: disable=protected-access
            mock_delete.assert_called()

    @parameterized.parameters(
        # Test when auto provisioning is enabled with topology assignment
        dict(
            enable_tpu_slice_auto_provisioning=True,
            topology_assignment=[["subblock-1", "subblock-2"]],
            expect_label=True,
            expect_annotation=True,
        ),
        # Test when auto provisioning is disabled
        dict(
            enable_tpu_slice_auto_provisioning=False,
            topology_assignment=[["subblock-1", "subblock-2"]],
            expect_label=False,
            expect_annotation=False,
        ),
        # Test when auto provisioning is None (not set)
        dict(
            enable_tpu_slice_auto_provisioning=None,
            topology_assignment=[["subblock-1", "subblock-2"]],
            expect_label=False,
            expect_annotation=False,
        ),
        # Test when auto provisioning is enabled but no topology assignment
        dict(
            enable_tpu_slice_auto_provisioning=True,
            topology_assignment=None,
            expect_label=False,
            expect_annotation=False,
        ),
    )
    def test_build_leaderworkerset(
        self,
        enable_tpu_slice_auto_provisioning,
        topology_assignment,
        expect_label,
        expect_annotation,
    ):
        """Test _build_leaderworkerset with enable_tpu_slice_auto_provisioning."""
        cfg, bundler_cfg = self._job_config(
            command="test-command",
            bundler_cls=CloudBuildBundler,
            enable_tpu_slice_auto_provisioning=enable_tpu_slice_auto_provisioning,
        )

        # Mock the builder to return a simple leader worker template
        mock_leader_worker_template = {
            "size": 8,
            "workerTemplate": {
                "metadata": {"labels": {"test-label": "test-value"}},
                "spec": {"containers": []},
            },
        }

        # Create a mock builder that returns our mock template
        mock_builder = mock.Mock()
        mock_builder.return_value = mock_leader_worker_template

        # Create the GKE job instance first
        gke_job = cfg.instantiate(bundler=bundler_cfg.instantiate())

        # Replace the builder with our mock (this is what we're testing)
        gke_job._builder = mock_builder

        # Mock get_topology_assignment
        with mock.patch(
            f"{job.__name__}.get_topology_assignment",
            return_value=topology_assignment,
        ):
            # Build the leaderworkerset
            lws_spec = gke_job._build_leaderworkerset()

            # Check metadata
            self.assertIn("metadata", lws_spec)
            self.assertIn("name", lws_spec["metadata"])
            self.assertEqual(cfg.name, lws_spec["metadata"]["name"])

            # Check labels
            labels = lws_spec["metadata"].get("labels", {})
            slice_auto_provisioning_label = (
                "tpu-provisioner.cloud.google.com/slice-autoprovisioning"
            )
            if expect_label:
                self.assertIn(slice_auto_provisioning_label, labels)
                self.assertEqual("async", labels[slice_auto_provisioning_label])
            else:
                self.assertNotIn(slice_auto_provisioning_label, labels)

            # Check annotations
            annotations = lws_spec["metadata"].get("annotations", {})
            slice_selection_annotation = "tpu-provisioner.cloud.google.com/slice-selection"
            if expect_annotation:
                self.assertIn(slice_selection_annotation, annotations)
                slice_selection = json.loads(annotations[slice_selection_annotation])
                self.assertIn("workers", slice_selection)
                self.assertEqual(topology_assignment, slice_selection["workers"])
            else:
                self.assertNotIn(slice_selection_annotation, annotations)

            # Verify exclusive topology annotations are removed when auto provisioning
            if expect_annotation:
                self.assertNotIn(
                    "leaderworkerset.sigs.k8s.io/subgroup-exclusive-topology",
                    annotations,
                )

            # Check spec
            self.assertIn("spec", lws_spec)
            self.assertIn("replicas", lws_spec["spec"])
            self.assertIn("leaderWorkerTemplate", lws_spec["spec"])

    @parameterized.product(
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
        labels=[None, {"env": "tpu-test"}, {"team": "research", "experiment": "training"}],
    )
    def test_build_leaderworkerset_labels(
        self,
        bundler_cls,
        labels: Optional[dict] = None,
    ):
        """Test that labels are properly set in LeaderWorkerSet metadata."""
        cfg, bundler_cfg = self._job_config(
            command="test-command",
            bundler_cls=bundler_cls,
        )
        # Set labels on the config
        cfg = cfg.set(labels=labels)
        gke_job: job.GKELeaderWorkerSet = cfg.instantiate(bundler=bundler_cfg.instantiate())
        # pylint: disable-next=protected-access
        lws_spec = gke_job._build_leaderworkerset()
        lws_metadata = lws_spec["metadata"]
        lws_labels = lws_metadata.get("labels", {})

        # Test basic metadata
        self.assertEqual(lws_metadata["name"], cfg.name)

        # Test labels
        if labels is None:
            # When labels is None, labels dict should be empty
            self.assertEqual(lws_labels, {})
        else:
            # When labels is provided, they should be present in metadata
            for key, value in labels.items():
                self.assertIn(key, lws_labels)
                self.assertEqual(lws_labels[key], value)
