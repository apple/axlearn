# Copyright © 2024 Apple Inc.

"""Tests GKERunnerJob."""

# pylint: disable=no-self-use,protected-access
from collections.abc import Sequence
from typing import Optional
from unittest import mock

import kubernetes as k8s
import requests
from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import BASTION_JOB_VERSION_ENV_VAR
from axlearn.cloud.common.utils import FlagConfigurable, define_flags, from_flags
from axlearn.cloud.gcp import bundler, node_pool_provisioner
from axlearn.cloud.gcp.job_flink import FlinkJobStatus
from axlearn.cloud.gcp.jobset_utils import BASTION_JOB_VERSION_LABEL, TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.runners import gke as runner_gke
from axlearn.cloud.gcp.runners import named_runner_configs
from axlearn.cloud.gcp.runners.gke import (
    GKERunnerJob,
    _infer_job_count,
    _infer_job_version,
    _infer_reservation,
)
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.config import REQUIRED, Required, config_class


def _mock_replicated_jobs(
    reservations: Sequence[str],
    bastion_job_version: Optional[int] = None,
    num_replicas: Optional[int] = None,
):
    job_version_label = (
        {"metadata": {"labels": {BASTION_JOB_VERSION_LABEL: str(bastion_job_version)}}}
        if bastion_job_version
        else {}
    )
    if num_replicas:
        replicas_field = {"replicas": num_replicas}
    else:
        replicas_field = {}

    return [
        {
            "name": "test-job",
            **replicas_field,
            "template": {
                "spec": {
                    "template": {
                        "spec": {
                            "nodeSelector": (
                                {"cloud.google.com/reservation-name": reservation}
                                if reservation != "spot"
                                else {"cloud.google.com/gke-spot": "true"}
                            )
                        },
                        **job_version_label,
                    },
                },
            },
        }
        for reservation in reservations
    ]


# TODO(markblee): Consolidate {TPU,GPU}GKERunnerTests.
class GPUGKERunnerJobTest(parameterized.TestCase):
    """Tests GPUGKERunnerJob."""

    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        mock_user = mock.patch("os.environ", {"USER": "test"})
        with (
            mock_user,
            mock_gcp_settings(
                [runner_gke.__name__, bundler.__name__, node_pool_provisioner.__name__],
                settings=self._settings,
            ),
        ):
            return super().run(result)

    def _job_config(self, *, command: str, **kwargs) -> GKERunnerJob.Config:
        fv = flags.FlagValues()
        cfg = named_runner_configs("gke_gpu_a3_high_single")
        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                setattr(fv, key, value)
        fv.set_default("instance_type", "gpu-a3-highgpu-8g-256")
        fv.mark_as_parsed()
        return from_flags(cfg, fv, command=command)

    @parameterized.product(
        name=[None, "test-name"],
        cluster=[None, "test-cluster"],
        gcsfuse_mount_spec=[None, ["gcs_path=my-test-path"]],
    )
    def test_from_flags(self, name, cluster, gcsfuse_mount_spec):
        cfg = self._job_config(
            command="test-command",
            name=name,
            cluster=cluster,
            gcsfuse_mount_spec=gcsfuse_mount_spec,
        )
        if name:
            self.assertEqual(cfg.name, name)
        else:
            self.assertIsNotNone(cfg.name)
        self.assertEqual(cfg.cluster, cluster or self._settings["gke_cluster"])
        self.assertEqual(cfg.inner.builder.name, cfg.name)
        self.assertEqual(cfg.inner.builder.command, "test-command")
        if gcsfuse_mount_spec:
            fuse = cfg.inner.builder.gcsfuse_mount
            self.assertEqual(fuse.gcs_path, "my-test-path")

    @parameterized.product(
        status=[
            GKERunnerJob.Status.FAILED,
            GKERunnerJob.Status.SUCCEEDED,
            GKERunnerJob.Status.COMPLETED,
        ],
    )
    def test_exit(self, status):
        cfg = self._job_config(command="", name="test-name", cluster="test-cluster")
        job: GKERunnerJob = cfg.instantiate(bundler=mock.Mock())
        with mock.patch.multiple(
            job, _get_status=mock.Mock(return_value=status), _delete=mock.DEFAULT
        ):
            job._execute()

    def test_delete(self):
        cfg = self._job_config(command="", name="test-name", cluster="test-cluster")
        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())
        with mock.patch.multiple(job, _inner=mock.DEFAULT, _pre_provisioner=mock.DEFAULT):
            job._delete()
            job._inner._delete.assert_called()  # pytype: disable=attribute-error

    def test_start(self):
        cfg = self._job_config(
            command="",
            name="test-name",
            cluster="test-cluster",
        )
        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())

        with mock.patch.multiple(
            job,
            _get_status=mock.Mock(
                side_effect=[
                    runner_gke.GKERunnerJob.Status.NOT_STARTED,
                    runner_gke.GKERunnerJob.Status.COMPLETED,
                ]
            ),
            _delete=mock.DEFAULT,
            _inner=mock.DEFAULT,
            _pre_provisioner=mock.DEFAULT,
        ):
            job._execute()
            job._inner.execute.assert_called()  # pytype: disable=attribute-error

    def test_pre_provisioner(self):
        class DummyProvisioner(FlagConfigurable):
            """A dummy provisioner."""

            @config_class
            class Config(FlagConfigurable.Config):
                provisioner_only_config: Required[int] = REQUIRED

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                flags.DEFINE_integer("provisioner_only_config", None, "", flag_values=fv)

        cfg: GKERunnerJob.Config = GKERunnerJob.default_config().set(
            pre_provisioner=DummyProvisioner.default_config()
        )

        # Check that provisioner flags are available.
        fv = flags.FlagValues()
        define_flags(cfg, fv)
        self.assertIn("enable_pre_provisioner", fv)
        self.assertIsNone(fv["enable_pre_provisioner"].default)
        fv.mark_as_parsed()

        # Should be disabled by default, even if initially in the config.
        cfg = from_flags(cfg, fv)

        # Check disable-autoprovisioning true for the jobset.
        self.assertFalse(cfg.enable_pre_provisioner)
        self.assertIsNone(cfg.pre_provisioner)


class TPUGKERunnerJobTest(parameterized.TestCase):
    """Tests TPUGKERunnerJob."""

    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        mock_user = mock.patch("os.environ", {"USER": "test"})
        with (
            mock_user,
            mock_gcp_settings(
                [runner_gke.__name__, bundler.__name__, node_pool_provisioner.__name__],
                settings=self._settings,
            ),
        ):
            return super().run(result)

    def _job_config(
        self, *, name: str, command: str, env_vars: Optional[dict] = None, **kwargs
    ) -> GKERunnerJob.Config:
        fv = flags.FlagValues()
        cfg = named_runner_configs("gke_tpu_single")
        define_flags(cfg, fv)
        # Set `name` as a default; since implementations typically use `generate_job_name`, we
        # want to exercise the case that the default value of name is not overridden.
        fv.set_default("name", name)
        for key, value in kwargs.items():
            if value is not None:
                setattr(fv, key, value)
        if env_vars:
            fv.env = [f"{k}:{v}" for k, v in env_vars.items()]
        fv.set_default("instance_type", "tpu-v4-8")
        fv.mark_as_parsed()
        return from_flags(cfg, fv, command=command)

    @parameterized.product(
        name=[None, "test-name"],
        cluster=[None, "test-cluster"],
        enable_pre_provisioner=[None, False, True],
        gcsfuse_mount_spec=[None, ["gcs_path=my-test-path"]],
        env_vars=[None, {"test": "123"}],
    )
    def test_from_flags(self, name, cluster, enable_pre_provisioner, gcsfuse_mount_spec, env_vars):
        cfg = self._job_config(
            command="test-command",
            name=name,
            cluster=cluster,
            enable_pre_provisioner=enable_pre_provisioner,
            gcsfuse_mount_spec=gcsfuse_mount_spec,
            env_vars=env_vars,
        )
        if name:
            self.assertEqual(cfg.name, name)
        else:
            self.assertIsNotNone(cfg.name)
        self.assertEqual(cfg.cluster, cluster or self._settings["gke_cluster"])
        self.assertEqual(cfg.enable_pre_provisioner, enable_pre_provisioner)
        builder_cfg: TPUReplicatedJob.Config = cfg.inner.builder
        self.assertIsInstance(builder_cfg, TPUReplicatedJob.Config)
        self.assertEqual(builder_cfg.name, cfg.name)
        self.assertEqual(builder_cfg.output_dir, cfg.output_dir)
        self.assertIn(cfg.name, cfg.output_dir)
        if gcsfuse_mount_spec:
            fuse = builder_cfg.gcsfuse_mount
            self.assertEqual(fuse.gcs_path, "my-test-path")

        # Test that TPU defaults are set.
        self.assertIn("TPU_TYPE", builder_cfg.env_vars)
        if env_vars is not None:
            for k, v in env_vars.items():
                self.assertEqual(builder_cfg.env_vars[k], v)

        # Should be instantiable.
        runner: GKERunnerJob = cfg.instantiate(bundler=mock.Mock())

        # Inner should have consistent configs.
        final_config = runner.config
        inner_config = runner._inner.config
        for key, value in final_config.items():
            if key not in ("klass", "service_account") and key in inner_config.keys():
                self.assertEqual(value, getattr(inner_config, key), msg=key)

    def test_default_name(self):
        """Tests that default name works even when env doesn't contain USER."""
        fv = flags.FlagValues()
        GKERunnerJob.define_flags(fv)
        fv.mark_as_parsed()
        GKERunnerJob.set_defaults(fv)
        self.assertIsNotNone(fv["name"].default)

    @parameterized.product(
        status=[
            runner_gke.GKERunnerJob.Status.FAILED,
            runner_gke.GKERunnerJob.Status.SUCCEEDED,
            runner_gke.GKERunnerJob.Status.COMPLETED,
        ],
        enable_pre_provisioner=[None, False, True],
    )
    def test_exit(self, status, enable_pre_provisioner):
        cfg = self._job_config(
            command="",
            name="test-name",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
        )
        job: GKERunnerJob = cfg.instantiate(bundler=mock.Mock())
        with mock.patch.multiple(
            job, _get_status=mock.Mock(return_value=status), _delete=mock.DEFAULT
        ):
            job._execute()

    @parameterized.parameters(
        dict(
            status=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
            expected="test-reservation",
        ),
        dict(
            status=dict(replicatedJobs=_mock_replicated_jobs(["spot"])),
            expected=None,
        ),
        dict(
            status=dict(replicatedJobs=_mock_replicated_jobs(["spot", "test-reservation"])),
            expected="test-reservation",
        ),
        dict(
            status=dict(replicatedJobs=[{"template": {}}]),
            expected=None,
        ),
    )
    def test_infer_reservation(self, status: dict, expected: Optional[str] = None):
        self.assertEqual(expected, _infer_reservation(status))

    @parameterized.parameters(
        dict(
            spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
            expected=None,
        ),
        dict(
            spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], num_replicas=1)),
            expected=1,
        ),
        dict(
            spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], num_replicas=2)),
            expected=2,
        ),
    )
    def test_infer_job_count(self, spec: dict, expected: Optional[str] = None):
        self.assertEqual(expected, _infer_job_count(spec))

    @parameterized.parameters(
        dict(
            status=dict(
                replicatedJobs=_mock_replicated_jobs(["test-reservation"], bastion_job_version=None)
            ),
            expected=None,
        ),
        dict(
            status=dict(
                replicatedJobs=_mock_replicated_jobs(["test-reservation"], bastion_job_version=1)
            ),
            expected=1,
        ),
        dict(
            status=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
            expected=None,
        ),
    )
    def test_infer_job_version(self, status: dict, expected: Optional[str] = None):
        self.assertEqual(expected, _infer_job_version(status))

    @parameterized.product(
        (
            # Conditions is set, so we use it.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    conditions=[
                        dict(type="COMPLETED", status="TRUE"),
                    ]
                ),
                spec=None,
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.COMPLETED,
            ),
            # Ignore conditions with status.lower() != "true".
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    conditions=[
                        dict(type="COMPLETED", status="FALSE"),
                        dict(type="FAILED", status="TRUE"),
                    ]
                ),
                spec=None,
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.FAILED,
            ),
            # Missing conditions entirely, fallback to child job statuses.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=0, ready=1, succeeded=0),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"], num_replicas=1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # Missing conditions entirely, fallback to child job statuses.
            # Ignore conditions with status.lower() != "true".
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    conditions=[dict(type="COMPLETED", status="FALSE")],
                    replicatedJobsStatus=[
                        dict(failed=0, ready=1, succeeded=0),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"], num_replicas=1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # At least one job failed. We go to PENDING until conditions is set,
            # or until replicated job statuses change.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=1, ready=1, succeeded=0),
                    ],
                ),
                spec=None,
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.PENDING,
            ),
            # At least one job failed without conditions, and tier does not match.
            dict(
                tier="0",
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=1, ready=1, succeeded=0),
                    ],
                ),
                spec=None,
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Number of replicated job statuses do not match slices.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=0, ready=1, succeeded=0),
                    ],
                ),
                spec=None,
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.UNKNOWN,
            ),
            # All replicated jobs succeeded. No need to wait for jobset conditions.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=0, ready=0, succeeded=2),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"], num_replicas=2)),
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.SUCCEEDED,
            ),
            # Ignore active and missing statuses.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=1, ready=1),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"], num_replicas=1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # Missing jobset is reported as "not started".
            dict(
                tier=None,
                job_version=None,
                status=k8s.client.exceptions.ApiException(status=404),
                spec=None,
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.NOT_STARTED,
            ),
            # All statuses are 0.
            dict(
                tier=None,
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=0, ready=0, succeeded=0),
                    ],
                ),
                spec=None,
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.PENDING,
            ),
            # All statuses are 0 and tiers do not match (thus will be recreated).
            dict(
                tier="0",
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(failed=0, ready=0, succeeded=0),
                    ],
                ),
                spec=None,
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Jobset reservation and bastion tier do not match.
            dict(
                tier="1",
                job_version=None,
                status={},
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Jobset reservation and bastion tier do not match.
            dict(
                tier="1",
                job_version=None,
                status={},
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot", "test-reservation"])),
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Jobset reservation and bastion tier do not match.
            # In this case, we allow the job to keep running.
            dict(
                tier="0",
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=2, ready=2),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"], num_replicas=2)),
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # Missing reservation / invalid spec will be treated as spot.
            dict(
                tier="0",
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=2, ready=2),
                    ],
                ),
                spec=dict(replicatedJobs=[{"replicas": 2, "template": {}}]),
                num_slices=2,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # Job version has increased from None.
            dict(
                tier="0",
                job_version=1,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=1, ready=1),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], None, 1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.UPDATING,
            ),
            # Job version has increased from a non-None number.
            dict(
                tier="0",
                job_version=4,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=1, ready=1),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 3, 1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.UPDATING,
            ),
            # Job version has decreased, in which case, no update.
            dict(
                tier="0",
                job_version=1,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=1, ready=1),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 2, 1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # Job version is set to None, in which case, no update.
            dict(
                tier="0",
                job_version=None,
                status=dict(
                    replicatedJobsStatus=[
                        dict(active=1, ready=1),
                    ],
                ),
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 2, 1)),
                num_slices=1,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
        ),
        enable_pre_provisioner=(None, False, True),
    )
    def test_get_status(
        self,
        status: dict,
        num_slices: int,
        expected: runner_gke.GKERunnerJob.Status,
        tier: str,
        job_version: Optional[int],
        spec: dict,
        enable_pre_provisioner: Optional[bool] = None,
    ):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
            num_replicas=num_slices,
        )
        job: GKERunnerJob = cfg.instantiate(bundler=mock.Mock())

        if isinstance(status, Exception):
            mock_get_status = mock.Mock(side_effect=status)
        else:
            mock_get_status = mock.Mock(return_value=dict(status=status, spec=spec))

        with (
            mock.patch.dict(
                "os.environ", {"BASTION_TIER": tier, BASTION_JOB_VERSION_ENV_VAR: job_version}
            ),
            mock.patch(
                "kubernetes.client.CustomObjectsApi",
                return_value=mock.Mock(get_namespaced_custom_object_status=mock_get_status),
            ),
        ):
            self.assertEqual(expected, job._get_status())

    @parameterized.parameters(
        # Don't need to reschedule if no node-pool exists.
        dict(node_pool_by_provisioner={}, expect_delete_count=0),
        # Test a case when tier=0 matches reservation.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "reservationAffinity": {
                                "key": "compute.googleapis.com/reservation-name",
                                "values": ["test-reservation"],
                            },
                            "labels": {"provisioner-nodepool-id": "test-name"},
                        },
                    }
                ]
            },
            tier=0,
            expect_delete_count=0,
            enable_pre_provisioner=False,
        ),
        # Test a case when tier=1 matches spot.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "taints": [
                                {
                                    "key": "cloud.google.com/gke-spot",
                                    "value": "true",
                                    "effect": "NO_SCHEDULE",
                                }
                            ],
                            "labels": {"provisioner-nodepool-id": "test-name"},
                        },
                    }
                ]
            },
            tier=1,
            expect_delete_count=0,
            enable_pre_provisioner=False,
        ),
        # Tier=0 doesn't match spot, and delete goes through the first time.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "taints": [
                                {
                                    "key": "cloud.google.com/gke-spot",
                                    "value": "true",
                                    "effect": "NO_SCHEDULE",
                                }
                            ],
                            "labels": {"provisioner-nodepool-id": "test-name"},
                        },
                    }
                ]
            },
            tier=0,
            expect_delete_count=1,
            enable_pre_provisioner=False,
        ),
        # Tier=1 doesn't match reservation, and delete goes through the first time.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "reservationAffinity": {
                                "key": "compute.googleapis.com/reservation-name",
                                "values": ["test-reservation"],
                            },
                            "labels": {"provisioner-nodepool-id": "test-name"},
                        },
                    }
                ]
            },
            tier=1,
            expect_delete_count=1,
            enable_pre_provisioner=False,
        ),
        # Don't need to reschedule if no node-pool exists.
        dict(node_pool_by_provisioner={}, expect_delete_count=0, enable_pre_provisioner=True),
        # Test a case when tier=0 matches reservation.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "reservationAffinity": {
                                "key": "compute.googleapis.com/reservation-name",
                                "values": ["test-reservation"],
                            },
                            "labels": {PRE_PROVISIONER_LABEL: "test-name"},
                        },
                    }
                ]
            },
            tier=0,
            expect_delete_count=0,
            enable_pre_provisioner=True,
        ),
        # Test a case when tier=1 matches spot.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "taints": [
                                {
                                    "key": "cloud.google.com/gke-spot",
                                    "value": "true",
                                    "effect": "NO_SCHEDULE",
                                }
                            ],
                            "labels": {PRE_PROVISIONER_LABEL: "test-name"},
                        },
                    }
                ]
            },
            tier=1,
            expect_delete_count=0,
            enable_pre_provisioner=True,
        ),
        # Tier=0 doesn't match spot, and delete goes through the first time.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "taints": [
                                {
                                    "key": "cloud.google.com/gke-spot",
                                    "value": "true",
                                    "effect": "NO_SCHEDULE",
                                }
                            ],
                            "labels": {PRE_PROVISIONER_LABEL: "test-name"},
                        },
                    }
                ]
            },
            tier=0,
            expect_delete_count=1,
            enable_pre_provisioner=True,
        ),
        # Tier=1 doesn't match reservation, and delete goes through the first time.
        dict(
            node_pool_by_provisioner={
                "test-name": [
                    {
                        "name": "pool0",
                        "config": {
                            "reservationAffinity": {
                                "key": "compute.googleapis.com/reservation-name",
                                "values": ["test-reservation"],
                            },
                            "labels": {PRE_PROVISIONER_LABEL: "test-name"},
                        },
                    }
                ]
            },
            tier=1,
            expect_delete_count=1,
            enable_pre_provisioner=True,
        ),
    )
    def test_reschedule(
        self,
        node_pool_by_provisioner,
        expect_delete_count,
        tier=None,
        enable_pre_provisioner=False,
    ):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
        )
        # Node pool test cases assume "test-name".
        self.assertEqual("test-name", cfg.name)

        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())

        mock_job = mock.patch.multiple(
            job,
            _get_status=mock.Mock(
                side_effect=[
                    runner_gke.GKERunnerJob.Status.RESCHEDULED,
                    runner_gke.GKERunnerJob.Status.COMPLETED,
                ]
            ),
            _delete=mock.DEFAULT,
            _inner=mock.DEFAULT,
            _pre_provisioner=mock.DEFAULT,
        )
        mock_list_node_pools_by_label_key = mock.Mock(return_value=node_pool_by_provisioner)
        mock_delete_node_pools = mock.Mock()
        mock_node_pool = mock.patch.multiple(
            runner_gke.__name__,
            delete_node_pools=mock_delete_node_pools,
            list_node_pools_by_label_key=mock_list_node_pools_by_label_key,
        )
        mock_env = mock.patch("os.environ", {"BASTION_TIER": tier} if tier is not None else {})
        with mock_env, mock_job, mock_node_pool:
            job._reschedule()

            self.assertEqual(expect_delete_count, mock_delete_node_pools.call_count)

            # Jobset should always be deleted.
            job._inner._delete.assert_called()  # pytype: disable=attribute-error

    @parameterized.parameters(None, False, True)
    def test_delete(self, enable_pre_provisioner):
        cfg = self._job_config(
            name="test-name",
            command="",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
        )
        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())

        with mock.patch.multiple(
            job,
            _inner=mock.DEFAULT,
            _pre_provisioner=mock.DEFAULT,
        ):
            job._delete()

            job._inner._delete.assert_called()  # pytype: disable=attribute-error

            if enable_pre_provisioner:
                # pytype: disable=attribute-error
                job._pre_provisioner.delete_for.assert_called()
                # pytype: enable=attribute-error

    @parameterized.parameters(None, False, True)
    def test_start(self, enable_pre_provisioner):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
        )
        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())

        with mock.patch.multiple(
            job,
            _get_status=mock.Mock(
                side_effect=[
                    runner_gke.GKERunnerJob.Status.NOT_STARTED,
                    runner_gke.GKERunnerJob.Status.COMPLETED,
                ]
            ),
            _delete=mock.DEFAULT,
            _inner=mock.DEFAULT,
            _pre_provisioner=mock.DEFAULT,
        ):
            job._execute()

            if enable_pre_provisioner:
                # pytype: disable=attribute-error
                job._pre_provisioner.create_for.assert_called()
                # pytype: enable=attribute-error

            job._inner.execute.assert_called()  # pytype: disable=attribute-error

    @parameterized.parameters(None, False, True)
    def test_update(self, enable_pre_provisioner):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            enable_pre_provisioner=enable_pre_provisioner,
        )
        job: GKERunnerJob = cfg.set(status_interval_seconds=0).instantiate(bundler=mock.Mock())

        with mock.patch.multiple(
            job,
            _get_status=mock.Mock(
                side_effect=[
                    runner_gke.GKERunnerJob.Status.UPDATING,
                    runner_gke.GKERunnerJob.Status.COMPLETED,
                ]
            ),
            _delete=mock.DEFAULT,
            _inner=mock.DEFAULT,
            _pre_provisioner=mock.DEFAULT,
        ):
            job._execute()

            # pytype: disable=attribute-error
            job._pre_provisioner.delete_for.assert_not_called()
            job._inner._delete.assert_called()
            # pytype: enable=attribute-error

    def test_name_alias(self):
        """Tests that names set via flag aliases are retained."""
        with (
            mock_gcp_settings(
                [runner_gke.__name__, bundler.__name__, node_pool_provisioner.__name__],
                default_mock_settings(),
            ),
        ):
            cfg: GKERunnerJob.Config = GKERunnerJob.default_config()
            fv = flags.FlagValues()
            define_flags(cfg, fv)
            fv.mark_as_parsed()
            self.assertIsNone(fv.name)
            self.assertIsNone(fv["name"].default)
            flags.DEFINE_alias("alias_name", "name", flag_values=fv)
            fv.set_default("alias_name", "test-name")
            from_flags(cfg, fv)
            self.assertEqual(cfg.name, fv.alias_name)


class FlinkGKERunnerJobTest(parameterized.TestCase):
    def run(self, result=None):
        # Run tests under mock user and settings.
        self._settings = default_mock_settings()
        mock_user = mock.patch("os.environ", {"USER": "test"})
        with (
            mock_user,
            mock_gcp_settings(
                [runner_gke.__name__, bundler.__name__, node_pool_provisioner.__name__],
                settings=self._settings,
            ),
        ):
            return super().run(result)

    def _job_config(self, *, command: str, **kwargs) -> GKERunnerJob.Config:
        fv = flags.FlagValues()
        cfg = named_runner_configs("gke_tpu_flink")
        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                setattr(fv, key, value)
        fv.mark_as_parsed()
        return from_flags(cfg, fv, command=command)

    @parameterized.parameters(
        dict(
            flink_response={"jobs": [{"state": FlinkJobStatus.FINISHED.value}]},
            expected=runner_gke.GKERunnerJob.Status.SUCCEEDED,
        ),
        dict(
            flink_response={"jobs": [{"state": FlinkJobStatus.FAILED.value}]},
            expected=runner_gke.GKERunnerJob.Status.FAILED,
        ),
        dict(
            flink_response={"jobs": [{"state": FlinkJobStatus.FAILING.value}]},
            expected=runner_gke.GKERunnerJob.Status.FAILED,
        ),
        dict(
            flink_response={"jobs": [{"state": "RUNNING"}]},
            expected=runner_gke.GKERunnerJob.Status.READY,
        ),
        dict(
            flink_response={"jobs": [{"no_state_key": "???"}]},
            expected=runner_gke.GKERunnerJob.Status.FAILED,
        ),
        dict(
            flink_response={"jobs": []},
            expected=runner_gke.GKERunnerJob.Status.FAILED,
        ),
    )
    @mock.patch("axlearn.cloud.gcp.runners.gke.requests.get")
    def test_get_flink_job_status(self, mock_get, flink_response, expected):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            instance_type="v5p-8",
        )
        job: runner_gke.FlinkGKERunnerJob = cfg.instantiate(bundler=mock.Mock())
        job._inner = runner_gke.FlinkTPUGKEJob(cfg.inner, bundler=mock.Mock())
        job._inner.job_manager_ip = "127.0.0.1"

        mock_resp = mock.Mock()
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = flink_response
        mock_get.return_value = mock_resp

        result = job._get_flink_job_status()
        self.assertEqual(result, expected)

    @mock.patch("axlearn.cloud.gcp.runners.gke.requests.get")
    def test_get_flink_job_status_raises_on_request_error(self, mock_get):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            instance_type="v5p-8",
        )
        job: runner_gke.FlinkGKERunnerJob = cfg.instantiate(bundler=mock.Mock())
        job._inner = runner_gke.FlinkTPUGKEJob(cfg.inner, bundler=mock.Mock())
        job._inner.job_manager_ip = "127.0.0.1"  # Assuming this is now a private runtime attribute

        mock_get.side_effect = requests.RequestException("network issue")

        with self.assertRaises(RuntimeError) as context:
            job._get_flink_job_status()

        # Updated expected error string
        self.assertIn("Unexpected error while getting Flink job status", str(context.exception))

        # Optional: confirm cause was preserved
        self.assertIsInstance(context.exception.__cause__, requests.RequestException)

    @parameterized.product(
        (
            # SUCCEEDED
            dict(
                status={
                    "status": {
                        "conditions": [
                            {
                                "type": "Complete",
                                "status": "True",
                            }
                        ],
                    }
                },
                flink_status=runner_gke.GKERunnerJob.Status.SUCCEEDED,
                expected=runner_gke.GKERunnerJob.Status.SUCCEEDED,
            ),
            # FAILED
            dict(
                status={
                    "status": {
                        "conditions": [
                            {
                                "type": "Failed",
                                "status": "True",
                            }
                        ],
                    }
                },
                flink_status=None,
                expected=runner_gke.GKERunnerJob.Status.FAILED,
            ),
            # PENDING
            dict(
                status={"status": {"active": 0, "succeeded": 0, "failed": 0}},
                flink_status=None,
                expected=runner_gke.GKERunnerJob.Status.PENDING,
            ),
            # READY
            dict(
                status={"status": {"active": 1, "succeeded": 0, "failed": 0}},
                flink_status=None,
                expected=runner_gke.GKERunnerJob.Status.READY,
            ),
            # UNKNOWN
            dict(
                status={"status": {"active": 0, "succeeded": 1, "failed": 1}},
                flink_status=None,
                expected=runner_gke.GKERunnerJob.Status.UNKNOWN,
            ),
            # NOT_STARTED
            dict(
                status=k8s.client.exceptions.ApiException(status=404),
                flink_status=None,
                expected=runner_gke.GKERunnerJob.Status.NOT_STARTED,
            ),
            # Permission error
            dict(
                status=k8s.client.exceptions.ApiException(status=403),
                flink_status=None,
                expected=k8s.client.exceptions.ApiException(status=403),
            ),
        )
    )
    def test_get_status(
        self,
        status: dict,
        flink_status,
        expected,
    ):
        cfg = self._job_config(
            command="test-command",
            name="test-name",
            cluster="test-cluster",
            instance_type="v5p-8",
        )
        job: runner_gke.FlinkGKERunnerJob = cfg.instantiate(bundler=mock.Mock())
        job._inner = runner_gke.FlinkTPUGKEJob(cfg.inner, bundler=mock.Mock())
        job._inner.job_manager_ip = "127.0.0.1"

        if isinstance(status, Exception):
            mock_get_status = mock.Mock(side_effect=status)
        else:
            mock_get_status = mock.Mock(return_value=status)

        with mock.patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=mock.Mock(get_namespaced_custom_object_status=mock_get_status),
        ):
            if isinstance(expected, Exception):
                # Expecting Exception (like permission error)
                with self.assertRaises(Exception) as context:
                    job._get_status()
                self.assertEqual(str(expected), str(context.exception))
            else:
                if flink_status is not None:
                    # Patch _get_flink_job_status() only when needed
                    with mock.patch.object(job, "_get_flink_job_status", return_value=flink_status):
                        self.assertEqual(expected, job._get_status())
                else:
                    self.assertEqual(expected, job._get_status())
