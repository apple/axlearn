# Copyright © 2024 Apple Inc.

"""Tests GKERunnerJob."""

# pylint: disable=no-self-use,protected-access
import contextlib
from collections.abc import Iterator, Sequence
from typing import Optional, Union, cast
from unittest import mock

import kubernetes as k8s
from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import BASTION_JOB_VERSION_ENV_VAR
from axlearn.cloud.gcp import bundler, node_pool_provisioner
from axlearn.cloud.gcp.job import GPUGKEJob, TPUGKEJob
from axlearn.cloud.gcp.jobs import gke_runner
from axlearn.cloud.gcp.jobs.bastion_vm_test import _mock_job
from axlearn.cloud.gcp.jobs.gke_runner import (
    _get_runner_or_exit,
    _infer_job_version,
    _infer_reservation,
)
from axlearn.cloud.gcp.jobset_utils import BASTION_JOB_VERSION_LABEL
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.test_utils import mock_gcp_settings


def _mock_replicated_jobs(reservations: Sequence[str], bastion_job_version: Optional[int] = None):
    job_version_label = (
        {"metadata": {"labels": {BASTION_JOB_VERSION_LABEL: str(bastion_job_version)}}}
        if bastion_job_version
        else {}
    )

    return [
        {
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
                }
            }
        }
        for reservation in reservations
    ]


class GPUGKERunnerJobTest(parameterized.TestCase):
    """Tests GPUGKERunnerJob."""

    @contextlib.contextmanager
    def _job_config(
        self,
        *,
        name: str,
        cluster: str,
        service_account: str,
        gcsfuse_mount_spec: Optional[str] = None,
    ) -> Iterator[tuple[gke_runner.GPUGKERunnerJob.Config, dict]]:
        mock_user = mock.patch("os.environ", {"USER": "test"})
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone-a",
            "ttl_bucket": "settings-ttl-bucket",
            "gke_cluster": "settings-cluster",
            "default_dockerfile": "settings-dockerfile",
            "docker_repo": "settings-repo",
        }
        with (
            mock_user,
            mock_gcp_settings(
                [gke_runner.__name__, bundler.__name__, node_pool_provisioner.__name__],
                mock_settings,
            ),
        ):
            fv = flags.FlagValues()
            gke_runner.GPUGKERunnerJob.define_flags(fv)
            if name:
                fv.set_default("name", name)
            if cluster:
                fv.set_default("cluster", cluster)
            if service_account:
                fv.set_default("service_account", service_account)
            if gcsfuse_mount_spec:
                fv.set_default("gcsfuse_mount_spec", gcsfuse_mount_spec)
            fv.set_default("instance_type", "gpu-a3-highgpu-8g-256")
            fv.mark_as_parsed()
            yield gke_runner.GPUGKERunnerJob.from_flags(fv), mock_settings

    @parameterized.product(
        name=[None, "test-name"],
        cluster=[None, "test-cluster"],
        service_account=[None, "test-sa"],
        gcsfuse_mount_spec=[None, ["gcs_path=my-test-path"]],
    )
    def test_from_flags(self, name, cluster, service_account, gcsfuse_mount_spec):
        with self._job_config(
            name=name,
            cluster=cluster,
            service_account=service_account,
            gcsfuse_mount_spec=gcsfuse_mount_spec,
        ) as (cfg, mock_settings):
            if name:
                self.assertEqual(cfg.name, name)
            else:
                self.assertIsNotNone(cfg.name)
            self.assertEqual(cfg.cluster, cluster or mock_settings["gke_cluster"])
            self.assertEqual(cfg.service_account, service_account or "default")
            if gcsfuse_mount_spec:
                fuse = cast(GPUGKEJob.Config, cfg.inner).builder.gcsfuse_mount
                self.assertEqual(fuse.gcs_path, "my-test-path")

    @parameterized.product(
        status=[
            gke_runner.GKERunnerJob.Status.FAILED,
            gke_runner.GKERunnerJob.Status.SUCCEEDED,
            gke_runner.GKERunnerJob.Status.COMPLETED,
        ],
    )
    def test_exit(self, status):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
        ) as (cfg, _):
            cfg.bundler.set(image="test")
            job: gke_runner.GPUGKERunnerJob = cfg.set(command="").instantiate()

            mock_job = mock.patch.multiple(
                job, _get_status=mock.Mock(return_value=status), _delete=mock.DEFAULT
            )

            with mock_job:
                job._execute()

    def test_delete(self):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
        ) as (cfg, _):
            cfg.bundler.set(image="test")

            job: gke_runner.GPUGKERunnerJob = cfg.set(
                command="", status_interval_seconds=0
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )

            with mock_job:
                job._delete()
                job._inner._delete.assert_called()  # pytype: disable=attribute-error

    def test_start(self):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
        ) as (
            cfg,
            _,
        ):
            cfg.bundler.set(image="test")

            job: gke_runner.GPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _get_status=mock.Mock(
                    side_effect=[
                        gke_runner.GKERunnerJob.Status.NOT_STARTED,
                        gke_runner.GKERunnerJob.Status.COMPLETED,
                    ]
                ),
                _get_job_credentials=mock.DEFAULT,
                _delete=mock.DEFAULT,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )

            with mock_job:
                job._execute()
                job._inner.execute.assert_called()  # pytype: disable=attribute-error


class TPUGKERunnerJobTest(parameterized.TestCase):
    """Tests TPUGKERunnerJob."""

    @contextlib.contextmanager
    def _job_config(
        self,
        *,
        name: str,
        cluster: str,
        service_account: str,
        enable_pre_provisioner: Optional[bool] = None,
        gcsfuse_mount_spec: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> Iterator[tuple[gke_runner.TPUGKERunnerJob.Config, dict]]:
        mock_user = mock.patch("os.environ", {"USER": "test"})
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone-a",
            "ttl_bucket": "settings-ttl-bucket",
            "gke_cluster": "settings-cluster",
            "default_dockerfile": "settings-dockerfile",
            "docker_repo": "settings-repo",
        }
        with (
            mock_user,
            mock_gcp_settings(
                [gke_runner.__name__, bundler.__name__, node_pool_provisioner.__name__],
                mock_settings,
            ),
        ):
            fv = flags.FlagValues()
            gke_runner.TPUGKERunnerJob.define_flags(fv)
            if name:
                fv.set_default("name", name)
            if cluster:
                fv.set_default("cluster", cluster)
            if service_account:
                fv.set_default("service_account", service_account)
            fv.set_default("enable_pre_provisioner", enable_pre_provisioner)
            if gcsfuse_mount_spec:
                fv.set_default("gcsfuse_mount_spec", gcsfuse_mount_spec)
            if env_vars:
                fv.set_default("env", [f"{k}:{v}" for k, v in env_vars.items()])
            fv.set_default("instance_type", "tpu-v4-8")
            fv.mark_as_parsed()
            yield gke_runner.TPUGKERunnerJob.from_flags(fv), mock_settings

    @parameterized.product(
        name=[None, "test-name"],
        cluster=[None, "test-cluster"],
        service_account=[None, "test-sa"],
        enable_pre_provisioner=[None, False, True],
        gcsfuse_mount_spec=[None, ["gcs_path=my-test-path"]],
        env_vars=[None, {"test": "123"}],
    )
    def test_from_flags(
        self, name, cluster, service_account, enable_pre_provisioner, gcsfuse_mount_spec, env_vars
    ):
        with self._job_config(
            name=name,
            cluster=cluster,
            service_account=service_account,
            enable_pre_provisioner=enable_pre_provisioner,
            gcsfuse_mount_spec=gcsfuse_mount_spec,
            env_vars=env_vars,
        ) as (cfg, mock_settings):
            if name:
                self.assertEqual(cfg.name, name)
            else:
                self.assertIsNotNone(cfg.name)
            self.assertEqual(cfg.cluster, cluster or mock_settings["gke_cluster"])
            self.assertEqual(cfg.service_account, service_account or "default")
            self.assertEqual(cfg.enable_pre_provisioner, enable_pre_provisioner)
            if gcsfuse_mount_spec:
                fuse = cast(TPUGKEJob.Config, cfg.inner).builder.gcsfuse_mount
                self.assertEqual(fuse.gcs_path, "my-test-path")

            # Test that TPU defaults are set.
            self.assertIn("TPU_TYPE", cfg.env_vars)
            if env_vars is not None:
                for k, v in env_vars.items():
                    self.assertEqual(cfg.env_vars[k], v)

            # Instantiating should propagate fields.
            cfg.bundler.image = "FAKE"
            runner = cfg.instantiate()
            final_config: gke_runner.TPUGKERunnerJob.Config = runner.config
            inner_config: TPUGKEJob.Config = runner._inner.config
            for key, value in final_config.items():
                if key not in ("klass", "bundler") and key in inner_config.keys():
                    self.assertEqual(value, getattr(inner_config, key), msg=key)

    @parameterized.product(
        status=[
            gke_runner.GKERunnerJob.Status.FAILED,
            gke_runner.GKERunnerJob.Status.SUCCEEDED,
            gke_runner.GKERunnerJob.Status.COMPLETED,
        ],
        enable_pre_provisioner=[None, False, True],
    )
    def test_exit(self, status, enable_pre_provisioner):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (cfg, _):
            cfg.bundler.set(image="test")
            job: gke_runner.TPUGKERunnerJob = cfg.set(command="").instantiate()

            mock_job = mock.patch.multiple(
                job, _get_status=mock.Mock(return_value=status), _delete=mock.DEFAULT
            )

            with mock_job:
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
                expected=gke_runner.GKERunnerJob.Status.COMPLETED,
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
                expected=gke_runner.GKERunnerJob.Status.FAILED,
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
                spec=None,
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.READY,
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
                spec=None,
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.READY,
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
                expected=gke_runner.GKERunnerJob.Status.PENDING,
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
                expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
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
                expected=gke_runner.GKERunnerJob.Status.UNKNOWN,
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
                spec=None,
                num_slices=2,
                expected=gke_runner.GKERunnerJob.Status.SUCCEEDED,
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
                spec=None,
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.READY,
            ),
            # Missing jobset is reported as "not started".
            dict(
                tier=None,
                job_version=None,
                status=k8s.client.exceptions.ApiException(status=404),
                spec=None,
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.NOT_STARTED,
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
                expected=gke_runner.GKERunnerJob.Status.PENDING,
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
                expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Jobset reservation and bastion tier do not match.
            dict(
                tier="1",
                job_version=None,
                status={},
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
                num_slices=2,
                expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
            ),
            # Jobset reservation and bastion tier do not match.
            dict(
                tier="1",
                job_version=None,
                status={},
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot", "test-reservation"])),
                num_slices=2,
                expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
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
                spec=dict(replicatedJobs=_mock_replicated_jobs(["spot"])),
                num_slices=2,
                expected=gke_runner.GKERunnerJob.Status.READY,
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
                spec=dict(replicatedJobs=[{"template": {}}]),
                num_slices=2,
                expected=gke_runner.GKERunnerJob.Status.READY,
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
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], None)),
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.UPDATING,
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
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 3)),
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.UPDATING,
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
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 2)),
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.READY,
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
                spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"], 2)),
                num_slices=1,
                expected=gke_runner.GKERunnerJob.Status.READY,
            ),
        ),
        enable_pre_provisioner=(None, False, True),
    )
    def test_get_status(
        self,
        status: dict,
        num_slices: int,
        expected: gke_runner.GKERunnerJob.Status,
        tier: str,
        job_version: Optional[int],
        spec: dict,
        enable_pre_provisioner: Optional[bool] = None,
    ):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (cfg, _):
            cfg.inner.accelerator.set(instance_type="v4-8", num_replicas=num_slices)
            cfg.bundler.set(image="test")
            job: gke_runner.TPUGKERunnerJob = cfg.set(command="").instantiate()

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
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (cfg, _):
            cfg.bundler.set(image="test")
            # Node pool test cases assume "test-name".
            self.assertEqual("test-name", cfg.name)

            job: gke_runner.TPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
                enable_pre_provisioner=enable_pre_provisioner,
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _get_status=mock.Mock(
                    side_effect=[
                        gke_runner.GKERunnerJob.Status.RESCHEDULED,
                        gke_runner.GKERunnerJob.Status.COMPLETED,
                    ]
                ),
                _get_job_credentials=mock.DEFAULT,
                _delete=mock.DEFAULT,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )
            mock_list_node_pools_by_label_key = mock.Mock(return_value=node_pool_by_provisioner)
            mock_delete_node_pools = mock.Mock()
            mock_node_pool = mock.patch.multiple(
                gke_runner.__name__,
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
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (
            cfg,
            _,
        ):
            cfg.bundler.set(image="test")

            job: gke_runner.TPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
                enable_pre_provisioner=enable_pre_provisioner,
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )

            with mock_job:
                job._delete()

                job._inner._delete.assert_called()  # pytype: disable=attribute-error

                if enable_pre_provisioner:
                    # pytype: disable=attribute-error
                    job._pre_provisioner.delete_for.assert_called()
                    # pytype: enable=attribute-error

    @parameterized.parameters(None, False, True)
    def test_start(self, enable_pre_provisioner):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (
            cfg,
            _,
        ):
            cfg.bundler.set(image="test")

            job: gke_runner.TPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
                enable_pre_provisioner=enable_pre_provisioner,
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _get_status=mock.Mock(
                    side_effect=[
                        gke_runner.GKERunnerJob.Status.NOT_STARTED,
                        gke_runner.GKERunnerJob.Status.COMPLETED,
                    ]
                ),
                _get_job_credentials=mock.DEFAULT,
                _delete=mock.DEFAULT,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )

            with mock_job:
                job._execute()

                if enable_pre_provisioner:
                    # pytype: disable=attribute-error
                    job._pre_provisioner.create_for.assert_called()
                    # pytype: enable=attribute-error

                job._inner.execute.assert_called()  # pytype: disable=attribute-error

    @parameterized.parameters(None, False, True)
    def test_update(self, enable_pre_provisioner):
        with self._job_config(
            name="test-name",
            cluster="test-cluster",
            service_account="test-sa",
            enable_pre_provisioner=enable_pre_provisioner,
        ) as (
            cfg,
            _,
        ):
            cfg.bundler.set(image="test")

            job: gke_runner.TPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
                enable_pre_provisioner=enable_pre_provisioner,
            ).instantiate()

            mock_job = mock.patch.multiple(
                job,
                _get_status=mock.Mock(
                    side_effect=[
                        gke_runner.GKERunnerJob.Status.UPDATING,
                        gke_runner.GKERunnerJob.Status.COMPLETED,
                    ]
                ),
                _get_job_credentials=mock.DEFAULT,
                _delete=mock.DEFAULT,
                _inner=mock.DEFAULT,
                _pre_provisioner=mock.DEFAULT,
            )

            with mock_job:
                job._execute()

                # pytype: disable=attribute-error
                job._pre_provisioner.delete_for.assert_not_called()
                job._inner._delete.assert_called()
                # pytype: enable=attribute-error


class MainTest(parameterized.TestCase):
    """Tests CLI entrypoint."""

    @parameterized.parameters(
        dict(instance_type="tpu", expected=gke_runner.TPUGKERunnerJob),
        dict(instance_type="tpu-v4-8", expected=gke_runner.TPUGKERunnerJob),
        dict(instance_type="gpu-a3-highgpu-8g-256", expected=gke_runner.GPUGKERunnerJob),
        dict(instance_type="gpu", expected=app.UsageError("instance_type")),
    )
    def test_get_runner_or_exit(self, instance_type: str, expected: Union[Exception, type]):
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                _get_runner_or_exit(instance_type)
        else:
            self.assertEqual(expected, _get_runner_or_exit(instance_type))

    @parameterized.product(
        [
            dict(runner=gke_runner.TPUGKERunnerJob, instance_type="tpu-v4-8"),
            dict(runner=gke_runner.GPUGKERunnerJob, instance_type="gpu-a3-highgpu-8g-256"),
        ],
        action=["start", "stop", "update"],
    )
    def test_load_kube_config(self, action, runner, instance_type):
        # load_kube_config should only be called if using gke action.
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone",
            "gke_cluster": "settings-cluster",
        }
        mock_job = _mock_job(
            runner,
            bundler_kwargs={},
            settings_kwargs=mock_settings,
        )
        mock_utils = mock.patch.multiple(
            gke_runner.__name__,
            load_kube_config=mock.DEFAULT,
            delete_k8s_jobset=mock.DEFAULT,
            with_tpu_training_defaults=mock.DEFAULT,
            list_node_pools_by_label_key=mock.DEFAULT,
        )
        with mock_gcp_settings(gke_runner.__name__, mock_settings), mock_job, mock_utils as m:
            fv = flags.FlagValues()
            gke_runner.TPUGKERunnerJob.define_flags(fv)
            fv.set_default("name", "test")
            fv.set_default("instance_type", instance_type)
            fv.mark_as_parsed()
            gke_runner.main(["cli", action, "test_command"], flag_values=fv)
            call_kwargs = m["load_kube_config"].call_args[1]
            self.assertEqual("settings-project", call_kwargs["project"])
            self.assertEqual("settings-zone", call_kwargs["zone"])
            self.assertEqual("settings-cluster", call_kwargs["cluster"])

    @parameterized.parameters(
        # Node pools for test-job-0 exists.
        dict(
            node_pool_by_provisioner={
                "test-job-0": [
                    {
                        "name": "pool0",
                        "config": {"labels": {PRE_PROVISIONER_LABEL: "test-job-0"}},
                    },
                    {
                        "name": "pool1",
                        "config": {"labels": {PRE_PROVISIONER_LABEL: "test-job-0"}},
                    },
                ]
            },
            expect_delete_np_count=2,
        ),
        # No node pools for test-job-0.
        dict(
            node_pool_by_provisioner={
                "test-job-1": [
                    {
                        "name": "pool0",
                        "config": {"labels": {PRE_PROVISIONER_LABEL: "test-job-1"}},
                    },
                    {
                        "name": "pool1",
                        "config": {"labels": {PRE_PROVISIONER_LABEL: "test-job-1"}},
                    },
                ]
            },
            expect_delete_np_count=0,
        ),
    )
    def test_stop(self, node_pool_by_provisioner, expect_delete_np_count):
        mock_job = _mock_job(
            gke_runner.TPUGKERunnerJob,
            bundler_kwargs={},
            settings_kwargs={},
        )
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone",
            "gke_cluster": "settings-cluster",
        }

        mock_utils = mock.patch.multiple(
            gke_runner.__name__,
            load_kube_config=mock.DEFAULT,
            delete_k8s_jobset=mock.DEFAULT,
            with_tpu_training_defaults=mock.DEFAULT,
            list_node_pools_by_label_key=mock.Mock(return_value=node_pool_by_provisioner),
            delete_node_pools=mock.DEFAULT,
        )
        with mock_gcp_settings(gke_runner.__name__, mock_settings), mock_job, mock_utils:
            fv = flags.FlagValues()
            gke_runner.TPUGKERunnerJob.define_flags(fv)
            fv.set_default("name", "test-job-0")
            fv.mark_as_parsed()
            gke_runner.main(["cli", "stop"], flag_values=fv)

            # pytype: disable=attribute-error
            gke_runner.delete_k8s_jobset.assert_called()
            gke_runner.delete_node_pools.assert_called()
            self.assertEqual(
                expect_delete_np_count,
                len(gke_runner.delete_node_pools.call_args.args[0]),
            )
            # pytype: enable=attribute-error
