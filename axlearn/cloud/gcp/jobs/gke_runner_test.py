# Copyright © 2024 Apple Inc.

"""Tests GKERunnerJob."""

# pylint: disable=no-self-use,protected-access
import contextlib
from typing import Iterator, Optional, Sequence, Type, Union
from unittest import mock

import kubernetes as k8s
from absl import app, flags
from absl.testing import parameterized
from googleapiclient import errors

from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.jobs import gke_runner
from axlearn.cloud.gcp.jobs.bastion_vm_test import _mock_job
from axlearn.cloud.gcp.jobs.gke_runner import (
    _get_node_pool_by_label,
    _get_runner_or_exit,
    _infer_reservation,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings


def _mock_replicated_jobs(reservations: Sequence[str]):
    return [
        {
            "template": {
                "spec": {
                    "template": {
                        "spec": {
                            "nodeSelector": {"cloud.google.com/reservation-name": reservation}
                            if reservation != "spot"
                            else {"cloud.google.com/gke-spot": "true"}
                        },
                    },
                }
            }
        }
        for reservation in reservations
    ]


class TPUGKERunnerJobTest(parameterized.TestCase):
    """Tests TPUGKERunnerJob."""

    @contextlib.contextmanager
    def _job_config(
        self, name: str, cluster: str, service_account: str
    ) -> Iterator[tuple[gke_runner.TPUGKERunnerJob, dict]]:
        mock_user = mock.patch("os.environ", {"USER": "test"})
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone-a",
            "ttl_bucket": "settings-ttl-bucket",
            "gke_cluster": "settings-cluster",
            "default_dockerfile": "settings-dockerfile",
            "docker_repo": "settings-repo",
        }
        with mock_user, mock_gcp_settings([gke_runner.__name__, bundler.__name__], mock_settings):
            fv = flags.FlagValues()
            gke_runner.TPUGKERunnerJob.define_flags(fv)
            if name:
                fv.set_default("name", name)
            if cluster:
                fv.set_default("cluster", cluster)
            if service_account:
                fv.set_default("service_account", service_account)
            fv.set_default("instance_type", "tpu-v4-8")
            fv.mark_as_parsed()
            yield gke_runner.TPUGKERunnerJob.from_flags(fv), mock_settings

    @parameterized.product(
        name=[None, "test-name"],
        cluster=[None, "test-cluster"],
        service_account=[None, "test-sa"],
    )
    def test_from_flags(self, name, cluster, service_account):
        with self._job_config(name, cluster, service_account) as (cfg, mock_settings):
            if name:
                self.assertEqual(cfg.name, name)
            else:
                self.assertIsNotNone(cfg.name)
            self.assertEqual(cfg.cluster, cluster or mock_settings["gke_cluster"])
            self.assertEqual(cfg.service_account, service_account or "default")

            # Test that TPU defaults are set.
            self.assertIn("TPU_TYPE", cfg.env_vars)

    @parameterized.parameters(
        gke_runner.GKERunnerJob.Status.FAILED,
        gke_runner.GKERunnerJob.Status.SUCCEEDED,
        gke_runner.GKERunnerJob.Status.COMPLETED,
    )
    def test_exit(self, status):
        with self._job_config("test-name", "test-cluster", "test-sa") as (cfg, _):
            cfg.bundler.set(image="test")
            job: gke_runner.TPUGKERunnerJob = cfg.set(command="").instantiate()
            with mock.patch.object(job, "_get_status", return_value=status):
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
        # Conditions is set, so we use it.
        dict(
            status=dict(
                conditions=[
                    dict(type="COMPLETED", status="TRUE"),
                ]
            ),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.COMPLETED,
        ),
        # Ignore conditions with status.lower() != "true".
        dict(
            status=dict(
                conditions=[
                    dict(type="COMPLETED", status="FALSE"),
                    dict(type="FAILED", status="TRUE"),
                ]
            ),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.FAILED,
        ),
        # Missing conditions entirely, fallback to child job statuses.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=0, ready=1, succeeded=0),
                ],
            ),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.READY,
        ),
        # Missing conditions entirely, fallback to child job statuses.
        # Ignore conditions with status.lower() != "true".
        dict(
            status=dict(
                conditions=[dict(type="COMPLETED", status="FALSE")],
                replicatedJobsStatus=[
                    dict(failed=0, ready=1, succeeded=0),
                ],
            ),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.READY,
        ),
        # At least one job failed. We go to PENDING until conditions is set, or until replicated job
        # statuses change.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=1, ready=1, succeeded=0),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.PENDING,
        ),
        # At least one job failed without conditions, and tier does not match.
        dict(
            tier="0",
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=1, ready=1, succeeded=0),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
        ),
        # Number of replicated job statuses do not match slices.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=0, ready=1, succeeded=0),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.UNKNOWN,
        ),
        # All replicated jobs succeeded. No need to wait for jobset conditions.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=0, ready=0, succeeded=2),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.SUCCEEDED,
        ),
        # Ignore active and missing statuses.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(active=1, ready=1),
                ],
            ),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.READY,
        ),
        # Missing jobset is reported as "not started".
        dict(
            status=k8s.client.exceptions.ApiException(status=404),
            num_slices=1,
            expected=gke_runner.GKERunnerJob.Status.NOT_STARTED,
        ),
        # All statuses are 0.
        dict(
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=0, ready=0, succeeded=0),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.PENDING,
        ),
        # All statuses are 0 and tiers do not match (thus will be recreated).
        dict(
            tier="0",
            status=dict(
                replicatedJobsStatus=[
                    dict(failed=0, ready=0, succeeded=0),
                ],
            ),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
        ),
        # Jobset reservation and bastion tier do not match.
        dict(
            tier="1",
            status={},
            spec=dict(replicatedJobs=_mock_replicated_jobs(["test-reservation"])),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
        ),
        # Jobset reservation and bastion tier do not match.
        dict(
            tier="1",
            status={},
            spec=dict(replicatedJobs=_mock_replicated_jobs(["spot", "test-reservation"])),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.RESCHEDULED,
        ),
        # Jobset reservation and bastion tier do not match.
        # In this case, we allow the job to keep running.
        dict(
            tier="0",
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
            status=dict(
                replicatedJobsStatus=[
                    dict(active=2, ready=2),
                ],
            ),
            spec=dict(replicatedJobs=[{"template": {}}]),
            num_slices=2,
            expected=gke_runner.GKERunnerJob.Status.READY,
        ),
    )
    def test_get_status(
        self,
        status: dict,
        num_slices: int,
        expected: gke_runner.GKERunnerJob.Status,
        tier: Optional[str] = None,
        spec: Optional[dict] = None,
    ):
        with self._job_config("test-name", "test-cluster", "test-sa") as (cfg, _):
            cfg.inner.accelerator.set(instance_type="v4-8", num_replicas=num_slices)
            cfg.bundler.set(image="test")
            job: gke_runner.TPUGKERunnerJob = cfg.set(command="").instantiate()

            if isinstance(status, Exception):
                mock_get_status = mock.Mock(side_effect=status)
            else:
                mock_get_status = mock.Mock(return_value=dict(status=status, spec=spec))

            with (
                mock.patch.dict("os.environ", {"BASTION_TIER": tier}),
                mock.patch(
                    "kubernetes.client.CustomObjectsApi",
                    return_value=mock.Mock(get_namespaced_custom_object_status=mock_get_status),
                ),
            ):
                self.assertEqual(expected, job._get_status())

    @parameterized.parameters(
        # Don't need to reschedule if no node-pool exists.
        dict(node_pool=None, expect_delete_count=0),
        # Test a case when tier=0 matches reservation.
        dict(
            node_pool={
                "name": "pool0",
                "config": {
                    "reservationAffinity": {
                        "key": "compute.googleapis.com/reservation-name",
                        "values": ["test-reservation"],
                    },
                    "labels": {"provisioner-nodepool-id": "test-name"},
                },
            },
            tier=0,
            expect_delete_count=0,
        ),
        # Test a case when tier=1 matches spot.
        dict(
            node_pool={
                "name": "pool0",
                "config": {
                    "taints": [
                        {
                            "key": "cloud.google.com/gke-spot",
                            "value": "true",
                            "effect": "NO_SCHEDULE",
                        }
                    ],
                    "labels": {"provisioner-nodepool-id": "pool0"},
                },
            },
            tier=1,
            expect_delete_count=0,
        ),
        # Tier=0 doesn't match spot, and delete goes through the first time.
        dict(
            node_pool={
                "name": "pool0",
                "config": {
                    "taints": [
                        {
                            "key": "cloud.google.com/gke-spot",
                            "value": "true",
                            "effect": "NO_SCHEDULE",
                        }
                    ],
                    "labels": {"provisioner-nodepool-id": "pool0"},
                },
            },
            tier=0,
            expect_delete_count=1,
        ),
        # Tier=1 doesn't match reservation, and delete goes through the first time.
        dict(
            node_pool={
                "name": "pool0",
                "config": {
                    "reservationAffinity": {
                        "key": "compute.googleapis.com/reservation-name",
                        "values": ["test-reservation"],
                    },
                    "labels": {"provisioner-nodepool-id": "pool0"},
                },
            },
            tier=1,
            expect_delete_count=1,
        ),
        # Test that we retry deletes.
        dict(
            node_pool={
                "name": "pool0",
                "config": {
                    "reservationAffinity": {
                        "key": "compute.googleapis.com/reservation-name",
                        "values": ["test-reservation"],
                    },
                    "labels": {"provisioner-nodepool-id": "pool0"},
                },
            },
            tier=1,
            expect_delete_count=2,
            delete_node_pool=[
                errors.HttpError(resp=mock.Mock(status=400), content="Conflict".encode("utf-8")),
                None,
            ],
        ),
    )
    def test_reschedule(self, node_pool, expect_delete_count, delete_node_pool=None, tier=None):
        with self._job_config("test-name", "test-cluster", "test-sa") as (cfg, _):
            cfg.bundler.set(image="test")
            # Node pool test cases assume "test-name".
            self.assertEqual("test-name", cfg.name)

            job: gke_runner.TPUGKERunnerJob = cfg.set(
                command="",
                status_interval_seconds=0,
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
            )
            mock_get_node_pool = mock.Mock(return_value=node_pool)
            mock_delete_node_pool = mock.Mock(side_effect=delete_node_pool or [None])
            mock_node_pool = mock.patch.multiple(
                gke_runner.__name__,
                _node_pool_resource=mock.DEFAULT,
                _delete_node_pool=mock_delete_node_pool,
                _get_node_pool_by_label=mock_get_node_pool,
                get_credentials=mock.DEFAULT,
            )
            mock_env = mock.patch("os.environ", {"BASTION_TIER": tier} if tier is not None else {})
            with mock_env, mock_job, mock_node_pool:
                job._reschedule(max_tries=2, retry_interval=0.1)
                cluster_id = (
                    "projects/settings-project/" "locations/settings-zone/" "clusters/test-cluster"
                )
                self.assertEqual(
                    {
                        "label": "test-name",
                        "parent": cluster_id,
                    },
                    mock_get_node_pool.call_args[1],
                )
                self.assertEqual(expect_delete_count, mock_delete_node_pool.call_count)
                if expect_delete_count:
                    self.assertEqual(
                        {"name": f"{cluster_id}/nodePools/{node_pool['name']}"},
                        mock_delete_node_pool.call_args[1],
                    )
                # Jobset should always be deleted.
                job._delete.assert_called()  # pytype: disable=attribute-error


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    @parameterized.parameters(
        dict(node_pools={}, label="test", expected=None),
        dict(node_pools={"nodePools": []}, label="test", expected=None),
        dict(
            node_pools={
                "nodePools": [
                    {"name": "pool0", "config": {"labels": {"provisioner-nodepool-id": "hello"}}},
                    {"name": "pool1", "config": {"labels": {"provisioner-nodepool-id": "test"}}},
                ]
            },
            label="test",
            expected={"name": "pool1", "config": {"labels": {"provisioner-nodepool-id": "test"}}},
        ),
    )
    def test_get_node_pool_by_label(self, node_pools, label, expected):
        with mock.patch(f"{gke_runner.__name__}._list_node_pools", return_value=node_pools):
            self.assertEqual(
                expected,
                _get_node_pool_by_label(
                    mock.MagicMock(),
                    label=label,
                    parent="projects/test-project/locations/test-region/clusters/test-cluster",
                ),
            )


class MainTest(parameterized.TestCase):
    """Tests CLI entrypoint."""

    @parameterized.parameters(
        dict(instance_type="tpu", expected=gke_runner.TPUGKERunnerJob),
        dict(instance_type="tpu-v4-8", expected=gke_runner.TPUGKERunnerJob),
        dict(instance_type="gpu", expected=app.UsageError("instance_type")),
    )
    def test_get_runner_or_exit(self, instance_type: str, expected: Union[Exception, Type]):
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                _get_runner_or_exit(instance_type)
        else:
            self.assertEqual(expected, _get_runner_or_exit(instance_type))

    @parameterized.parameters("start", "stop")
    def test_load_kube_config(self, action):
        # load_kube_config should only be called if using gke action.
        mock_settings = {
            "project": "settings-project",
            "zone": "settings-zone",
            "gke_cluster": "settings-cluster",
        }
        mock_job = _mock_job(
            gke_runner.TPUGKERunnerJob,
            bundler_kwargs={},
            settings_kwargs=mock_settings,
        )
        mock_utils = mock.patch.multiple(
            gke_runner.__name__,
            load_kube_config=mock.DEFAULT,
            delete_k8s_jobset=mock.DEFAULT,
            with_tpu_training_defaults=mock.DEFAULT,
        )
        with mock_gcp_settings(gke_runner.__name__, mock_settings), mock_job, mock_utils as m:
            fv = flags.FlagValues()
            gke_runner.TPUGKERunnerJob.define_flags(fv)
            fv.set_default("name", "test")
            fv.set_default("instance_type", "tpu-v4-8")
            fv.mark_as_parsed()
            gke_runner.main(["cli", action, "test_command"], flag_values=fv)
            call_kwargs = m["load_kube_config"].call_args[1]
            self.assertEqual("settings-project", call_kwargs["project"])
            self.assertEqual("settings-zone", call_kwargs["zone"])
            self.assertEqual("settings-cluster", call_kwargs["cluster"])
