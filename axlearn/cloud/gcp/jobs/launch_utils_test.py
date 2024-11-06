# Copyright © 2024 Apple Inc.

"""Tests launch utilities."""
# pylint: disable=protected-access

import contextlib
import dataclasses
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Union
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobMetadata, JobSpec, JobState, JobStatus
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import Table
from axlearn.cloud.gcp.jobs import launch_utils
from axlearn.cloud.gcp.jobs.launch_utils import (
    _parse_resource_flags_from_command,
    jobs_table,
    match_by_regex,
    project_usage_table,
    serialized_flags_for_job,
    user_usage_table,
    validate_resource_flags,
    with_k8s_jobset_state,
    with_qrm_tpu_state,
)
from axlearn.cloud.gcp.tpu import TpuInfo
from axlearn.cloud.gcp.utils import GCPAPI


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    def test_serialized_flags_for_job(self):
        fv = flags.FlagValues()
        flags.DEFINE_string("test_discarded", None, "Test discarded flag", flag_values=fv)

        class DummyJob(Job):
            @classmethod
            def define_flags(cls, fv):
                flags.DEFINE_string("test_kept", "value", "Test kept flag", flag_values=fv)
                flags.DEFINE_multi_string(
                    "test_multi",
                    ["value1", "value2"],
                    "Test kept multi-flag",
                    flag_values=fv,
                )

        DummyJob.define_flags(fv)
        self.assertEqual(
            ["--test_kept=value", "--test_multi=value1", "--test_multi=value2"],
            serialized_flags_for_job(fv, job=DummyJob),
        )

    @parameterized.parameters(
        # Matches any "start" command.
        dict(
            matcher=match_by_regex(match_regex=dict(start=".*"), gcp_api=GCPAPI.QRM.value),
            cases=[
                dict(action="start", instance_type="", gcp_api=GCPAPI.QRM.value, expected=True),
                dict(
                    action="start",
                    instance_type="test type",
                    gcp_api=GCPAPI.QRM.value,
                    expected=True,
                ),
                # Missing matcher for list.
                dict(action="list", instance_type="", gcp_api=GCPAPI.QRM.value, expected=False),
                # Does not match GKE.
                dict(action="start", instance_type="", gcp_api=GCPAPI.GKE.value, expected=False),
                # Matches both upper/lowercase.
                dict(
                    action="start",
                    instance_type="v4-8",
                    gcp_api=GCPAPI.QRM.value.lower(),
                    expected=True,
                ),
            ],
        ),
        # Matches TPU types.
        dict(
            matcher=match_by_regex(
                match_regex=dict(start=r"v(\d)+.*-(\d)+", list="tpu"),
                gcp_api=GCPAPI.GKE.value,
            ),
            cases=[
                dict(action="start", instance_type="v4-8", gcp_api=GCPAPI.GKE.value, expected=True),
                dict(
                    action="start",
                    instance_type="v5litepod-16",
                    gcp_api=GCPAPI.GKE.value,
                    expected=True,
                ),
                dict(action="start", instance_type="tpu", gcp_api=GCPAPI.GKE.value, expected=False),
                dict(action="list", instance_type="tpu", gcp_api=GCPAPI.GKE.value, expected=True),
                # Does not match QRM.
                dict(
                    action="start", instance_type="v4-8", gcp_api=GCPAPI.QRM.value, expected=False
                ),
                # Matches both upper/lowercase.
                dict(
                    action="start",
                    instance_type="v4-8",
                    gcp_api=GCPAPI.GKE.value.lower(),
                    expected=True,
                ),
            ],
        ),
    )
    def test_match_by_regex(self, matcher, cases):
        for case in cases:
            self.assertEqual(
                case["expected"],
                matcher(
                    action=case["action"],
                    instance_type=case["instance_type"],
                    gcp_api=case["gcp_api"],
                ),
            )

    @parameterized.parameters(
        dict(
            command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update -"
            "-enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep infinity",
            enable_pre_provisioner=True,
            instance_type="tpu-v5litepod-16",
            num_replicas=1,
        ),
        dict(
            command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--noenable_pre_provisioner --tpu_type=tpu-v5litepod-32 --num_slices=2 "
            "-- sleep infinity",
            enable_pre_provisioner=False,
            instance_type="tpu-v5litepod-32",
            num_replicas=2,
        ),
        dict(
            command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--tpu_type=tpu-v5litepod-32 --num_slices=2 "
            "-- sleep infinity",
            enable_pre_provisioner=None,
            instance_type="tpu-v5litepod-32",
            num_replicas=2,
        ),
    )
    def test_parse_resource_flags_from_command(
        self, command, enable_pre_provisioner, instance_type, num_replicas
    ):
        parsed_flags = _parse_resource_flags_from_command(command)

        self.assertEqual(parsed_flags.enable_pre_provisioner, enable_pre_provisioner)
        self.assertEqual(parsed_flags.instance_type, instance_type)
        self.assertEqual(parsed_flags.num_replicas, num_replicas)

    @parameterized.parameters(
        dict(
            original_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep infinity",
            updated_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep 30",
            expected=None,
        ),
        dict(
            original_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep infinity",
            updated_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-32 --num_replicas=1 "
            "-- sleep infinity",
            expected=ValueError("instance_type"),
        ),
        dict(
            original_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep infinity",
            updated_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update "
            "--enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_slices=2 "
            "-- sleep infinity",
            expected=ValueError("num_replicas"),
        ),
        dict(
            original_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update"
            " --instance_type=tpu-v5litepod-16 --num_replicas=1 -- sleep infinity",
            updated_command="python3 -m axlearn.cloud.gcp.jobs.gke_runner update"
            " --enable_pre_provisioner --instance_type=tpu-v5litepod-16 --num_replicas=1 "
            "-- sleep infinity",
            expected=ValueError("pre_provisioner"),
        ),
    )
    def test_validate_resource_flags(
        self, original_command, updated_command, expected: Union[Exception, type]
    ):
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            validate_resource_flags(original_command, updated_command)


class TestListUtils(parameterized.TestCase):
    """Tests list utils."""

    _mock_jobs = {
        f"job_{project}{user}{i}": BastionJob(
            spec=JobSpec(
                version=1,
                name=f"job_{project}{user}{i}",
                command=f"command {i}",
                cleanup_command=f"cleanup {i}",
                env_vars={},
                metadata=JobMetadata(
                    user_id=f"user_{user}",
                    project_id=f"project_{project}",
                    creation_time=datetime.now(),
                    resources={"v4": 8},
                    job_id="test-id",
                ),
            ),
            state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
            command_proc=None,
            cleanup_proc=None,
        )
        for project in range(2)
        for user in range(2)
        for i in range(2)
    }

    def test_jobs_table(self):
        print(jobs_table(self._mock_jobs))
        self.assertEqual(
            Table(
                headings=[
                    "NAME",
                    "USER_ID",
                    "JOB_STATE",
                    "PROJECT_ID",
                    "RESOURCES",
                    "PRIORITY",
                    "JOB_ID",
                    "TIER",
                ],
                rows=[
                    [
                        f"job_{p}{u}{i}",
                        f"user_{u}",
                        "ACTIVE",
                        f"project_{p}",
                        "{'v4': 8}",
                        "5",
                        "test-id",
                        "0",
                    ]
                    for p in range(2)
                    for u in range(2)
                    for i in range(2)
                ],
            ),
            jobs_table(self._mock_jobs),
        )

    def test_user_usage_table(self):
        self.assertEqual(
            Table(
                headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                rows=[["user_0", "v4", 32, 4], ["user_1", "v4", 32, 4]],
            ),
            user_usage_table(self._mock_jobs),
        )

    def test_project_usage_table(self):
        self.assertEqual(
            Table(
                headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                rows=[["project_0", "v4", 32, 4], ["project_1", "v4", 32, 4]],
            ),
            project_usage_table(self._mock_jobs),
        )

    def test_with_qrm_tpu_state(self):
        mock_states = {"job_000": "ACTIVE", "job_001": "DELETING", "job_100": ""}
        mock_qrm_tpus = [
            TpuInfo(name=job_name, accelerator_type="", state=state, metadata={})
            for job_name, state in mock_states.items()
        ]
        with mock.patch.multiple(
            launch_utils.__name__,
            tpu_resource=mock.DEFAULT,
            get_credentials=mock.DEFAULT,
            list_tpu_info=mock.Mock(return_value=mock_qrm_tpus),
        ):
            table = with_qrm_tpu_state(jobs_table)(self._mock_jobs)
            expected = [
                [{mock_states.get(job_name, "PENDING") or "UNKNOWN"}]
                for job_name in self._mock_jobs
            ]
            self.assertEqual(expected, table.get_col("QRM_STATE"))

    @parameterized.parameters(
        "--num_slices=2",
        "--num_slices 2",
        "--num_replicas 2",
    )
    def test_with_qrm_tpu_state_replicas(self, replica_flag):
        # Test a job with multislice.
        job = self._mock_jobs["job_000"]
        mock_jobs = {
            "job_000": dataclasses.replace(
                job,
                spec=dataclasses.replace(job.spec, command=f"{job.spec.command} {replica_flag}"),
            )
        }
        mock_states = {"job_000-0": "ACTIVE", "job_000-1": "PENDING"}
        mock_qrm_tpus = [
            TpuInfo(name=job_name, accelerator_type="", state=state, metadata={})
            for job_name, state in mock_states.items()
        ]
        with mock.patch.multiple(
            launch_utils.__name__,
            tpu_resource=mock.DEFAULT,
            get_credentials=mock.DEFAULT,
            list_tpu_info=mock.Mock(return_value=mock_qrm_tpus),
        ):
            table = with_qrm_tpu_state(jobs_table)(mock_jobs)
            self.assertEqual([[{"ACTIVE", "PENDING"}]], table.get_col("QRM_STATE"))

    def test_with_k8s_jobset_state(self):
        mock_k8s_jobsets = {
            "job_000": [
                SimpleNamespace(status=SimpleNamespace(active=1, ready=1)),
                SimpleNamespace(status=SimpleNamespace(ready=1)),
            ],
            "job_001": [SimpleNamespace(status=SimpleNamespace(failed=1, succeeded=0))],
            "job_100": [SimpleNamespace()],
        }
        with mock.patch(f"{launch_utils.__name__}.list_k8s_jobsets", return_value=mock_k8s_jobsets):
            table = with_k8s_jobset_state(jobs_table, namespace="default")(self._mock_jobs)
            expected = {
                "job_000": {"active": 1, "ready": 2, "failed": 0, "succeeded": 0},
                "job_001": {"active": 0, "ready": 0, "failed": 1, "succeeded": 0},
                "job_100": {"active": 0, "ready": 0, "failed": 0, "succeeded": 0},
            }
            self.assertEqual(
                [
                    [json.dumps(expected[job_name]) if job_name in expected else "PENDING"]
                    for job_name in self._mock_jobs
                ],
                table.get_col("GKE_STATE"),
            )
