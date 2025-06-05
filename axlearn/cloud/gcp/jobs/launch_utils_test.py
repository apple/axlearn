# Copyright © 2024 Apple Inc.

"""Tests launch utilities."""
# pylint: disable=protected-access

import contextlib
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
from axlearn.cloud.common.utils import FlagConfigurable, Table, define_flags
from axlearn.cloud.gcp.jobs import launch_utils
from axlearn.cloud.gcp.jobs.launch import BaseBastionManagedJob, _JobType
from axlearn.cloud.gcp.jobs.launch_utils import (
    _parse_resource_flags_from_command,
    infer_module_qualname,
    jobs_table,
    match_by_regex,
    match_gcp_api,
    project_usage_table,
    serialized_flags_for_config,
    user_usage_table,
    validate_resource_flags,
    with_k8s_jobset_state,
)
from axlearn.cloud.gcp.utils import GCPAPI
from axlearn.common.config import config_class


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    def test_infer_module_qualname(self):
        self.assertEqual(
            "axlearn.cloud.gcp.jobs.launch", infer_module_qualname(BaseBastionManagedJob)
        )

    def test_serialized_flags_for_config(self):
        fv = flags.FlagValues()
        flags.DEFINE_string("test_discarded", None, "Test discarded flag", flag_values=fv)

        class Child(FlagConfigurable):
            @classmethod
            def define_flags(cls, fv):
                flags.DEFINE_string("test_inner", "inner", "Inner flag", flag_values=fv)

        class DummyJob(Job):
            @config_class
            class Config(Job.Config):
                # Test that child configs are also included.
                inner: FlagConfigurable.Config = Child.default_config()

            @classmethod
            def define_flags(cls, fv):
                flags.DEFINE_string("test_kept", "value", "Test kept flag", flag_values=fv)
                flags.DEFINE_multi_string(
                    "test_multi",
                    ["value1", "value2"],
                    "Test kept multi-flag",
                    flag_values=fv,
                )

        define_flags(DummyJob.default_config(), fv)
        self.assertEqual(
            [
                "--test_kept=value",
                "--test_multi=value1",
                "--test_multi=value2",
                "--test_inner=inner",
            ],
            serialized_flags_for_config(DummyJob.default_config(), fv),
        )

    @parameterized.parameters(
        # Matches both upper/lowercase.
        dict(gcp_api=GCPAPI.GKE.value.lower(), expected=True),
        dict(gcp_api=GCPAPI.GKE.value.upper(), expected=True),
        dict(gcp_api="other", expected=False),
    )
    def test_match_gcp_api(self, gcp_api: str, expected):
        fv = flags.FlagValues()
        flags.DEFINE_string("gcp_api", gcp_api, "", flag_values=fv)
        fv.mark_as_parsed()
        self.assertEqual(expected, match_gcp_api(GCPAPI.GKE.value)(action="start", flag_values=fv))

    @parameterized.parameters(
        # Matches TPU types.
        dict(
            matcher=match_by_regex(
                match_regex=dict(instance_type=r"v(\d)+.*-(\d)+", job_type=_JobType.DEFAULT.value),
            ),
            cases=[
                dict(
                    instance_type="v4-8",
                    job_type=_JobType.DEFAULT.value,
                    expected=True,
                ),
                dict(
                    instance_type="v5litepod-16",
                    job_type=_JobType.DEFAULT.value,
                    expected=True,
                ),
                dict(
                    instance_type="tpu",
                    job_type=_JobType.DEFAULT.value,
                    expected=False,
                ),
                dict(
                    instance_type="v4-8",
                    job_type=_JobType.FLINK.value,
                    expected=False,
                ),
            ],
        ),
        dict(
            matcher=match_by_regex(
                match_regex=dict(
                    action="list", instance_type=".*", job_type=_JobType.DEFAULT.value
                ),
            ),
            cases=[
                dict(
                    instance_type="v4-8",
                    job_type=_JobType.DEFAULT.value,
                    action="list",
                    expected=True,
                ),
                dict(
                    instance_type="v4-8",
                    job_type=_JobType.DEFAULT.value,
                    action="start",
                    expected=False,
                ),
            ],
        ),
    )
    def test_match_by_regex(self, matcher, cases):
        for case in cases:
            fv = flags.FlagValues()
            flags.DEFINE_string("instance_type", case["instance_type"], "", flag_values=fv)
            flags.DEFINE_string("job_type", case["job_type"], "", flag_values=fv)
            fv.mark_as_parsed()
            self.assertEqual(
                case["expected"],
                matcher(action=case.get("action", "start"), flag_values=fv),
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
            table = with_k8s_jobset_state(jobs_table, namespace=_JobType.DEFAULT.value)(
                self._mock_jobs
            )
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
