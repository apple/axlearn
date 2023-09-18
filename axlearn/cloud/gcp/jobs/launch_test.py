# Copyright © 2023 Apple Inc.

"""Tests launchers."""
# pylint: disable=protected-access

from datetime import datetime
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.job import Job
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.gcp.jobs import launch
from axlearn.cloud.gcp.jobs.bastion_vm import Job as BastionJob
from axlearn.cloud.gcp.jobs.bastion_vm import JobState as BastionJobState
from axlearn.cloud.gcp.jobs.bastion_vm import deserialize_jobspec, new_jobspec
from axlearn.cloud.gcp.jobs.launch import (
    BaseBastionLaunchJob,
    Launcher,
    LaunchTPUJob,
    _get_launcher_or_exit,
    _match_by_regex,
)
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    @parameterized.parameters(
        # Matches any "start" command.
        dict(
            matcher=_match_by_regex(match_regex=dict(start=".*")),
            cases=[
                dict(action="start", instance_type="", expected=True),
                dict(action="start", instance_type="test type", expected=True),
                # Missing matcher for list.
                dict(action="list", instance_type="", expected=False),
            ],
        ),
        # Matches TPU types.
        dict(
            matcher=_match_by_regex(match_regex=dict(start=r"v(\d)+.*-(\d)+", list="tpu")),
            cases=[
                dict(action="start", instance_type="v4-8", expected=True),
                dict(action="start", instance_type="v5litepod-16", expected=True),
                dict(action="start", instance_type="tpu", expected=False),
                dict(action="list", instance_type="tpu", expected=True),
            ],
        ),
    )
    def test_match_by_regex(self, matcher, cases):
        for case in cases:
            self.assertEqual(case["expected"], matcher(case["action"], case["instance_type"]))

    def test_get_launcher_or_exit(self):
        def match_tpu(action, instance_type):
            del action
            return instance_type == "tpu"

        class DummyTPULauncher(Job):
            pass

        mock_launchers = [
            Launcher(job_cls=DummyTPULauncher, matcher=match_tpu, description="test"),
        ]
        with mock.patch(f"{launch.__name__}._LAUNCHERS", mock_launchers):
            launcher = _get_launcher_or_exit(action="start", instance_type="tpu")
            self.assertEqual(launcher.job_cls, DummyTPULauncher)
            self.assertEqual(launcher.matcher, match_tpu)

            with self.assertRaises(app.UsageError):
                _get_launcher_or_exit(action="start", instance_type="other")


class TestBaseBastionLaunchJob(parameterized.TestCase):
    """Tests BaseBastionLaunchJob."""

    def _mock_config(self, **kwargs):
        cfg = BaseBastionLaunchJob.default_config().set(
            instance_type="test",
            user_id="test_user",
            project_id=None,
            name="test_job",
            command="test_command",
            max_tries=1,
            retry_interval=60,
        )
        cfg.set(**kwargs)

        def dummy_submit_job(name: str, job_spec_file: str = ""):
            del name
            if job_spec_file:
                with open(job_spec_file, "r", encoding="utf-8") as f:
                    spec = deserialize_jobspec(f)
                    self.assertEqual(spec.name, cfg.name)
                    self.assertEqual(spec.command, cfg.command)
                    self.assertEqual(spec.metadata.user_id, cfg.user_id)
                    self.assertEqual(spec.metadata.project_id, cfg.project_id or "none")
            return mock.MagicMock()

        return cfg.set(bastion=config_for_function(dummy_submit_job).set(name="test_bastion"))

    def test_start(self):
        # Test with defaults.
        job = self._mock_config().instantiate()
        job._execute()

        # Test with bundler.
        bundler = mock.MagicMock()
        job = self._mock_config(bundler=config_for_function(lambda: bundler)).instantiate()
        job._execute()
        self.assertTrue(bundler.bundle.called)

        # Test with invalid project id.
        project_id = "test_project"
        patch_fns = mock.patch.multiple(
            launch.__name__,
            gcp_settings=mock.Mock(return_value=""),
            get_user_projects=mock.Mock(return_value=["other_project"]),
        )
        with patch_fns, self.assertRaisesRegex(ValueError, "other_project"):
            job = self._mock_config(project_id=project_id).instantiate()
            job._execute()

        # Test with valid project id.
        project_id = "test_project"
        patch_fns = mock.patch.multiple(
            launch.__name__,
            gcp_settings=mock.Mock(return_value=""),
            get_user_projects=mock.Mock(return_value=["test_project"]),
        )
        with patch_fns:
            job = self._mock_config(project_id=project_id).instantiate()
            job._execute()

    def test_list(self):
        mock_jobs = {
            "job0": BastionJob(
                spec=new_jobspec(
                    name="test_job0",
                    command="command",
                    metadata=JobMetadata(
                        user_id="test_user",
                        project_id="test_project",
                        creation_time=datetime.now(),
                        resources={"v4": 8},
                    ),
                ),
                state=BastionJobState.PENDING,
                command_proc=None,
                cleanup_proc=None,
            ),
            "job1": BastionJob(
                spec=new_jobspec(
                    name="test_job1",
                    command="command",
                    metadata=JobMetadata(
                        user_id="test_user1",
                        project_id="test_project",
                        creation_time=datetime.now(),
                        resources={"v4": 8, "v5": 16},
                    ),
                ),
                state=BastionJobState.ACTIVE,
                command_proc=None,
                cleanup_proc=None,
            ),
            "job2": BastionJob(
                spec=new_jobspec(
                    name="test_job2",
                    command="command",
                    metadata=JobMetadata(
                        user_id="test_user1",
                        project_id="test_project1",
                        creation_time=datetime.now(),
                        resources={"v4": 16},
                    ),
                ),
                state=BastionJobState.ACTIVE,
                command_proc=None,
                cleanup_proc=None,
            ),
        }
        with mock.patch(f"{launch.__name__}.download_job_batch", return_value=mock_jobs):
            job = self._mock_config().instantiate()
            out = job._list()
            self.assertEqual(out["jobs"], mock_jobs)
            self.assertEqual(
                job._jobs_table(out["jobs"]),
                dict(
                    headings=[
                        "NAME",
                        "USER_ID",
                        "JOB_STATE",
                        "PROJECT_ID",
                        "RESOURCES",
                        "PRIORITY",
                    ],
                    rows=[
                        [
                            "test_job0",
                            "test_user",
                            BastionJobState.PENDING,
                            "test_project",
                            "{'v4': 8}",
                            "5",
                        ],
                        [
                            "test_job1",
                            "test_user1",
                            BastionJobState.ACTIVE,
                            "test_project",
                            "{'v4': 8, 'v5': 16}",
                            "5",
                        ],
                        [
                            "test_job2",
                            "test_user1",
                            BastionJobState.ACTIVE,
                            "test_project1",
                            "{'v4': 16}",
                            "5",
                        ],
                    ],
                ),
            )
            self.assertEqual(
                job._usage_table(out["usage_by_project"]),
                dict(
                    headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                    rows=[
                        # Note that pending jobs are excluded from usage.
                        ["test_project", "v5", 16, 1],
                        ["test_project1", "v4", 16, 1],
                        ["test_project", "v4", 8, 1],
                    ],
                ),
            )
            self.assertEqual(
                job._usage_table(out["usage_by_user"]),
                dict(
                    headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                    rows=[
                        ["test_user1", "v5", 16, 1],
                        ["test_user1", "v4", 24, 2],
                    ],
                ),
            )


class TestLaunchTPUJob(TestWithTemporaryCWD):
    """Tests LaunchTPUJob."""

    def test_flags(self):
        fv = flags.FlagValues()
        flags.DEFINE_string("instance_type", "test-type", help="test", flag_values=fv)
        fv.mark_as_parsed()

        patch_fns = mock.patch.multiple(
            launch.__name__,
            shared_bastion_name=mock.Mock(return_value="shared-bastion"),
            generate_job_name=mock.Mock(return_value="job-name"),
        )
        with patch_fns:
            LaunchTPUJob.define_flags(fv)

        # Check some basic flags.
        self.assertEqual(fv.bastion, "shared-bastion")
        self.assertEqual(fv.name, "job-name")
        self.assertIn("tpu_type", fv)
        self.assertIn("bundler_type", fv)
        self.assertIsNotNone(fv["name"].default)
        self.assertIsNotNone(fv["bundler_type"].default)
        self.assertEqual(fv["tpu_type"].default, "test-type")

        # Make sure bundler config is constructed properly.
        cfg = LaunchTPUJob.from_flags(fv, command="test command")
        self.assertIn("tpu", cfg.bundler.extras)

        # Make sure command is expected.
        for flag in ["name", "bundler_type", "tpu_type"]:
            self.assertIn(f"--{flag}={fv[flag].default}", cfg.command)
        self.assertIn("test command", cfg.command)
