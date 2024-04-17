# Copyright © 2023 Apple Inc.

"""Tests launchers."""
# pylint: disable=protected-access

from datetime import datetime
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobState as BastionJobState
from axlearn.cloud.common.bastion import deserialize_jobspec, new_jobspec
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.jobs import bastion_vm, launch, tpu_runner
from axlearn.cloud.gcp.jobs.launch import (
    BaseBastionManagedJob,
    BastionDirectory,
    BastionManagedTPUJob,
    Launcher,
    _get_launcher_or_exit,
)
from axlearn.cloud.gcp.jobs.launch_utils import jobs_table, usage_table
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

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


class TestBaseBastionManagedJob(parameterized.TestCase):
    """Tests BaseBastionManagedJob."""

    def _mock_config(self, **kwargs):
        class DummyRunner(Job):
            def _execute(self):
                pass

        cfg = BaseBastionManagedJob.default_config().set(
            instance_type="test",
            user_id="test_user",
            project_id=None,
            zone="test_zone",
            name="test_job",
            command="test_command",
            max_tries=1,
            retry_interval=60,
            bastion_name="test_bastion",
            priority=3,
            output_dir="test-output",
            runner=DummyRunner.default_config().set(
                max_tries=1,
                retry_interval=60,
                command="",
            ),
        )
        cfg.set(**kwargs)
        test_fixture = self

        class FakeBastionDirectory(BastionDirectory):
            def submit_job(self, job_name: str, *, job_spec_file: str):
                test_fixture.assertEqual("temp_dir", self.config.root_dir)
                with open(job_spec_file, "r", encoding="utf-8") as f:
                    spec = deserialize_jobspec(f)
                    test_fixture.assertEqual(spec.name, cfg.name)
                    test_fixture.assertEqual(spec.command, cfg.command)
                    test_fixture.assertEqual(spec.metadata.user_id, cfg.user_id)
                    test_fixture.assertEqual(spec.metadata.project_id, cfg.project_id or "none")
                    test_fixture.assertEqual(spec.metadata.priority, cfg.priority)

        return cfg.set(bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir"))

    @parameterized.parameters(
        dict(output_dir="", instance_type="test", expected=ValueError("output_dir")),
        dict(
            output_dir="test-output",
            instance_type="",
            expected=ValueError("instance_type"),
        ),
    )
    def test_empty(self, expected, **kwargs):
        # Ensure that output_dir is provided.
        cfg = self._mock_config().set(**kwargs)
        with self.assertRaisesRegex(type(expected), str(expected)):
            cfg.instantiate()

    def test_start(self):
        mock_get_vm_node = mock.patch(
            f"{launch.__name__}._get_bastion_vm", return_value=dict(status="RUNNING")
        )
        with mock_get_vm_node:
            # Test with defaults.
            job = self._mock_config().instantiate()
            job._execute()

            # Test with bundler.
            mock_bundler = mock.MagicMock()
            cfg = self._mock_config()
            cfg.runner.bundler = config_for_function(lambda: mock_bundler)
            job = cfg.instantiate()
            job._execute()
            self.assertTrue(mock_bundler.bundle.called)

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
        job = self._mock_config().instantiate()
        with mock.patch.object(job._bastion_dir, "list_jobs", return_value=mock_jobs):
            out = job._list()
            self.assertEqual(out["jobs"], mock_jobs)
            self.assertEqual(
                jobs_table(out["jobs"]),
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
                usage_table(out["usage_by_project"]),
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
                usage_table(out["usage_by_user"]),
                dict(
                    headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                    rows=[
                        ["test_user1", "v5", 16, 1],
                        ["test_user1", "v4", 24, 2],
                    ],
                ),
            )


class TestBastionManagedTPUJob(TestWithTemporaryCWD):
    """Tests BastionManagedTPUJob."""

    @parameterized.product(
        name=[None, "test-name"],
        output_dir=[None, "test-output"],
        zone=[None, "test-zone"],
        action=["start", "list"],
        bastion=[None, "test-bastion"],
        bundler_type=[None, "artifactregistry"],
    )
    def test_flags(self, name, output_dir, zone, action, bastion, bundler_type):
        # Construct flags.
        fv = flags.FlagValues()
        flags.DEFINE_string("instance_type", "test-type", help="test", flag_values=fv)
        fv.mark_as_parsed()

        patch_fns = mock.patch.multiple(
            launch.__name__,
            generate_job_name=mock.Mock(return_value="job-name"),
        )
        mock_settings = {
            "ttl_bucket": "ttl_bucket",
            "permanent_bucket": "permanent_bucket",
            "zone": zone or "default-zone",
            "docker_repo": "test-repo",
            "default_dockerfile": "test-dockerfile",
        }
        patch_settings = mock_gcp_settings(
            [launch.__name__, bastion_vm.__name__, tpu_runner.__name__, bundler.__name__],
            settings=mock_settings,
        )

        with patch_fns, patch_settings:
            BastionManagedTPUJob.define_flags(fv)

            # Parse argv.
            argv = ["cli"]
            if name is not None:
                argv.append(f"--name={name}")
            if output_dir is not None:
                argv.append(f"--output_dir={output_dir}")
            if zone is not None:
                argv.append(f"--zone={zone}")
            if bastion is not None:
                argv.append(f"--bastion={bastion}")
            if bundler_type is not None:
                argv.append(f"--bundler_type={bundler_type}")
                argv.append("--bundler_spec=image=test")
            fv(argv)

            # Check some basic flags.
            self.assertEqual(fv.bastion, bastion)
            self.assertEqual(fv.name, name or "job-name")
            self.assertEqual(fv.zone, zone)
            self.assertIn("tpu_type", fv)
            self.assertIn("bundler_type", fv)
            self.assertIsNotNone(fv["name"].default)
            self.assertEqual(fv["tpu_type"].default, "test-type")
            self.assertEqual(fv.output_dir, output_dir)

            cfg = BastionManagedTPUJob.from_flags(fv, command="test command", action=action)

            self.assertIsNone(cfg.bundler)
            if action == "start":
                self.assertIsNotNone(cfg.runner)
                self.assertIsNotNone(cfg.runner.bundler)
                self.assertIn("tpu", cfg.runner.bundler.extras)
            else:
                self.assertIsNone(cfg.runner)

            # Check bastion flag. If None, we should infer from zone in mock_settings.
            if bastion:
                self.assertEqual(bastion, cfg.bastion_name)
            elif zone is None:
                self.assertEqual(
                    f"{mock_settings['zone']}-{bastion_vm._SHARED_BASTION_SUFFIX}",
                    cfg.bastion_name,
                )
            else:
                self.assertEqual(f"{zone}-{bastion_vm._SHARED_BASTION_SUFFIX}", cfg.bastion_name)

            # Check output_dir.
            if output_dir is None:
                self.assertEqual(cfg.output_dir, f"gs://ttl_bucket/axlearn/jobs/{fv.name}")
            else:
                self.assertEqual(cfg.output_dir, output_dir)

        if action == "start":
            # Make sure command is expected.
            for flag in ["name", "bundler_type", "tpu_type"]:
                if fv[flag].value is not None:
                    self.assertIn(f"--{flag}={fv[flag].value}", cfg.command)
            self.assertIn("test command", cfg.command)
        else:
            self.assertIsNone(cfg.command)

        # Should be instantiable.
        job: BastionManagedTPUJob = cfg.instantiate()
        # Bundler should be propagated to runner.
        if action == "start":
            self.assertIsNotNone(job.runner.bundler)
