# Copyright © 2023 Apple Inc.

"""Tests launchers."""
# pylint: disable=protected-access

import copy
import io
import shlex
from contextlib import redirect_stdout
from datetime import datetime
from typing import Optional, Union
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobState as BastionJobState
from axlearn.cloud.common.bastion import JobStatus, deserialize_jobspec, new_jobspec
from axlearn.cloud.common.bundler import BUNDLE_EXCLUDE, Bundler
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.common.types import JobSpec
from axlearn.cloud.common.utils import FlagConfigurable, define_flags, from_flags
from axlearn.cloud.gcp import bundler, node_pool_provisioner
from axlearn.cloud.gcp.jobs import bastion_vm, launch
from axlearn.cloud.gcp.jobs.launch import (
    _ACTIONS,
    _RUNNER_ACTIONS,
    BaseBastionManagedJob,
    BastionDirectory,
    BastionManagedGKEJob,
    _get_launcher_or_exit,
    _get_runner_or_exit,
    _infer_runner_name,
    _JobType,
    _prelaunch_flags,
    _private_flags,
    _RegistryMember,
)
from axlearn.cloud.gcp.jobs.launch_utils import (
    Table,
    jobs_table,
    project_usage_table,
    user_usage_table,
)
from axlearn.cloud.gcp.runners import gke as runner_gke
from axlearn.cloud.gcp.runners import named_runner_configs
from axlearn.cloud.gcp.runners.base import BaseRunnerJob
from axlearn.cloud.gcp.runners.gke import GKERunnerJob
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.cloud.gcp.utils import GCPAPI
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    def _prelaunch_flags(self):
        fv = flags.FlagValues()
        _prelaunch_flags(fv=fv)
        fv.mark_as_parsed()
        return fv

    def test_get_launcher_or_exit(self):
        fv = self._prelaunch_flags()

        # Without a runner specified for start, should raise.
        with self.assertRaisesRegex(app.UsageError, "runner"):
            _get_launcher_or_exit(action="start", flag_values=fv)

        fv.instance_type = "tpu-v4-8"
        cfg = _get_launcher_or_exit(action="start", flag_values=fv)
        self.assertIsInstance(cfg, BastionManagedGKEJob.Config)

        # Modifying should have not mutate the global config.
        orig_namespace = cfg.namespace
        cfg.namespace = orig_namespace + "test"
        cfg = _get_launcher_or_exit(action="start", flag_values=fv)
        self.assertEqual(cfg.namespace, orig_namespace)

        mock_launchers = [
            _RegistryMember(
                config=BastionManagedGKEJob.default_config(),
                matcher=lambda *_1, **_2: False,
                description="test",
            ),
        ]
        with (
            mock.patch(f"{launch.__name__}._LAUNCHERS", mock_launchers),
            self.assertRaises(app.UsageError),
        ):
            _get_launcher_or_exit(action="start", flag_values=fv)

    @parameterized.parameters(
        dict(instance_type="tpu-v4-8", expected="gke_tpu_single"),
        dict(instance_type="gpu-a3-highgpu-8g", expected="gke_gpu_a3_high_single"),
        dict(job_type=_JobType.FLINK.value, expected="gke_tpu_flink"),
        dict(expected=app.UsageError("runner_name")),
    )
    def test_infer_runner_name(self, expected, **kwargs):
        fv = self._prelaunch_flags()
        for k, v in kwargs.items():
            setattr(fv, k, v)
        if isinstance(expected, app.UsageError):
            with self.assertRaisesRegex(type(expected), str(expected)):
                _infer_runner_name(fv)
        else:
            runner = _infer_runner_name(fv)
            self.assertEqual(runner, expected)
            self.assertIsNotNone(named_runner_configs(expected))  # Sanity check.

    @parameterized.parameters(dict(runner_name="gke_tpu_single", expected=GKERunnerJob.Config))
    def test_get_runner_or_exit(self, runner_name, expected):
        fv = self._prelaunch_flags()
        fv.runner_name = runner_name
        self.assertIsInstance(_get_runner_or_exit(fv), expected)

    def test_private_flags(self):
        fv = flags.FlagValues()
        self.assertNotIn("name", fv)
        _private_flags(fv)
        # Test that launcher flags are registered.
        self.assertIn("name", fv)


class _DummyRunner(BaseRunnerJob):
    def _execute(self):
        pass


class _DummyBundler(Bundler):
    def id(self, name: str):
        return name

    def bundle(self, name: str):
        del name
        pass


def _common_flags(job: BaseBastionManagedJob.Config) -> flags.FlagValues:
    fv = flags.FlagValues()
    define_flags(job, fv)
    kwargs = dict(
        user_id="test_user",
        project_id=None,
        zone="test_zone",
        name="test_job",
        bastion="test_bastion",
        priority=3,
        output_dir="test-output",
        bundler_type="cloudbuild",
        bundler_spec=["image=tpu"],
        # Runner flags.
        project="test_project",
        max_tries=1,
        retry_interval=60,
    )
    for k, v in kwargs.items():
        if k in fv._flags():
            setattr(fv, k, v)
    fv.mark_as_parsed()
    return fv


class TestBaseBastionManagedJob(parameterized.TestCase):
    """Tests BaseBastionManagedJob."""

    def run(self, **kwargs):
        # Run tests under mock settings.
        with mock_gcp_settings([launch.__name__, bastion_vm.__name__, bundler.__name__]):
            return super().run(**kwargs)

    def _mock_config(
        self,
        action="start",
        runner: Optional[BaseRunnerJob.Config] = _DummyRunner.default_config(),
        **kwargs,
    ) -> BaseBastionManagedJob.Config:
        cfg: BaseBastionManagedJob.Config = BaseBastionManagedJob.default_config().set(
            runner=runner
        )
        cfg = from_flags(cfg, _common_flags(cfg), action=action)
        cfg.bundler = _DummyBundler.default_config()
        cfg.set(**kwargs)
        test_fixture = self

        class FakeBastionDirectory(BastionDirectory):
            """A FakeBastionDirectory class."""

            def submit_job(self, job_name: str, *, job_spec_file: str):
                test_fixture.assertEqual("temp_dir", self.config.root_dir)
                with open(job_spec_file, encoding="utf-8") as f:
                    spec = deserialize_jobspec(f)
                    test_fixture.assertEqual(spec.name, cfg.name)
                    test_fixture.assertEqual(spec.command, cfg.command)
                    test_fixture.assertEqual(spec.metadata.user_id, cfg.user_id)
                    test_fixture.assertEqual(spec.metadata.project_id, cfg.project_id or "none")
                    test_fixture.assertEqual(spec.metadata.priority, cfg.priority)
                    test_fixture.assertIsNotNone(spec.metadata.job_id)

        return cfg.set(bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir"))

    def test_output_dir_required(self):
        # Ensure that output_dir is provided.
        with self.assertRaisesRegex(ValueError, "output_dir"):
            self._mock_config().set(output_dir="").instantiate()

    @parameterized.parameters(_ACTIONS)
    def test_validate_bundler(self, action: str):
        """Tests that bundler flags are optional for some actions and required for others."""
        cfg = BaseBastionManagedJob.default_config().set(runner=_DummyRunner.default_config())
        fv = _common_flags(cfg)
        fv.bundler_type = None
        fv.bundler_spec = []

        if action in _RUNNER_ACTIONS:
            with self.assertRaises(Exception):
                from_flags(cfg, fv, action=action)
        else:
            from_flags(cfg, fv, action=action)

    def test_submit(self):
        # Test with defaults.
        cfg = self._mock_config(action="start")
        job = cfg.instantiate()
        job_spec = job.submit()
        self.assertIsNotNone(job_spec)

        # Test with bundler.
        mock_bundler = mock.MagicMock()
        cfg = self._mock_config(action="start")
        cfg.bundler = config_for_function(lambda: mock_bundler)
        job = cfg.instantiate()
        job_spec = job.submit()
        self.assertTrue(mock_bundler.bundle.called)
        self.assertIsNotNone(job_spec)

    def test_run(self):
        # Test with invalid project id.
        project_id = "test_project"
        patch_fns = mock.patch.multiple(
            launch.__name__,
            get_user_projects=mock.Mock(return_value=["other_project"]),
        )
        with patch_fns, self.assertRaisesRegex(ValueError, "other_project"):
            job = self._mock_config(action="run", project_id=project_id).instantiate()
            job.run()

        # Test with valid project id.
        project_id = "test_project"
        patch_fns = mock.patch.multiple(
            launch.__name__,
            get_user_projects=mock.Mock(return_value=["test_project"]),
        )
        with patch_fns:
            job = self._mock_config(action="run", project_id=project_id).instantiate()
            job.run()

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
                        job_id="test-id0",
                    ),
                ),
                state=BastionJobState(status=JobStatus.PENDING),
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
                        job_id="test-id1",
                    ),
                ),
                state=BastionJobState(status=JobStatus.ACTIVE, metadata={"tier": 1}),
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
                        job_id="test-id2",
                    ),
                ),
                state=BastionJobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                command_proc=None,
                cleanup_proc=None,
            ),
        }
        job: BaseBastionManagedJob = self._mock_config().instantiate()
        with mock.patch.object(job._bastion_dir, "list_jobs", return_value=mock_jobs):
            jobs = job.list()
            self.assertEqual(jobs, mock_jobs)
            self.assertEqual(
                jobs_table(jobs),
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
                            "test_job0",
                            "test_user",
                            JobStatus.PENDING,
                            "test_project",
                            "{'v4': 8}",
                            "5",
                            "test-id0",
                            "None",
                        ],
                        [
                            "test_job1",
                            "test_user1",
                            JobStatus.ACTIVE,
                            "test_project",
                            "{'v4': 8, 'v5': 16}",
                            "5",
                            "test-id1",
                            "1",
                        ],
                        [
                            "test_job2",
                            "test_user1",
                            JobStatus.ACTIVE,
                            "test_project1",
                            "{'v4': 16}",
                            "5",
                            "test-id2",
                            "0",
                        ],
                    ],
                ),
            )
            self.assertEqual(
                project_usage_table(jobs),
                Table(
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
                user_usage_table(jobs),
                Table(
                    headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
                    rows=[
                        ["test_user1", "v5", 16, 1],
                        ["test_user1", "v4", 24, 2],
                    ],
                ),
            )


class TestBastionManagedGKEJob(TestWithTemporaryCWD):
    """Tests BastionManagedGKEJob."""

    def run(self, **kwargs):
        # Run tests under mock settings.
        self._settings = default_mock_settings()
        patch_name = mock.patch(f"{launch.__name__}.generate_job_name", return_value="job-name")
        with (
            patch_name,
            mock_gcp_settings(
                [
                    launch.__name__,
                    bastion_vm.__name__,
                    bundler.__name__,
                    runner_gke.__name__,
                    node_pool_provisioner.__name__,
                ],
                settings=self._settings,
            ),
        ):
            return super().run(**kwargs)

    @parameterized.product(
        [
            dict(
                name=None,
                output_dir=None,
                env_id=None,
                bastion=None,
                bundler_type=None,
                bundler_exclude=None,
                namespace=None,
                project=None,
                cluster=None,
            ),
            dict(
                name="test-name",
                output_dir="test-output",
                env_id="test-env-id",
                bastion="test-bastion",
                bundler_type="artifactregistry",
                bundler_exclude=["a", "b"],
                namespace="test-namespace",
                project="test-project",
                cluster="test-cluster",
            ),
        ],
        action=_ACTIONS,
    )
    def test_tpu_flags(
        self,
        action,
        name,
        output_dir,
        env_id,
        bastion,
        bundler_type,
        bundler_exclude,
        namespace,
        project,
        cluster,
    ):
        # Construct flags.
        fv = flags.FlagValues()
        _prelaunch_flags(fv=fv)
        fv.set_default("gcp_api", GCPAPI.GKE.lower())
        fv.mark_as_parsed()

        cfg = BastionManagedGKEJob.default_config().set(
            runner=named_runner_configs("gke_tpu_single")
        )
        define_flags(cfg, fv)

        # Parse argv.
        argv = ["cli", "--bundler_spec=image=test"]
        if name is not None:
            argv.append(f"--name={name}")
        if output_dir is not None:
            argv.append(f"--output_dir={output_dir}")
        if project is not None:
            argv.append(f"--project={project}")
        if env_id is not None:
            argv.append(f"--env_id={env_id}")
        if bastion is not None:
            argv.append(f"--bastion={bastion}")
        if bundler_type is not None:
            argv.append(f"--bundler_type={bundler_type}")
        if namespace is not None:
            argv.append(f"--namespace={namespace}")
        if cluster is not None:
            argv.append(f"--cluster={cluster}")
        if bundler_exclude:
            argv.extend([f"--bundler_exclude={exclude}" for exclude in bundler_exclude])
        argv.extend(["--instance_type=tpu-v4-8", "--num_replicas=2"])
        fv(argv)

        # Check some basic flags.
        self.assertEqual(fv.bastion, bastion)
        self.assertIn("instance_type", fv)
        self.assertIn("bundler_type", fv)
        self.assertEqual(fv.instance_type, "tpu-v4-8")
        self.assertEqual(fv.output_dir, output_dir)
        self.assertEqual(fv.cluster, cluster)
        self.assertEqual(fv.bundler_exclude, bundler_exclude or BUNDLE_EXCLUDE)

        from_flags_kwargs = dict(command="test command", action=action)
        cfg: BastionManagedGKEJob.Config = from_flags(cfg, fv, **from_flags_kwargs)

        self.assertEqual(cfg.name, name or "job-name")
        self.assertEqual(cfg.project, project or self._settings["project"])
        self.assertEqual(cfg.env_id, env_id or self._settings["env_id"])
        self.assertEqual(cfg.namespace, namespace or "default")

        if action in _RUNNER_ACTIONS:
            self.assertIsNotNone(cfg.runner)
            self.assertIsNotNone(cfg.bundler)
            self.assertEqual(bundler_exclude or BUNDLE_EXCLUDE, cfg.bundler.exclude)
            if bundler_type is None:
                # Defaults to cloud build.
                self.assertIs(cfg.bundler.klass, bundler.CloudBuildBundler)

        # Check bastion flag. If None, we should infer from env_id in mock_settings.
        if bastion:
            self.assertEqual(bastion, cfg.bastion_name)
        elif env_id is None:
            self.assertEqual(
                f"{self._settings['env_id']}-gke-bastion",
                cfg.bastion_name,
            )
        else:
            self.assertEqual(f"{env_id}-gke-bastion", cfg.bastion_name)

        # Check output_dir.
        if output_dir is None:
            self.assertEqual(cfg.output_dir, f"gs://settings-ttl-bucket/axlearn/jobs/{cfg.name}")
        else:
            self.assertEqual(cfg.output_dir, output_dir)

        # Test infer tpu resources.
        # pylint: disable-next=too-many-function-args
        self.assertEqual({"v4": 16}, cfg.resources(cfg))

        if action in _RUNNER_ACTIONS:
            # Make sure command is expected.
            for flag in ["name", "bundler_type", "instance_type"]:
                if fv[flag].value is not None:
                    self.assertIn(f"--{flag}={fv[flag].value}", cfg.command)
            self.assertIn("test command", cfg.command)
        else:
            self.assertIsNone(cfg.command)

        with mock.patch(f"{launch.__name__}.load_kube_config") as mock_kube_config:
            # Should be instantiable.
            job: BastionManagedGKEJob = cfg.instantiate()

            # Make sure we load_kube_config with the right values.
            self.assertEqual(
                {"project": cfg.project, "zone": cfg.zone, "cluster": cfg.cluster},
                mock_kube_config.call_args[1],
            )

        # Bundler should be propagated to runner.
        if action in _RUNNER_ACTIONS:
            self.assertIsNotNone(job._runner._bundler)
            self.assertIs(job._bundler, job._runner._bundler)
        else:
            self.assertIsNone(job._bundler)
            self.assertIsNone(job._runner)

    @parameterized.parameters(
        dict(name="test_job", expected=ValueError("invalid characters")),
        dict(name="test-job", expected=None),
    )
    def test_execute(self, name: str, expected: Optional[Exception]):
        class FakeBastionDirectory(BastionDirectory):
            pass

        # Tests the validation logic that happens within TPU builder.
        cfg = BastionManagedGKEJob.default_config().set(
            runner=named_runner_configs("gke_tpu_single")
        )
        fv = _common_flags(cfg)
        fv.instance_type = "tpu-v4-8"
        fv.name = name
        cfg = from_flags(cfg, fv, action="start").set(
            bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir")
        )
        patch_kube_config = mock.patch(f"{launch.__name__}.load_kube_config")
        patch_submit = mock.patch(f"{launch.__name__}.{BaseBastionManagedJob.__name__}.submit")

        with patch_kube_config, patch_submit as mock_submit:
            if isinstance(expected, Exception):
                with self.assertRaisesRegex(type(expected), str(expected)):
                    job: BastionManagedGKEJob = cfg.instantiate()

                mock_submit.assert_not_called()
            else:
                job: BastionManagedGKEJob = cfg.instantiate()
                job_spec = job.submit()
                mock_submit.assert_called_once()
                self.assertIsNotNone(job_spec)

    @parameterized.parameters(None, 0, 1)
    def test_update(self, job_version):
        job_name = "test_job0"

        job_spec = new_jobspec(
            name=job_name,
            command="command",
            metadata=JobMetadata(
                user_id="test_user",
                project_id="test_project",
                creation_time=datetime.now(),
                resources={"v4": 8},
                job_id="test-id0",
                version=job_version,
            ),
        )

        class FakeBastionDirectory(BastionDirectory):
            def get_job(self, job_name: str) -> JobSpec:
                return copy.deepcopy(job_spec)

            def update_job(self, job_name: str, *, job_spec: JobSpec) -> JobSpec:
                return job_spec

        cfg = BastionManagedGKEJob.default_config().set(runner=_DummyRunner.default_config())
        cfg = from_flags(cfg, _common_flags(cfg), action="update").set(
            bundler=_DummyBundler.default_config(),
            bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir"),
        )
        cfg.set(name=job_name)
        patch_kube_config = mock.patch(f"{launch.__name__}.load_kube_config")

        with patch_kube_config:
            job: BastionManagedGKEJob = cfg.instantiate()
            # Update the job.
            updated_job_spec = job.update()
            updated_version = (job_spec.metadata.version or 0) + 1
            self.assertEqual(updated_job_spec.metadata.version, updated_version)

    def test_instance_type(self):
        """Tests --instance_type is retained for backwards compat."""

        fv = flags.FlagValues()
        flags.DEFINE_string("instance_type", "tpu-v4-8", "", flag_values=fv)
        cfg = BastionManagedGKEJob.default_config().set(runner=_DummyRunner.default_config())
        define_flags(cfg, fv)
        fv.mark_as_parsed()
        self.assertEqual(fv.instance_type, "tpu-v4-8")


class MainTest(parameterized.TestCase):
    def run(self, **kwargs):
        # Run tests under mock settings.
        patch_kube = mock.patch.multiple(launch.__name__, load_kube_config=mock.DEFAULT)
        with (
            patch_kube,
            mock_gcp_settings(
                [
                    launch.__name__,
                    bastion_vm.__name__,
                    bundler.__name__,
                    runner_gke.__name__,
                    node_pool_provisioner.__name__,
                ]
            ),
        ):
            return super().run(**kwargs)

    @parameterized.parameters(
        # Test that runner name is inferred from instance type.
        dict(action="start", instance_type="tpu-v4-8", expected="gke_tpu_single"),
        dict(action="run", instance_type="tpu-v4-8", expected="gke_tpu_single"),
        # Test that runner name is inferred from job type.
        dict(action="start", instance_type="tpu-v4-8", job_type="flink", expected="gke_tpu_flink"),
        # Test that runner name is requird if using runner action.
        dict(action="start", expected=app.UsageError("Unable to infer --runner_name")),
        dict(action="run", expected=app.UsageError("Unable to infer --runner_name")),
        # Test that runner name is optional if not using runner action.
        dict(action="list", expected=None),
        dict(action="stop", expected=None),
        dict(action="stop", wait_for_stop=None, expected=None),
        dict(action="stop", nowait_for_stop=None, expected=None),
    )
    def test_main_infer_runner_name(self, action: str, expected: Union[Exception, type], **kwargs):
        argv = ["cli", action]
        argv.extend([f"--{key}={value}" if value else f"--{key}" for key, value in kwargs.items()])
        argv.append("--dry_run")  # Don't run anything.
        argv.extend(["--bundler_type=cloudbuild", "--bundler_spec=image=tpu"])

        with mock.patch("sys.argv", argv):
            fv = flags.FlagValues()  # Don't modify global FLAGS.
            _private_flags(fv)
            fv(argv)  # Parse flags.
            if isinstance(expected, Exception):
                with self.assertRaisesRegex(type(expected), str(expected)):
                    launch.main(argv, fv=fv)
            else:
                launch.main(argv, fv=fv)
                self.assertEqual(expected, fv.runner_name)

    # TODO(markblee): Add more tests for the flag command path.
    @parameterized.product(
        [
            # Tests that we should be able to pass multiple string-separated tokens.
            dict(
                command="test command",
                action="start",
                # This is the command that goes to "run".
                expected="test command",
                as_flag=False,
            ),
            # Same test with "run".
            dict(
                command="test command",
                action="run",
                expected="test command",
                as_flag=False,
            ),
            # Same test with flags instead of argv.
            dict(
                command="test command",
                action="run",
                expected="test command",
                as_flag=True,
            ),
            # Quotes or other shell constructs require the entire command to be quoted.
            dict(
                command="'test \"command with quotes\"'",
                action="start",
                # This is the command that goes to "run".
                expected="'test \"command with quotes\"'",
                as_flag=False,
            ),
            # Same test with "run".
            dict(
                command="'test \"command with quotes\"'",
                action="run",
                # The command itself shouldn't be quoted (so it can provided directly to k8s yaml);
                # but the inner quotes should be retained.
                expected='test "command with quotes"',
                as_flag=False,
            ),
            # Shell commands can be provided by quoting.
            dict(
                command="'if [ true ]; then echo 1; fi'",
                action="submit",
                expected="'if [ true ]; then echo 1; fi'",
                as_flag=False,
            ),
            dict(
                command="'if [ true ]; then echo 1; fi'",
                action="run",
                # The command itself shouldn't be quoted (so it can provided directly to k8s yaml).
                expected="if [ true ]; then echo 1; fi",
                as_flag=False,
            ),
        ],
    )
    def test_main_command(self, command, action, expected, as_flag=False):
        """Tests that command specified as positional turns into flag."""
        argv_flags = [
            "--bastion=fake-bastion",
            "--instance_type=tpu-v4-8",
            "--bundler_type=cloudbuild",
            "--bundler_spec=image=tpu",
            "--dry_run",  # Don't run anything.
        ]
        argv = ["cli", action] + argv_flags
        if as_flag:
            argv.append(f"--command={shlex.quote(command)}")
        else:
            argv.extend(["--", *shlex.split(command)])

        def instantiate(cfg):
            run_argv = shlex.split(cfg.command)
            if as_flag:
                cmd_fv = flags.FlagValues()
                flags.DEFINE_string("command", None, "", flag_values=cmd_fv)
                cmd_fv(run_argv, known_only=True)
                actual_command = cmd_fv.command
            else:
                actual_command = cfg.command.split(" -- ")[-1]

            self.assertEqual(expected, actual_command, msg=f"{actual_command=}")
            # Check that original flags are retained.
            self.assertContainsSubset(argv_flags, run_argv)

        with (
            mock.patch("sys.argv", argv),
            mock.patch.object(
                BaseBastionManagedJob.Config,
                "instantiate",
                side_effect=instantiate,
                autospec=True,
            ) as mock_instantiate,
        ):
            fv = flags.FlagValues()  # Don't modify global FLAGS.
            _private_flags(fv)
            fv(argv)  # Parse flags.
            launch.main(argv, fv=fv)
            mock_instantiate.assert_called_once()

    @parameterized.parameters(
        # Prints all runners if just --help.
        dict(argv=["--help"], expected="Possible runners are"),
        # Prints some additional launch help with --instance_type.
        dict(argv=["--instance_type=tpu-v4-8", "--help"], expected="--command"),
        # Prints some additional launch help with --runner_name.
        dict(argv=["--runner_name=gke_tpu_single", "--help"], expected="--command"),
    )
    def test_help(self, argv: list, expected: str):
        f = io.StringIO()
        argv = ["cli", *argv, "--dry_run"]  # Don't run anything.
        with redirect_stdout(f), mock.patch("sys.argv", argv):
            fv = flags.FlagValues()  # Don't modify global FLAGS.
            _private_flags(fv)
            app.usage(writeto_stdout=True)

        output = f.getvalue().strip()
        self.assertIn(expected, output)

    @parameterized.parameters(True, False)
    def test_submit(self, inner_enabled: bool):
        """Tests that submit and run get the same flags."""

        class Child(FlagConfigurable):
            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                flags.DEFINE_string("inner", None, "", flag_values=fv)

        class DummyRunner(BaseRunnerJob):
            """A dummy runner that changes its config structure based on flags."""

            @config_class
            class Config(FlagConfigurable.Config):
                inner_enabled: Required[bool] = REQUIRED
                inner: Optional[Child.Config] = Child.default_config()

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                flags.DEFINE_bool("inner_enabled", inner_enabled, "", flag_values=fv)

            @classmethod
            def from_flags(cls, fv, **kwargs):
                cfg = super().from_flags(fv, **kwargs)
                # Set inner to None if disabled.
                if not cfg.inner_enabled:
                    cfg.inner = None
                return cfg

        def instantiate(cfg: BaseBastionManagedJob.Config):
            # Even though `DummyRunner` changes its config structure based on inner_enabled, we
            # validate that the command sent to the bastion still contains the flags for inner.
            # This allows the bastion to change the config structure in the same way.
            if inner_enabled:
                self.assertIn("--inner_enabled", cfg.command)
                self.assertIn("--inner=test-inner", cfg.command)
            else:
                self.assertIn("--noinner_enabled", cfg.command)
                self.assertIn("--inner=test-inner", cfg.command)

        argv_flags = [
            "--bastion=fake-bastion",
            "--instance_type=tpu-v4-8",
            "--bundler_type=cloudbuild",
            "--bundler_spec=image=tpu",
            "--inner=test-inner",
            "--dry_run",  # Don't run anything.
        ]
        argv = ["cli", "start"] + argv_flags

        cfg = DummyRunner.default_config()
        with (
            mock.patch(f"{launch.__name__}._get_runner_or_exit", return_value=cfg),
            mock.patch("sys.argv", argv),
            mock.patch.object(
                BaseBastionManagedJob.Config,
                "instantiate",
                side_effect=instantiate,
                autospec=True,
            ) as mock_instantiate,
        ):
            fv = flags.FlagValues()  # Don't modify global FLAGS.
            _private_flags(fv)
            fv(argv)  # Parse flags.
            launch.main(argv, fv=fv)
            mock_instantiate.assert_called_once()

    @parameterized.parameters(dict(argv=[]), dict(argv=["--env_id=test-env"]))
    def test_list(self, argv):
        argv = ["cli", "list", *argv]  # Don't run anything.
        mock_job = mock.Mock()

        def instantiate(cfg: BaseBastionManagedJob.Config):
            self.assertIsNotNone(cfg.env_id)
            return mock_job

        patch_job = mock.patch.object(
            BaseBastionManagedJob.Config, "instantiate", side_effect=instantiate, autospec=True
        )
        with mock.patch("sys.argv", argv), patch_job:
            fv = flags.FlagValues()  # Don't modify global FLAGS.
            _private_flags(fv)
            fv(argv)  # Parse flags.
            launch.main(argv, fv=fv)
            self.assertTrue(mock_job.list.called)
