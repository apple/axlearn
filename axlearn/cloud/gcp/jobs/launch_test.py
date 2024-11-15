# Copyright © 2023 Apple Inc.

"""Tests launchers."""
# pylint: disable=protected-access

import contextlib
import copy
from datetime import datetime
from typing import Optional
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import JobState as BastionJobState
from axlearn.cloud.common.bastion import JobStatus, deserialize_jobspec, new_jobspec
from axlearn.cloud.common.bundler import BUNDLE_EXCLUDE
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.common.types import JobSpec
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp import job as gcp_job
from axlearn.cloud.gcp.jobs import bastion_vm, gke_runner, launch, tpu_runner
from axlearn.cloud.gcp.jobs.launch import (
    BaseBastionManagedJob,
    BastionDirectory,
    BastionManagedGKEJob,
    BastionManagedTPUJob,
    Launcher,
    _get_launcher_or_exit,
    _prelaunch_flags,
)
from axlearn.cloud.gcp.jobs.launch_utils import (
    Table,
    jobs_table,
    project_usage_table,
    user_usage_table,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.utils import GCPAPI
from axlearn.common.config import config_for_function, maybe_instantiate
from axlearn.common.test_utils import TestWithTemporaryCWD


class TestUtils(parameterized.TestCase):
    """Tests util functions."""

    def test_get_launcher_or_exit(self):
        def match_qrm_tpu(*, action, instance_type, gcp_api):
            del action
            return instance_type == "tpu" and gcp_api == GCPAPI.QRM

        def match_gke(*, action, instance_type, gcp_api):
            del action, instance_type
            return gcp_api == GCPAPI.GKE

        class DummyTPULauncher(Job):
            pass

        class DummyGKELauncher(Job):
            pass

        mock_launchers = [
            Launcher(job_cls=DummyTPULauncher, matcher=match_qrm_tpu, description="test"),
            Launcher(job_cls=DummyGKELauncher, matcher=match_gke, description="test"),
        ]
        with mock.patch(f"{launch.__name__}._LAUNCHERS", mock_launchers):
            launcher = _get_launcher_or_exit(
                action="start", instance_type="tpu", gcp_api=GCPAPI.QRM
            )
            self.assertEqual(launcher.job_cls, DummyTPULauncher)
            self.assertEqual(launcher.matcher, match_qrm_tpu)

            launcher = _get_launcher_or_exit(
                action="start", instance_type="other", gcp_api=GCPAPI.GKE
            )
            self.assertEqual(launcher.job_cls, DummyGKELauncher)
            self.assertEqual(launcher.matcher, match_gke)

            with self.assertRaises(app.UsageError):
                _get_launcher_or_exit(action="start", instance_type="other", gcp_api=GCPAPI.QRM)


class _DummyRunner(Job):
    def _execute(self):
        pass


def _common_bastion_managed_job_kwargs() -> dict:
    return dict(
        instance_type="tpu-v4-8",
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
        runner=_DummyRunner.default_config().set(
            max_tries=1,
            retry_interval=60,
            command="",
        ),
    )


class TestBaseBastionManagedJob(parameterized.TestCase):
    """Tests BaseBastionManagedJob."""

    def _mock_config(self, **kwargs):
        cfg = BaseBastionManagedJob.default_config().set(**_common_bastion_managed_job_kwargs())
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
            job_spec = job._execute()
            self.assertIsNotNone(job_spec)

            # Test with bundler.
            mock_bundler = mock.MagicMock()
            cfg = self._mock_config()
            cfg.runner.bundler = config_for_function(lambda: mock_bundler)
            job = cfg.instantiate()
            job_spec = job._execute()
            self.assertTrue(mock_bundler.bundle.called)
            self.assertIsNotNone(job_spec)

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
        job = self._mock_config().instantiate()
        with mock.patch.object(job._bastion_dir, "list_jobs", return_value=mock_jobs):
            jobs = job._list()
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


class TestBastionManagedTPUJob(TestWithTemporaryCWD):
    """Tests BastionManagedTPUJob."""

    @parameterized.product(
        [
            dict(
                name=None,
                output_dir=None,
                zone=None,
                bastion=None,
                bundler_type=None,
                bundler_exclude=None,
            ),
            dict(
                name="test-name",
                output_dir="test-output",
                zone="test-zone",
                bastion="test-bastion",
                bundler_type="artifactregistry",
                bundler_exclude=["a", "b"],
            ),
        ],
        action=["start", "list"],
    )
    def test_flags(self, name, output_dir, zone, action, bastion, bundler_type, bundler_exclude):
        # Construct flags.
        fv = flags.FlagValues()
        _prelaunch_flags(fv=fv)
        fv.set_default("gcp_api", GCPAPI.QRM.lower())
        fv.mark_as_parsed()

        patch_fns = mock.patch.multiple(
            launch.__name__,
            generate_job_name=mock.Mock(return_value="job-name"),
        )
        mock_settings = {
            "ttl_bucket": "ttl_bucket",
            "permanent_bucket": "permanent_bucket",
            "project": "default-project",
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
            if bundler_exclude:
                argv.extend([f"--bundler_exclude={exclude}" for exclude in bundler_exclude])
            argv.append("--instance_type=tpu-v4-8")
            fv(argv)

            # Check some basic flags.
            self.assertEqual(fv.bastion, bastion)
            self.assertEqual(fv.name, name or "job-name")
            self.assertEqual(fv.zone, zone or mock_settings["zone"])
            self.assertIn("instance_type", fv)
            self.assertIn("bundler_type", fv)
            self.assertIsNotNone(fv["name"].default)
            self.assertEqual(fv.instance_type, "tpu-v4-8")
            self.assertEqual(fv.output_dir, output_dir)

            cfg = BastionManagedTPUJob.from_flags(fv, command="test command", action=action)
            self.assertIsNone(cfg.bundler)
            if action == "start":
                self.assertIsNotNone(cfg.runner)
                self.assertIsNotNone(cfg.runner.bundler)
                self.assertIn("tpu", cfg.runner.bundler.extras)
                self.assertEqual(bundler_exclude or BUNDLE_EXCLUDE, cfg.runner.bundler.exclude)
            else:
                self.assertIsNone(cfg.runner)

            # Check bastion flag. If None, we should infer from zone in mock_settings.
            if bastion:
                self.assertEqual(bastion, cfg.bastion_name)
            elif zone is None:
                self.assertEqual(
                    f"{mock_settings['zone']}-shared-bastion",
                    cfg.bastion_name,
                )
            else:
                self.assertEqual(f"{zone}-shared-bastion", cfg.bastion_name)

            # Check output_dir.
            if output_dir is None:
                self.assertEqual(cfg.output_dir, f"gs://ttl_bucket/axlearn/jobs/{fv.name}")
            else:
                self.assertEqual(cfg.output_dir, output_dir)

        if action == "start":
            # Make sure command is expected.
            for flag in ["name", "bundler_type", "instance_type"]:
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


class TestBastionManagedGKEJob(TestWithTemporaryCWD):
    """Tests BastionManagedGKEJob."""

    @parameterized.product(
        [
            dict(
                name=None,
                output_dir=None,
                zone=None,
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
                zone="test-zone",
                bastion="test-bastion",
                bundler_type="artifactregistry",
                bundler_exclude=["a", "b"],
                namespace="test-namespace",
                project="test-project",
                cluster="test-cluster",
            ),
        ],
        action=["start", "list", "update"],
    )
    def test_tpu_flags(
        self,
        action,
        name,
        output_dir,
        zone,
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

        patch_fns = mock.patch.multiple(
            launch.__name__,
            generate_job_name=mock.Mock(return_value="job-name"),
        )
        mock_settings = {
            "ttl_bucket": "ttl_bucket",
            "permanent_bucket": "permanent_bucket",
            "zone": zone or "default-zone",
            "project": project or "default-project",
            "docker_repo": "test-repo",
            "default_dockerfile": "test-dockerfile",
            "gke_cluster": "default-cluster",
        }
        patch_settings = mock_gcp_settings(
            [launch.__name__, bastion_vm.__name__, gke_runner.__name__, bundler.__name__],
            settings=mock_settings,
        )
        patch_project_zone = mock.patch.multiple(
            gcp_job.__name__,
            default_project=mock.Mock(return_value=mock_settings["project"]),
            default_zone=mock.Mock(return_value=mock_settings["zone"]),
        )
        tpu_gke_job = BastionManagedGKEJob.with_runner(gke_runner.TPUGKERunnerJob)

        with patch_fns, patch_settings, patch_project_zone:
            tpu_gke_job.define_flags(fv)

            # Parse argv.
            argv = ["cli", "--bundler_spec=image=test"]
            if name is not None:
                argv.append(f"--name={name}")
            if output_dir is not None:
                argv.append(f"--output_dir={output_dir}")
            if project is not None:
                argv.append(f"--project={project}")
            if zone is not None:
                argv.append(f"--zone={zone}")
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
            self.assertEqual(fv.name, name or "job-name")
            self.assertEqual(fv.project, project or mock_settings["project"])
            self.assertEqual(fv.zone, zone or mock_settings["zone"])
            self.assertIn("instance_type", fv)
            self.assertIn("bundler_type", fv)
            self.assertIsNotNone(fv["name"].default)
            self.assertEqual(fv.instance_type, "tpu-v4-8")
            self.assertEqual(fv.output_dir, output_dir)
            self.assertEqual(fv.namespace, namespace or "default")
            self.assertEqual(fv.cluster, cluster)
            self.assertEqual(fv.bundler_exclude, bundler_exclude or BUNDLE_EXCLUDE)

            from_flags_kwargs = dict(command="test command", action=action)
            cfg = tpu_gke_job.from_flags(fv, **from_flags_kwargs)

            self.assertIsNone(cfg.bundler)
            if action in ("start", "update"):
                self.assertIsNotNone(cfg.runner)
                self.assertIsNotNone(cfg.runner.bundler)
                self.assertIn("tpu", cfg.runner.bundler.extras)
                self.assertEqual(bundler_exclude or BUNDLE_EXCLUDE, cfg.runner.bundler.exclude)
                if bundler_type is None:
                    # Defaults to cloud build.
                    self.assertIs(cfg.runner.bundler.klass, bundler.CloudBuildBundler)
            else:
                self.assertIsNone(cfg.runner)

            # Check bastion flag. If None, we should infer from zone in mock_settings.
            if bastion:
                self.assertEqual(bastion, cfg.bastion_name)
            elif zone is None:
                self.assertEqual(
                    f"{mock_settings['zone']}-gke-bastion",
                    cfg.bastion_name,
                )
            else:
                self.assertEqual(f"{zone}-gke-bastion", cfg.bastion_name)

            # Check output_dir.
            if output_dir is None:
                self.assertEqual(cfg.output_dir, f"gs://ttl_bucket/axlearn/jobs/{fv.name}")
            else:
                self.assertEqual(cfg.output_dir, output_dir)

            # Test infer tpu resources.
            self.assertEqual({"v4": 16}, maybe_instantiate(cfg.resources))

        if action in ("start", "update"):
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
        if action in ("start", "update"):
            self.assertIsNotNone(job.runner.bundler)

    @parameterized.parameters(
        dict(name="test_job", expected=ValueError("invalid characters")),
        dict(name="test-job", expected=None),
    )
    def test_execute(self, name: str, expected: Optional[Exception]):
        class FakeBastionDirectory(BastionDirectory):
            pass

        tpu_gke_job = BastionManagedGKEJob.with_runner(_DummyRunner)
        cfg = tpu_gke_job.default_config().set(
            **_common_bastion_managed_job_kwargs(),
            namespace="default",
            project="test-project",
            cluster="test-cluster",
            bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir"),
        )
        cfg.set(name=name)
        patch_kube_config = mock.patch(f"{launch.__name__}.load_kube_config")
        patch_execute = mock.patch(f"{launch.__name__}.{BaseBastionManagedJob.__name__}._execute")

        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()

        with ctx, patch_kube_config, patch_execute as mock_execute:
            job: BastionManagedGKEJob = cfg.instantiate()
            job_spec = job._execute()

            if isinstance(expected, Exception):
                mock_execute.assert_not_called()
            else:
                mock_execute.assert_called_once()
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

        tpu_gke_job = BastionManagedGKEJob.with_runner(_DummyRunner)
        cfg = tpu_gke_job.default_config().set(
            **_common_bastion_managed_job_kwargs(),
            namespace="default",
            project="test-project",
            cluster="test-cluster",
            bastion_dir=FakeBastionDirectory.default_config().set(root_dir="temp_dir"),
        )
        cfg.set(name=job_name)
        patch_kube_config = mock.patch(f"{launch.__name__}.load_kube_config")

        with patch_kube_config:
            job: BastionManagedGKEJob = cfg.instantiate()

            # Update the job.
            updated_job_spec = job._update()

            updated_version = (job_spec.metadata.version or 0) + 1

            self.assertEqual(updated_job_spec.metadata.version, updated_version)
