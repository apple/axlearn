# Copyright © 2023 Apple Inc.

"""Tests bastion VM."""
# pylint: disable=protected-access

import contextlib
import tempfile
from datetime import datetime
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import (
    Job,
    JobState,
    _load_runtime_options,
    new_jobspec,
    set_runtime_options,
)
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.gcp.jobs import bastion_vm
from axlearn.cloud.gcp.jobs.bastion_vm import CreateBastionJob
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.config import _validate_required_fields, config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def mock_vm(module_name: str):
    """Mocks out VM create and delete."""

    def mock_create_vm(*args, **kwargs):
        del args, kwargs

    def mock_delete_vm(*args, **kwargs):
        del args, kwargs

    mocks = {
        "create_vm": mock_create_vm,
        "delete_vm": mock_delete_vm,
    }
    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once, and return the mocks.
        mocks = {
            name: stack.enter_context(mock.patch(f"{module_name}.{name}", side_effect=method))
            for name, method in mocks.items()
        }
        yield mocks


def _mock_create_config():
    mock_bundler = mock.MagicMock()
    mock_bundler.TYPE = "test-bundler"
    mock_bundler.install_command.return_value = "test_install"
    return CreateBastionJob.default_config().set(
        name="test",
        project="test-project",
        zone="test-zone",
        vm_type="test-vm",
        disk_size=123,
        max_tries=1,
        retry_interval=1,
        bundler=config_for_function(lambda: mock_bundler),
    )


class CreateBastionJobTest(TestWithTemporaryCWD):
    """Tests CreateBastionJob."""

    def test_create(self):
        cfg = _mock_create_config()
        job = cfg.instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd", return_value=None)
        mock_creds = mock.patch.object(job, "_get_job_credentials", return_value=None)
        mock_output_dir = mock.patch(f"{bastion_vm.__name__}.output_dir", return_value="temp_dir")
        with mock_output_dir, mock_execute, mock_creds, mock_vm(bastion_vm.__name__) as mocks:
            job._execute()
            self.assertTrue(mocks["create_vm"].called)
            self.assertIn(cfg.name, mocks["create_vm"].call_args.args)
            self.assertEqual(cfg.vm_type, mocks["create_vm"].call_args.kwargs["vm_type"])
            self.assertEqual(cfg.disk_size, mocks["create_vm"].call_args.kwargs["disk_size"])
            self.assertEqual("test-bundler", mocks["create_vm"].call_args.kwargs["bundler_type"])

    def test_delete(self):
        cfg = _mock_create_config()
        job = cfg.instantiate()

        mock_creds = mock.patch.object(job, "_get_job_credentials", return_value=None)
        with mock_creds, mock_vm(bastion_vm.__name__) as mocks:
            job._delete()
            self.assertTrue(mocks["delete_vm"].called)
            self.assertIn(cfg.name, mocks["delete_vm"].call_args.args)


@contextlib.contextmanager
def _mock_job(job, *, bundler_kwargs, settings_kwargs):
    mock_job = mock.MagicMock()
    mock_bundler = mock.MagicMock(**bundler_kwargs)
    mock_cfg = mock.MagicMock(**{"instantiate.return_value": mock_job})
    if hasattr(job, "from_flags"):
        mock_construct = mock.patch.object(job, "from_flags", return_value=mock_cfg)
    elif hasattr(job, "default_config"):
        mock_construct = mock.patch.object(job, "default_config", return_value=mock_cfg)
    mock_get_bundler = mock.patch(
        f"{bastion_vm.__name__}.get_bundler_config", return_value=mock_bundler
    )
    mock_settings = mock_gcp_settings(bastion_vm.__name__, settings_kwargs)
    with mock_construct, mock_get_bundler, mock_settings, mock_vm(bastion_vm.__name__):
        yield mock_cfg, mock_bundler, mock_job


class MainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_private_flags(self):
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        # Basic sanity check.
        self.assertIsNotNone(fv["vm_type"].default)

    def test_create(self):
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        fv.set_default("name", "test-bastion")
        fv.mark_as_parsed()

        # Bundler should use params from spec or fallback to defaults.
        bundler_kwargs = {"image": "test-image", "repo": "test-repo", "dockerfile": None}
        settings_kwargs = {
            "docker_repo": "default-repo",
            "default_dockerfile": "default-dockerfile",
        }
        mock_job = _mock_job(
            bastion_vm.CreateBastionJob,
            bundler_kwargs=bundler_kwargs,
            settings_kwargs=settings_kwargs,
        )
        # Check that the bundler is constructed properly.
        with mock_job as (_, mock_bundler, _):
            bastion_vm.main(["cli", "create"], flag_values=fv)
            self.assertEqual("test-image", mock_bundler.set.call_args.kwargs["image"])
            self.assertEqual("test-repo", mock_bundler.set.call_args.kwargs["repo"])
            self.assertEqual("default-dockerfile", mock_bundler.set.call_args.kwargs["dockerfile"])

    def test_start(self):
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        fv.set_default("name", "test-bastion")
        fv.mark_as_parsed()

        settings_kwargs = {
            "permanent_bucket": "test-bucket",
            "private_bucket": "test-private",
        }
        mock_job = _mock_job(
            bastion_vm.StartBastionJob, bundler_kwargs={}, settings_kwargs=settings_kwargs
        )
        with mock_job as (mock_cfg, _, _):
            bastion_vm.main(["cli", "start"], flag_values=fv)
            bastion_cfg = mock_cfg.set.call_args.kwargs["bastion"]
            _validate_required_fields(bastion_cfg)
            self.assertIn("test-bucket", bastion_cfg.output_dir)
            self.assertIn("test-bastion", bastion_cfg.output_dir)
            self.assertIn("test-private", bastion_cfg.scheduler.quota.quota_file)

    @parameterized.parameters(
        dict(execute=None, expect_raises=False),
        dict(execute=ValueError("No such container: test-bastion"), expect_raises=False),
        dict(execute=ValueError("some other error"), expect_raises=True),
    )
    def test_stop(self, execute, expect_raises):
        # Stopping a stopped bastion should not raise.
        mock_job = _mock_job(bastion_vm.CPUJob, bundler_kwargs={}, settings_kwargs={})
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        fv.set_default("name", "test-bastion")
        fv.mark_as_parsed()

        with mock_job as (mock_cfg, _, stop_job):
            mock_set = mock.patch.object(mock_cfg, "set", return_value=mock_cfg)
            mock_execute = mock.patch.object(stop_job, "execute", side_effect=execute)
            if expect_raises:
                ctx = self.assertRaisesRegex(type(execute), str(execute))
            else:
                ctx = contextlib.nullcontext()
            with mock_set, mock_execute, ctx:
                bastion_vm.main(["cli", "stop"], flag_values=fv)

    @parameterized.parameters(
        dict(
            input_value="y",
            # Returns terminated jobs for each sweep.
            mock_sweeps=[
                [],
                ["test_job0", "test_job2"],
                ["test_job1"],
            ],
        ),
        dict(input_value="n", mock_sweeps=[]),
    )
    def test_delete_child_jobs(self, input_value, mock_sweeps):
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        fv.set_default("name", "test-bastion")
        fv.set_default("delete_child_jobs", True)
        fv.mark_as_parsed()

        mock_jobs = {
            f"job{i}": Job(
                spec=new_jobspec(
                    name=f"test_job{i}",
                    command="command",
                    metadata=JobMetadata(
                        user_id="test_user",
                        project_id="test_project",
                        creation_time=datetime.now(),
                        resources={"v4": 8},
                    ),
                ),
                state=state,
                command_proc=None,
                cleanup_proc=None,
            )
            for i, state in enumerate([JobState.PENDING, JobState.ACTIVE, JobState.CANCELLING])
        }
        mock_utils = mock.patch.multiple(
            bastion_vm.__name__,
            download_job_batch=mock.Mock(return_value=(mock_jobs, set())),
            _stop_bastion=mock.Mock(return_value=None),
            input=mock.Mock(return_value=input_value),
        )
        settings_kwargs = {"permanent_bucket": "test-bucket"}
        mock_cleaner = _mock_job(
            bastion_vm.TPUCleaner, bundler_kwargs={}, settings_kwargs=settings_kwargs
        )
        mock_sleep = mock.patch(f"{bastion_vm.__name__}.time.sleep")

        with mock_sleep, mock_utils, mock_cleaner as (_, _, mock_cleaner_job):
            with mock.patch.object(mock_cleaner_job, "sweep", side_effect=mock_sweeps):
                bastion_vm.main(["cli", "stop"], flag_values=fv)

    @parameterized.parameters(
        dict(
            original=None,
            options=None,
            expected=app.UsageError("json"),
        ),
        dict(
            original=None,
            options=r'{"test": {"int": 123, "bool": true, "string": "test"}}',
            expected={"test": {"int": 123, "bool": True, "string": "test"}},
        ),
        dict(
            original={"test": {"int": 123, "bool": True, "string": "test"}},
            options=r'{"test": {"bool": false}, "test1": 1}',
            expected={"test": {"int": 123, "bool": False, "string": "test"}, "test1": 1},
        ),
    )
    def test_set(self, options, expected, original=None):
        fv = flags.FlagValues()
        bastion_vm._private_flags(flag_values=fv)
        fv.set_default("name", "test-bastion")
        fv.set_default("runtime_options", options)
        fv.mark_as_parsed()

        def do_set():
            with tempfile.TemporaryDirectory() as temp_dir:
                if original is not None:
                    set_runtime_options(temp_dir, **original)
                with mock.patch(f"{bastion_vm.__name__}.output_dir", return_value=temp_dir):
                    bastion_vm.main(["cli", "set"], flag_values=fv)
                return _load_runtime_options(temp_dir)

        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                do_set()
        else:
            self.assertEqual(expected, do_set())
