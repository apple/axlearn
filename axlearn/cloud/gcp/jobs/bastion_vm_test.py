# Copyright © 2023 Apple Inc.

"""Tests bastion VM."""
# pylint: disable=protected-access

import contextlib
from unittest import mock

from absl import flags

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
        with mock_execute, mock_creds, mock_vm(bastion_vm.__name__) as mocks:
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
    mock_from_flags = mock.patch.object(job, "from_flags", return_value=mock_cfg)
    mock_get_bundler = mock.patch(
        f"{bastion_vm.__name__}.get_bundler_config", return_value=mock_bundler
    )
    mock_settings = mock_gcp_settings(bastion_vm.__name__, settings_kwargs)
    with mock_from_flags, mock_get_bundler, mock_settings, mock_vm(bastion_vm.__name__):
        yield mock_cfg, mock_bundler


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
        with mock_job as (_, mock_bundler):
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
        with mock_job as (mock_cfg, _):
            bastion_vm.main(["cli", "start"], flag_values=fv)
            bastion_cfg = mock_cfg.set.call_args.kwargs["bastion"]
            _validate_required_fields(bastion_cfg)
            self.assertIn("test-bucket", bastion_cfg.output_dir)
            self.assertIn("test-bastion", bastion_cfg.output_dir)
            self.assertIn("test-private", bastion_cfg.scheduler.quota.quota_file)
