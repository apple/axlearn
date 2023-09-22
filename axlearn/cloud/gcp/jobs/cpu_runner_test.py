# Copyright © 2023 Apple Inc.

"""Tests CPU runner job."""
# pylint: disable=protected-access
import contextlib
import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.gcp import job as gcp_job
from axlearn.cloud.gcp.bundler import GCSTarBundler
from axlearn.cloud.gcp.jobs import cpu_runner
from axlearn.cloud.gcp.jobs.cpu_runner import (
    _COMMAND_SESSION_NAME,
    CPURunnerJob,
    launch_flags,
    main,
)
from axlearn.cloud.gcp.vm import VmInfo
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def mock_vm(module_name: str, running_from_vm: bool = True):
    """Mocks out VM get, create, and delete."""
    running_vms = {}

    def mock_create_vm(name: str, *args, **kwargs):
        del args, kwargs
        if name not in running_vms:
            vm = mock.MagicMock()
            vm.name = name
            vm.status = "CREATED"
            running_vms[name] = vm

    def mock_get_vm_node(name: str, *args, **kwargs):
        del args, kwargs
        return running_vms.get(name, None)

    def mock_get_vm_node_status(node):
        del node
        return "BOOTED"

    def mock_delete_vm(name: str, *args, **kwargs):
        del args, kwargs
        running_vms.pop(name, None)

    def mock_running_from_vm():
        return running_from_vm

    def mock_compute_resource(*args, **kwargs):
        del args, kwargs

    def mock_list_vm_info(creds):
        del creds
        return [VmInfo(name=vm.name, metadata={}) for vm in running_vms.values()]

    mocks = {
        "create_vm": mock_create_vm,
        "get_vm_node": mock_get_vm_node,
        "get_vm_node_status": mock_get_vm_node_status,
        "_compute_resource": mock_compute_resource,
        "delete_vm": mock_delete_vm,
        "running_from_vm": mock_running_from_vm,
        "list_vm_info": mock_list_vm_info,
    }
    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once, and return the mocks.
        mocks = {
            name: stack.enter_context(mock.patch(f"{module_name}.{name}", side_effect=method))
            for name, method in mocks.items()
        }
        yield mocks


@contextlib.contextmanager
def mock_credentials():
    mocks = [
        mock.patch(f"{gcp_job.__name__}.get_credentials"),
        mock.patch(f"{cpu_runner.__name__}.get_credentials"),
    ]
    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)
        yield


def _mock_config():
    bundler = mock.MagicMock()
    bundler.install_command.return_value = "test_install"
    return CPURunnerJob.default_config().set(
        name="test_name",
        output_dir="test_output",
        vm_type="n2-standard-2",
        disk_size=32,
        project="test_project",
        zone="test_zone",
        max_tries=1,
        retry_interval=60,
        bundler=config_for_function(lambda: bundler),
    )


class CPURunnerJobTest(TestWithTemporaryCWD):
    """Tests CPURunnerJob."""

    def test_start(self):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd", return_value=None)

        with mock_execute, mock_credentials(), mock_vm(cpu_runner.__name__) as mocks:
            job._start()
            mocks["create_vm"].assert_called()
            # Bundling should happen outside of start.
            job._bundler.bundle.assert_not_called()
            # Install should happen at run_command, not start.
            job._bundler.install_command.assert_not_called()

    @parameterized.parameters(True, False)
    def test_delete(self, retain_vm):
        cfg = _mock_config()
        job = cfg.set(command="", retain_vm=retain_vm).instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd", return_value=None)

        with mock_execute, mock_credentials(), mock_vm(cpu_runner.__name__) as mocks:
            job._delete()
            # Without any VM, no delete should be issued.
            self.assertEqual(False, mocks["delete_vm"].called)
            # Create a dummy VM, it should be deleted if not retaining VM.
            mocks["create_vm"](cfg.name)
            job._delete()
            self.assertEqual(not retain_vm, mocks["delete_vm"].called)

    @parameterized.parameters(True, False)
    def test_run_command(self, success):
        cfg = _mock_config()
        # Choose a stable temp path.
        temp_dir = f"/tmp/axlearn/cpu_runner_test/test_run_command/{success}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Convert to bash 'true' or 'false' commands.
        job = cfg.set(
            command="true" if success else "false", local_output_dir=temp_dir
        ).instantiate()
        self.assertIn(temp_dir, str(job._output_dir))

        def execute_command(cmd, detached_session=None, **kwargs):
            del kwargs
            # Only execute for the main session.
            if detached_session == _COMMAND_SESSION_NAME:
                self.assertIn(
                    # Use a stable hash for consistent runs.
                    hashlib.md5(cmd.encode("utf-8")).hexdigest(),
                    {"c2cba3cc375f084f509cb5c0228bbfa5", "49c8241d394b4bf69950b51301f7aa48"},
                    msg="_run_command has changed. Out of precaution, will not run the test.",
                )
                subprocess.run(f"bash -c {shlex.quote(cmd)}", shell=True, check=False)
                if success:
                    # Check for success status.
                    self.assertEqual("SUCCESS", job._status_file.read_text().strip())
                else:
                    # Check for failure status.
                    self.assertEqual("FAILED", job._status_file.read_text().strip())

        mock_execute = mock.patch.object(job, "_execute_remote_cmd", side_effect=execute_command)
        with mock_execute, mock_credentials(), mock_vm(cpu_runner.__name__):
            job._run_command()
            # Install should happen at run_command.
            job._bundler.install_command.assert_called()

    @parameterized.parameters("SUCCESS", None)
    def test_get_status(self, status):
        cfg = _mock_config()
        # Choose a stable temp path.
        temp_dir = f"/tmp/axlearn/cpu_runner_test/test_get_status/{status}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        job = cfg.set(command="", local_output_dir=temp_dir).instantiate()
        self.assertIn(temp_dir, str(job._status_file))

        # Create dummy status file.
        if status is not None:
            status_file = pathlib.Path(job._status_file)
            status_file.parent.mkdir(parents=True)
            status_file.write_text(status, encoding="utf-8")

        def execute_command(cmd, **kwargs):
            del kwargs
            # Don't bother running the mocked install_command.
            if cmd == job._bundler.install_command():
                return None
            self.assertIn(
                # Use a stable hash for consistent runs.
                hashlib.md5(cmd.encode("utf-8")).hexdigest(),
                {"7d49a7b4387e6a69930f1b9782e3d2f3", "f7fc7c4224e42e7928138e81e50f11dc"},
                msg="_get_status has changed. Out of precaution, will not run the test.",
            )
            return subprocess.run(
                f"bash -c {shlex.quote(cmd)}",
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )

        mock_execute = mock.patch.object(job, "_execute_remote_cmd", side_effect=execute_command)
        with mock_execute, mock_credentials(), mock_vm(cpu_runner.__name__):
            self.assertEqual(CPURunnerJob.Status.NOT_STARTED, job._get_status())
            job._start()

            if status is not None:
                self.assertEqual(CPURunnerJob.Status[status], job._get_status())
            else:
                self.assertEqual(CPURunnerJob.Status.NOT_RUNNING, job._get_status())

    def test_execute(self):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        @contextlib.contextmanager
        def mock_status(status):
            mocks = {
                "_get_status": mock.patch.object(
                    job, "_get_status", side_effect=[status, StopIteration()]
                ),
                "_delete": mock.patch.object(job, "_delete", return_value=None),
                "_start": mock.patch.object(job, "_start", return_value=None),
                "_run_command": mock.patch.object(job, "_run_command", return_value=None),
            }
            with contextlib.ExitStack() as stack:
                yield {name: stack.enter_context(patch) for name, patch in mocks.items()}

        try:
            with mock_status(CPURunnerJob.Status.SUCCESS) as mocks:
                job._execute()
                mocks["_delete"].assert_called()

            with self.assertRaisesRegex(ValueError, "failed"), mock_status(
                CPURunnerJob.Status.FAILED
            ) as mocks:
                job._execute()
                mocks["_delete"].assert_called()

            with mock_status(CPURunnerJob.Status.NOT_STARTED) as mocks:
                job._execute()
                mocks["_start"].assert_called()

            with mock_status(CPURunnerJob.Status.NOT_RUNNING) as mocks:
                job._execute()
                mocks["_run_command"].assert_called()
        except StopIteration:
            pass  # Expected.


@contextlib.contextmanager
def _mock_job(running_from_vm: bool):
    mock_job = mock.MagicMock()
    mock_cfg = mock.MagicMock(**{"instantiate.return_value": mock_job})
    patch = mock.patch.object(cpu_runner.CPURunnerJob, "from_flags", return_value=mock_cfg)

    with mock_vm(cpu_runner.__name__, running_from_vm=running_from_vm), patch:
        yield mock_job


class CPURunnerMainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_launch_flags(self):
        fv = flags.FlagValues()
        launch_flags(flag_values=fv)
        # Basic sanity check.
        self.assertEqual(fv["vm_type"].default, "n2-standard-16")

    @parameterized.parameters(True, False)
    def test_list(self, running_from_vm):
        # Test that list can be invoked without additional flags.
        with _mock_job(running_from_vm), mock_credentials():
            main(["cli", "list"])

    @parameterized.parameters(True, False)
    def test_stop(self, running_from_vm):
        # Test that stop can be invoked with just --name.
        with _mock_job(running_from_vm):
            main(["cli", "stop", "--name=test"])

    @parameterized.parameters(True, False)
    def test_start(self, running_from_vm):
        fv = flags.FlagValues()
        launch_flags(flag_values=fv)
        fv.mark_as_parsed()
        self.assertEqual(fv.bundler_type, GCSTarBundler.TYPE)

        with _mock_job(running_from_vm) as mock_job:
            with self.assertRaisesRegex(app.UsageError, "Invalid action"):
                main(["cli"])

            with self.assertRaisesRegex(app.UsageError, "Command is required"):
                main(["cli", "start"])

            main(["cli", "start", "--", "test_command"], flag_values=fv)
            if not running_from_vm:
                # Bundling should happen if running locally.
                mock_job._bundler.bundle.assert_called()
