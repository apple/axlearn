# Copyright © 2023 Apple Inc.

"""Tests jobs by launching commands on TPUs/VMs.

    python3 -m axlearn.cloud.gcp.job_test TPUJobTest.test_execute_from_local \
        --tpu_type=v4-8 --project=my-project --zone=my-zone

    python3 -m axlearn.cloud.gcp.job_test CPUJobTest.test_execute_from_local \
        --project=my-project --zone=my-zone

"""
import atexit
import contextlib
import os
import subprocess
from typing import Union
from unittest import mock

import pytest
from absl import flags, logging
from absl.testing import absltest

from axlearn.cloud.common.utils import configure_logging, generate_taskname
from axlearn.cloud.gcp.bundler import GCSTarBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import CPUJob, TPUJob, _kill_ssh_agent, _start_ssh_agent
from axlearn.cloud.gcp.tpu import create_tpu, delete_tpu
from axlearn.cloud.gcp.utils import common_flags, get_credentials
from axlearn.cloud.gcp.vm import create_vm, delete_vm
from axlearn.common.config import config_class
from axlearn.common.test_utils import TestCase


@contextlib.contextmanager
def mock_job(module_name: str):
    with mock.patch(f"{module_name}.get_credentials", return_value=None):
        yield


def _private_flags():
    common_flags()
    flags.DEFINE_string("tpu_type", "v4-8", "TPU type to test with")


FLAGS = flags.FLAGS


class DummyLaunchJob(TPUJob):
    """A dummy TPU job."""

    def _execute(self) -> Union[subprocess.CompletedProcess, subprocess.Popen]:
        """Provisions a TPU and launches a command."""
        cfg: TPUJob.Config = self.config
        bundle_id = self._bundler.bundle(cfg.name)
        credentials = get_credentials()
        create_tpu(
            cfg.name,
            tpu_type=cfg.tpu_type,
            bundler_type=self._bundler.TYPE,
            credentials=credentials,
        )
        out = self._execute_remote_cmd(
            f"{self._bundler.install_command(bundle_id)} && {cfg.command}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        delete_tpu(cfg.name, credentials=credentials)
        return out[0]


@pytest.mark.tpu
@pytest.mark.gs_login
class TPUJobTest(TestCase):
    """Tests TPUJob."""

    def test_execute_from_local(self):
        jobname = generate_taskname()
        atexit.register(delete_tpu, jobname, credentials=get_credentials())
        project = gcp_settings("project")
        zone = gcp_settings("zone")
        cfg = DummyLaunchJob.default_config().set(
            name=jobname,
            project=project,
            zone=zone,
            max_tries=1,
            retry_interval=60,
            bundler=GCSTarBundler.default_config(),
            tpu_type=FLAGS.tpu_type,
            command="pip list",
        )
        out = cfg.instantiate().execute()
        self.assertIn("axlearn", out.stdout)


class DummyBastionJob(CPUJob):
    """A dummy CPU job."""

    @config_class
    class Config(CPUJob.Config):
        # Type of VM.
        vm_type: str
        # Disk size in GB.
        disk_size: int

    def _execute(self) -> subprocess.CompletedProcess:
        """Provisions and launches a command on a VM."""
        cfg: DummyBastionJob.Config = self.config
        self._bundler.bundle(cfg.name)
        create_vm(
            cfg.name,
            vm_type=cfg.vm_type,
            disk_size=cfg.disk_size,
            bundler_type=self._bundler.TYPE,
            credentials=get_credentials(),
        )
        return self._execute_remote_cmd(
            cfg.command,
            detached=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


@pytest.mark.gs_login
class CPUJobTest(TestCase):
    """Tests CPUJob."""

    def test_execute_from_local(self):
        jobname = generate_taskname()
        atexit.register(delete_vm, jobname, credentials=get_credentials())
        cfg = DummyBastionJob.default_config().set(
            name=jobname,
            project=gcp_settings("project"),
            zone=gcp_settings("zone"),
            max_tries=1,
            retry_interval=60,
            bundler=GCSTarBundler.default_config(),
            vm_type="n2-standard-2",
            disk_size=64,
            command=f"mkdir -p /tmp/{jobname} && ls /tmp/",
        )
        out = cfg.instantiate().execute()
        self.assertIn(jobname, out.stdout)


class UtilTest(TestCase):
    """Tests util functions."""

    def test_ssh_agent(self):
        self.assertIsNone(os.getenv("SSH_AGENT_PID"))
        self.assertIsNone(os.getenv("SSH_AUTH_SOCK"))
        _start_ssh_agent()
        self.assertRegex(
            os.getenv("SSH_AUTH_SOCK", ""),
            r"/tmp/ssh-.+/agent.(\d+)",
        )
        self.assertTrue(os.path.exists(os.getenv("SSH_AUTH_SOCK")))
        self.assertRegex(os.getenv("SSH_AGENT_PID", ""), r"\d+")
        _kill_ssh_agent()
        self.assertIsNone(os.getenv("SSH_AGENT_PID"))
        self.assertIsNone(os.getenv("SSH_AUTH_SOCK"))


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    absltest.main()
