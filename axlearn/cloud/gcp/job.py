# Copyright © 2023 Apple Inc.

"""Utilities for executing commands on GCP.

Note that these utilities do not handle resource management.
"""
import atexit
import logging
import os
import pathlib
import re
import shlex
import subprocess
from typing import Any, Dict, Optional, Sequence, Union

from google.auth.credentials import Credentials

from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import subprocess_run
from axlearn.cloud.gcp.scopes import DEFAULT_TPU_SCOPES
from axlearn.cloud.gcp.tpu import _qrm_resource, _tpu_resource, get_queued_tpu_node, get_tpu_node
from axlearn.cloud.gcp.utils import get_credentials, running_from_vm
from axlearn.common.config import REQUIRED, Required, config_class


class GCPJob(Job):
    """Base GCP Job definition."""

    @config_class
    class Config(Job.Config):
        """Configures GCPJob."""

        # GCP project.
        project: Required[str] = REQUIRED
        # GCP zone.
        zone: Required[str] = REQUIRED
        # If not none, the current job will be executed as the service account.
        service_account: Optional[str] = None

    def _get_job_credentials(
        self,
        impersonate_scopes: Optional[Sequence[str]] = None,
    ) -> Credentials:
        """Returns the credentials the job runs as.

        Note that credentials are temporary and should be created on demand.

        Args:
            impersonate_scopes: Scopes of the impersonation token,
                following https://developers.google.com/identity/protocols/oauth2/scopes

        Returns:
            The temporary credentials, possibly impersonating `cfg.service_account`.
        """
        return get_credentials(
            impersonate_account=self.config.service_account, impersonate_scopes=impersonate_scopes
        )


class TPUJob(GCPJob):
    """Executes arbitrary commands on TPU-VMs."""

    @config_class
    class Config(GCPJob.Config):
        """Configures TPUJob."""

        tpu_type: Required[str] = REQUIRED
        # Number of TPU slices.
        num_slices: int = 1

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._local_home = pathlib.Path.home()
        self._use_iap = None  # Infer from public IP.

    def _ensure_ssh_keys(self):
        """Ensures SSH keys exist, or raises ValueError. Only necessary on remote VM."""
        # Seem to need to nuke this every time to avoid MITM warnings.
        hosts_file = self._local_home / ".ssh/google_compute_known_hosts"
        if hosts_file.exists():
            hosts_file.unlink()

        ssh_key = self._local_home / ".ssh/google_compute_engine"
        proc = subprocess_run(f"ssh-add {ssh_key}", check=False, capture_output=True)
        if proc.returncode:
            logging.warning("SSH key %s does not exist yet.", ssh_key)

    def _infer_iap(self):
        """Infers whether instance has public IP. If not, we tunnel through IAP."""
        if self._use_iap is None:
            cfg = self.config
            if cfg.num_slices > 1:
                node = get_queued_tpu_node(
                    cfg.name,
                    _qrm_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
                )
            else:
                node = get_tpu_node(
                    cfg.name,
                    _tpu_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
                )
            if node is None:
                raise ValueError(f"Expected TPU {cfg.name} to exist")
            for endpoint in node.get("networkEndpoints", []):
                for access_config in endpoint.get("accessConfig", []):
                    if access_config.get("natIP", None):
                        logging.info("Detected a public IP, not using IAP.")
                        self._use_iap = False
                        return False
            logging.info("Didn't find a public IP, using IAP.")
            self._use_iap = True
        return self._use_iap

    def _execute_remote_cmd(
        self,
        cmd: str,
        *,
        worker: Union[int, str] = "all",
        detached_session: Optional[str] = None,
        batch_size: Union[int, str] = 100,
        extra_ssh_flags: str = "",
        **kwargs,
    ) -> Sequence[subprocess.CompletedProcess]:
        """Executes a command on existing TPU-VM(s).

        Args:
            cmd: Command to run.
            worker: Worker ID. Defaults to "all".
            wait: Whether to wait for process to complete. If True, waits for command to complete,
                and returns a completed process. Caller can inspect outputs or exit codes. If False,
                spawns and returns a process. Caller can listen to logs in realtime.
            detached_session: If not None, run commands behind `screen` in detached mode. This is
                useful for persisting commands even if SSH is terminated. If not None, should be a
                string containing the session name.
            batch_size: Number of concurrent command executions. If 'all', run all commands
                simultaneously.
            extra_ssh_flags: Extra gcloud ssh flags.
            **kwargs: Forwarded to subprocess.

        Returns:
            A list of completed subprocesses. Each corresponds to execution of the command on a
            single slice.

        Raises:
            ValueError: If the name of the detached screen session is too long.
        """
        cfg = self.config
        from_vm = running_from_vm()
        cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
        if from_vm:
            self._ensure_ssh_keys()
            extra_ssh_flags = f"--internal-ip {extra_ssh_flags}"
        elif self._infer_iap():
            # Infer IAP flag if not running from VM.
            extra_ssh_flags = f"--tunnel-through-iap {extra_ssh_flags}"
        cmd = f"sudo bash -c {cmd}"
        if detached_session:
            # Even though the official limit is 100 chars, screen seems to silently exit even before
            # that.
            if len(detached_session) > 80:
                raise ValueError(f"Screen name {detached_session} is too long.")
            cmd = f"sudo screen -dmS {detached_session} {cmd}"
        logging.debug("Executing remote command on worker [%s]: '%s'", worker, cmd)
        if cfg.num_slices > 1:
            slices = [f"{cfg.name}-{i}" for i in range(cfg.num_slices)]
        else:
            slices = [cfg.name]
        procs = []
        for s in slices:
            cmd_for_slice = (
                f"gcloud alpha compute -q tpus tpu-vm ssh {s} "
                f"--project={cfg.project} "
                f"--zone={cfg.zone} "
                f"--worker={worker} "
                f"--batch-size={batch_size} "
                f'{extra_ssh_flags} --command="{cmd}"'
            )
            proc = subprocess_run(cmd_for_slice, **_prepare_subprocess_kwargs(kwargs))
            procs.append(proc)
        return procs

    def _execute(self) -> Any:
        """Performs some computation on remote TPU-VMs."""
        cfg: TPUJob.Config = self.config
        self._execute_remote_cmd(cfg.command)

    def execute(self) -> Any:
        """Wraps _execute with ssh-agent and retries. All args and kwargs are forwarded."""
        if running_from_vm():
            _start_ssh_agent()
        return super().execute()


class CPUJob(GCPJob):
    """Executes arbitrary commands on CPU VMs."""

    Config = GCPJob.Config

    def _execute_remote_cmd(
        self, cmd: str, *, detached_session: Optional[str] = None, **kwargs
    ) -> subprocess.CompletedProcess:
        """Executes a command on an existing VM.

        Args:
            cmd: Command to run.
            detached_session: If not None, run commands behind `screen` in detached mode. This is
                useful for persisting commands even if SSH is terminated. If not None, should be a
                string containing the session name.
            **kwargs: Forwarded to subprocess.

        Returns:
            A subprocess, either live or completed.
        """
        cfg = self.config
        logging.debug("Executing remote command: '%s'", cmd)
        cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
        # Use login shell. Note `-i` is not interactive.
        cmd = f"sudo -i bash -c {cmd}"
        if detached_session:
            cmd = f"sudo screen -dmS {detached_session} {cmd}"
        # Run via screen to persist command after SSH.
        cmd = (
            f"gcloud compute -q ssh {cfg.name} "
            f"--project={cfg.project} "
            f"--zone={cfg.zone} "
            f'--command="{cmd}"'
        )
        proc = subprocess_run(cmd, **_prepare_subprocess_kwargs(kwargs))
        logging.debug("Finished launching: '%s'.", cmd)
        return proc

    def _execute(self) -> Any:
        """Performs some computation on remote VMs."""
        cfg: CPUJob.Config = self.config
        self._execute_remote_cmd(cfg.command)


def _prepare_subprocess_kwargs(kwargs: Dict) -> Dict:
    """Enable check=True and capture all outputs by default."""
    kwargs.setdefault("text", True)
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", kwargs.keys().isdisjoint(["stdout", "stderr"]))
    return kwargs


def _kill_ssh_agent():
    """Terminates ssh-agent, e.g. as started by `_start_ssh_agent`."""
    subprocess_run("ssh-agent -k", check=False, capture_output=True)
    os.environ.pop("SSH_AUTH_SOCK", None)
    os.environ.pop("SSH_AGENT_PID", None)


def _start_ssh_agent():
    """Starts ssh-agent for SSH key handling.

    The ssh-agent is automatically terminated when the program exits.
    """
    if not os.getenv("SSH_AGENT_PID"):
        logging.info("ssh-agent is not running, starting it now...")
        process = subprocess_run("ssh-agent -s", stdout=subprocess.PIPE, check=True, text=True)
        # Example format:
        # pylint: disable-next=line-too-long
        # SSH_AUTH_SOCK=/tmp/ssh-g4aYlFVLLugX/agent.52090; export SSH_AUTH_SOCK;\nSSH_AGENT_PID=52091; export SSH_AGENT_PID;\necho Agent pid 52091;\n
        match = re.search(
            r"SSH_AUTH_SOCK=([^;]+);.*SSH_AGENT_PID=([^;]+);",
            process.stdout,
            re.MULTILINE | re.DOTALL,
        )
        auth_sock, agent_pid = match.groups()  # pytype: disable=attribute-error
        os.environ["SSH_AUTH_SOCK"] = auth_sock
        os.environ["SSH_AGENT_PID"] = agent_pid
        atexit.register(_kill_ssh_agent)
    logging.info("ssh-agent is running.")


def _prepare_cmd_for_gcloud_ssh(cmd: str) -> str:
    """Handles bash escapes to ensure `cmd` is compatible with gcloud `--command`."""
    cmd = shlex.quote(cmd)
    cmd = cmd.replace('"', '\\"')  # Escape double quotes for --command.
    cmd = cmd.replace("$", r"\$")  # Escape $ for --command.
    return cmd


def docker_command(
    cmd: str,
    *,
    image: str,
    detached_session: Optional[str] = None,
    env: Optional[Sequence[str]] = None,
    volumes: Optional[Dict[str, str]] = None,
    extra_docker_flags: Optional[Sequence[str]] = None,
) -> str:
    """Wraps a command with docker run.

    Args:
        cmd: Command to run.
        image: Docker image name.
        detached_session: If not None, runs in detached mode with the given name.
        env: Optional env vars to expose to container.
        volumes: Optional mapping of source/target volumes to mount.
        extra_docker_flags: Optional extra flags for docker run.

    Returns:
        The docker command.
    """
    cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
    cmd = f"/bin/bash -c {cmd}"
    env = " ".join([f"-e {e}" for e in (env or [])])
    volumes = " ".join([f"-v {src}:{dst}" for src, dst in (volumes or {}).items()])
    extra_docker_flags = " ".join(extra_docker_flags or [])
    detached = f"-d --name={detached_session}" if detached_session else ""
    cmd = (
        f"docker run --rm --privileged -u root --network=host {detached} {env} {volumes} "
        f"{extra_docker_flags} {image} {cmd}"
    )
    logging.debug("Docker run command: %s", cmd)
    return cmd
