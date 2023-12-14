# Copyright © 2023 Apple Inc.

"""Creates a VM and executes the given command on it.

By default, the script will monitor the status of the job and restart/delete if necessary.

Possible actions: [start|stop|list]

    Start:
        - Starts the job and monitors it, periodically printing the status.
        - If the job is already running, this will resume monitoring instead of restarting the job.
            To stop the job, invoke `stop` explicitly.

    Stop:
        - If not specifying --retain_vm (default), tears down the VM.
        - If specifying --retain_vm, stops any running command on the VM, but keeps the VM alive.
            Running `start` again will execute the new command.

    List:
        - Lists all VMs.

Examples:

    # Use a project with sufficient CPU quota.
    axlearn gcp config activate --label=cpu

    # Simple launch. Everything after -- is treated as the command.
    axlearn gcp vm start --name=$USER-test -- python3 my_script.py

    # Terminate the command and delete the VM.
    axlearn gcp vm stop --name=$USER-test

    # For debugging, it's often useful to keep the VM alive even if the job completes/fails.
    # To do so, specify --retain_vm.
    axlearn gcp vm start --name=$USER-test --retain_vm -- python3 my_script.py

    # Stop the command without deleting the VM.
    axlearn gcp vm stop --name=$USER-test --retain_vm

    # List running VMs.
    axlearn gcp vm list

"""
# pylint: disable=protected-access

import enum
import os
import pathlib
import shlex
import subprocess
import time
from typing import Sequence

from absl import app, flags, logging

from axlearn.cloud.common.bundler import bundler_flags, get_bundler_config
from axlearn.cloud.common.utils import configure_logging, generate_job_name, parse_action
from axlearn.cloud.gcp.bundler import GCSTarBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import CPUJob
from axlearn.cloud.gcp.utils import catch_auth, common_flags, get_credentials, running_from_vm
from axlearn.cloud.gcp.vm import (
    _compute_resource,
    create_vm,
    delete_vm,
    format_vm_info,
    get_vm_node,
    get_vm_node_status,
    list_vm_info,
)
from axlearn.common.config import REQUIRED, Required, config_class

_COMMAND_SESSION_NAME = "command"
_SYNC_OUTPUTS_SESSION_NAME = "sync_outputs"
FLAGS = flags.FLAGS


def launch_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bundler_flags(flag_values=flag_values)
    flag_values.set_default("project", gcp_settings("project", required=False))
    flag_values.set_default("zone", gcp_settings("zone", required=False))
    flag_values.set_default("bundler_type", GCSTarBundler.TYPE)
    # Note: don't use generate_taskname() here, as the VM may not have $USER.
    flags.DEFINE_string("name", None, "Job name.", flag_values=flag_values)
    flags.DEFINE_string(
        "vm_type",
        "n2-standard-16",
        "VM type. For available options, see: "
        "https://cloud.google.com/compute/docs/general-purpose-machines",
        flag_values=flag_values,
    )
    flags.DEFINE_integer("disk_size", 64, "Disk size of the VM in GB.", flag_values=flag_values)
    flags.DEFINE_integer("max_tries", 1, "Max attempts to launch the job.", flag_values=flag_values)
    flags.DEFINE_integer(
        "retry_interval", 60, "Interval in seconds between tries.", flag_values=flag_values
    )
    flags.DEFINE_bool(
        "retain_vm",
        False,
        "Whether to keep VM around after job completes. Useful for debugging.",
        flag_values=flag_values,
    )


# TODO(markblee): Unify some of this with tpu_runner.
class CPURunnerJob(CPUJob):
    """Runs and monitors a command on a VM."""

    @config_class
    class Config(CPUJob.Config):
        """Configures CPURunnerJob."""

        # Remote output directory.
        output_dir: Required[str] = REQUIRED
        # VM instance type.
        vm_type: Required[str] = REQUIRED
        # Disk size in GB.
        disk_size: Required[int] = REQUIRED
        # Interval to poll status.
        status_interval_seconds: float = 30
        # Local output directory. Mainly useful for testing.
        local_output_dir: str = "/output"
        # Whether to retain the VM after job completes.
        retain_vm: bool = False

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg = super().from_flags(fv, **kwargs)
        cfg.name = cfg.name or generate_job_name()
        cfg.output_dir = (
            cfg.output_dir or f"gs://{gcp_settings('ttl_bucket')}/axlearn/jobs/{cfg.name}"
        )
        cfg.bundler = get_bundler_config(bundler_type=fv.bundler_type, spec=fv.bundler_spec)
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._output_dir = pathlib.Path(cfg.local_output_dir) / cfg.name
        self._status_file = self._output_dir / "status"
        self._run_log = self._output_dir / "run.log"

    class Status(enum.Enum):
        """Job status."""

        UNKNOWN = "UNKNOWN"
        FAILED = "FAILED"
        SUCCESS = "SUCCESS"
        # The VM itself has not been provisioned.
        NOT_STARTED = "NOT_STARTED"
        # The VM is provisioned, but the command is not running.
        NOT_RUNNING = "NOT_RUNNING"
        # The VM is provisioned and the command is running.
        RUNNING = "RUNNING"

    def _set_status_command(self, status: Status):
        """Returns a command to set the current process status."""
        return f"mkdir -p {self._output_dir} && echo {status.name} > {self._status_file}"

    def _sync_outputs(self, *, session: str, src: str, dst: str, interval_s: int):
        """Starts a screen session to sync outputs to gs."""
        logging.info("Starting log sync...")
        self._execute_remote_cmd(
            f"while true; do gsutil -m rsync -r {src} {dst}; sleep {interval_s}; done",
            detached_session=session,
            shell=True,
        )
        logging.info("Log sync started.")

    def _install_bundle(self):
        """Installs the bundle on remote VM."""
        cfg: CPURunnerJob.Config = self.config
        logging.info("Installing the bundle...")
        self._execute_remote_cmd(
            self._bundler.install_command(self._bundler.id(cfg.name)),
            shell=True,
        )

    def _start(self):
        """Creates the VM if not already running."""
        cfg: CPURunnerJob.Config = self.config
        create_vm(
            cfg.name,
            vm_type=cfg.vm_type,
            disk_size=cfg.disk_size,
            credentials=self._get_job_credentials(),
            bundler_type=self._bundler.TYPE,
        )

    def _delete(self):
        """Stops the job and optionally deletes the VM."""
        cfg: CPURunnerJob.Config = self.config
        credentials = self._get_job_credentials()
        if not get_vm_node(cfg.name, _compute_resource(credentials)):
            logging.info("VM %s doesn't exist, nothing to do.", cfg.name)
            return
        # Copy final outputs.
        logging.info("Copying any final outputs...")
        # Set check=False, since _delete can be invoked prior to outputs being written.
        self._execute_remote_cmd(
            f"gsutil cp -r {self._output_dir}/* {cfg.output_dir}/output/$HOSTNAME/",
            shell=True,
            check=False,
        )
        # Attempt to stop on the VM itself.
        logging.info("Stopping any existing command...")
        self._execute_remote_cmd(
            f"rm {self._status_file}; "
            f"screen -XS {_COMMAND_SESSION_NAME} quit; "
            f"screen -XS {_SYNC_OUTPUTS_SESSION_NAME} quit; ",
            check=False,
        )
        if not cfg.retain_vm:
            delete_vm(cfg.name, credentials=credentials)
        else:
            logging.info("Not attempting to delete VM %s since retain_vm=True.", cfg.name)

    def _run_command(self):
        """Runs the command."""
        cfg: CPURunnerJob.Config = self.config
        self._install_bundle()
        self._sync_outputs(
            session=_SYNC_OUTPUTS_SESSION_NAME,
            src=f"{self._output_dir}/",
            dst=f"{cfg.output_dir}/output/$HOSTNAME/",
            interval_s=60,
        )
        # Run commands and pipe to log. Wrap command in parentheses/subshell so we can pipe outputs
        # even if multiple subcommands.
        cmd = f"""mkdir -p {self._output_dir}; echo "Starting command..." >> {self._run_log};
            ({cfg.command}) 2>&1 | tee -a {self._run_log};
            if [ ${{PIPESTATUS[0]}} -eq 0 ]; then
                echo "Setting status to SUCCESS..." >> {self._run_log};
                {self._set_status_command(CPURunnerJob.Status.SUCCESS)};
            else
                echo "Setting status to FAILED..." >> {self._run_log};
                {self._set_status_command(CPURunnerJob.Status.FAILED)};
            fi
        """
        logging.info("Starting remote command...")
        self._execute_remote_cmd(
            cmd,
            shell=True,
            detached_session=_COMMAND_SESSION_NAME,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _get_status(self):
        """Gets current job status."""
        cfg: CPURunnerJob.Config = self.config
        vm = get_vm_node(cfg.name, _compute_resource(self._get_job_credentials()))
        if vm is None or get_vm_node_status(vm) != "BOOTED":
            return CPURunnerJob.Status.NOT_STARTED

        proc = self._execute_remote_cmd(
            f"""screen -wipe;
            if screen -ls {_COMMAND_SESSION_NAME}; then
                echo {CPURunnerJob.Status.RUNNING.name};
            elif [[ -f {self._status_file} ]]; then
                cat {self._status_file};
            else
                echo {CPURunnerJob.Status.NOT_RUNNING.name};
            fi
            """,
            shell=True,
            check=False,
        )
        valid_statuses = set(status.name for status in CPURunnerJob.Status)
        if proc.returncode == 0:
            for line in proc.stdout.split("\n"):
                line = line.strip()
                if line in valid_statuses:
                    return CPURunnerJob.Status[line]
        else:
            logging.warning(
                "Failed to get job status. stdout=%s, stderr=%s", proc.stdout, proc.stderr
            )
        return CPURunnerJob.Status.UNKNOWN

    def _execute(self):
        cfg: CPURunnerJob.Config = self.config

        logging.info(
            "Logs will be available via:\ngsutil cat %s",
            os.path.join(cfg.output_dir, str(self._run_log).lstrip("/")),
        )
        while True:
            status = self._get_status()
            if status == CPURunnerJob.Status.SUCCESS:
                logging.info("Job completed successfully.")
                self._delete()
                return
            # Command failed, teardown and raise so we can retry.
            elif status == CPURunnerJob.Status.FAILED:
                # Delete only if there's a next try. Once retries are exhausted, _delete is invoked
                # automatically.
                if cfg.max_tries > 1:
                    self._delete()
                raise ValueError("Job failed.")
            # VM doesn't exist -- create it and launch the command.
            elif status == CPURunnerJob.Status.NOT_STARTED:
                self._start()
                logging.info("VMs have started.")
            # VM is ready but not running command -- start the command.
            elif status == CPURunnerJob.Status.NOT_RUNNING:
                logging.info("Job is not running. Running the command...")
                self._run_command()
            else:
                logging.info("Job currently has status: %s", status)
                time.sleep(cfg.status_interval_seconds)


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["start", "stop", "list"])

    if action == "list":
        print(format_vm_info(list_vm_info(get_credentials())))
        return

    cfg = CPURunnerJob.from_flags(flag_values)

    if action == "start":
        # Use shlex join so that quoted commands (e.g. 'a && b') retain quotes.
        command = shlex.join(argv[2:])
        if not command:
            raise app.UsageError("Command is required.")

        cfg.set(command=command)
        job = cfg.instantiate()
        # If not on bastion, bundle early so the user can cd away from cwd.
        if not running_from_vm():
            job._bundler.bundle(cfg.name)
        job.execute()
    elif action == "stop":
        job = cfg.set(command="").instantiate()
        job._delete()
    else:
        # Unreachable -- `parse_action` will handle validation.
        raise app.UsageError(f"Unsupported action: {action}")


if __name__ == "__main__":
    launch_flags()
    configure_logging(logging.INFO)
    app.run(main)
