# Copyright © 2023 Apple Inc.

"""AXLearn Google Cloud CLI."""

import logging

from axlearn.cli.utils import CommandGroup, get_path
from axlearn.cloud.common.config import load_configs
from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import infer_cli_name
from axlearn.cloud.gcp.config import CONFIG_NAMESPACE


def add_cmd_group(*, parent: CommandGroup):
    """Adds the root GCP command group."""

    gcp_cmd = CommandGroup("gcp", parent=parent)

    _, gcp_configs = load_configs(CONFIG_NAMESPACE, required=True)
    active_config = gcp_configs.get("_active", None)

    if active_config is None:
        logging.warning(
            "No GCP project has been activated; please run `%s gcp config activate`.",
            infer_cli_name(),
        )

    # Set common flags.
    gcp_cmd.add_flag(
        "--project", undefok=True, default=get_path(gcp_configs, f"{active_config}.project", None)
    )
    gcp_cmd.add_flag(
        "--zone", undefok=True, default=get_path(gcp_configs, f"{active_config}.zone", None)
    )

    # Configure projects.
    gcp_cmd.add_cmd_from_module(
        "config", module="axlearn.cloud.gcp.config", help="Configure GCP settings"
    )

    # Interact with compute.
    gcp_cmd.add_cmd_from_bash("sshvm", command="gcloud compute ssh", help="SSH into a VM")
    gcp_cmd.add_cmd_from_bash(
        "sshtpu",
        command="gcloud alpha compute tpus tpu-vm ssh",
        help="SSH into a TPU-VM",
    )

    # Interact with jobs.
    # TODO(markblee): Make the distinction between launch, tpu, and bastion more clear.
    gcp_cmd.add_cmd_from_module(
        "bundle",
        module="axlearn.cloud.gcp.bundler",
        help="Bundle the local directory",
    )
    gcp_cmd.add_cmd_from_module(
        "launch",
        module="axlearn.cloud.gcp.jobs.launch",
        help="Launch arbitrary commands on remote compute",
    )
    gcp_cmd.add_cmd_from_module(
        "vm",
        module="axlearn.cloud.gcp.jobs.cpu_runner",
        help="Create a VM and execute the given command on it",
    )
    gcp_cmd.add_cmd_from_module(
        "bastion",
        module="axlearn.cloud.gcp.jobs.bastion_vm",
        help="Launch jobs through Bastion orchestrator",
    )
    gcp_cmd.add_cmd_from_module(
        "dataflow",
        module="axlearn.cloud.gcp.jobs.dataflow",
        help="Run Dataflow jobs locally or on GCP",
    )
    gcp_cmd.add_cmd_from_module(
        "logs",
        module="axlearn.cloud.gcp.jobs.logs",
        help="View job Cloud Logging logs.",
    )

    # Auth command.
    docker_repo = get_path(gcp_configs, f"{active_config}.docker_repo", None)
    # For the purposes of the following commands, see:
    # https://cloud.google.com/docs/authentication/provide-credentials-adc#gcloud-credentials
    auth_command = "gcloud auth login && gcloud auth application-default login"
    if docker_repo:
        # Note: we currently assume that docker_repo is a GCP one.
        auth_command += f" && gcloud auth configure-docker {registry_from_repo(docker_repo)}"
    gcp_cmd.add_cmd_from_bash(
        "auth",
        command=auth_command,
        help="Authenticate to GCP",
        # Match no flags -- `gcloud auth ...` doesn't support `--project`, `--zone`, etc.
        filter_argv="a^",
    )
