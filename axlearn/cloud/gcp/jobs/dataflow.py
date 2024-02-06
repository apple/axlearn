# Copyright © 2023 Apple Inc.

"""Runs Dataflow jobs locally or in GCP.

The flow is:
1. Builds the dataflow worker image.
2. Runs the dataflow job either on dataflow (if runner is 'DataflowRunner', the default) or locally
    (if runner is 'DirectRunner').

Note: killing the script is not sufficient to stop a remote dataflow job; please use
`axlearn gcp dataflow stop` to do so. See below for examples.

Possible actions: [start|stop]

    Start:
        - If using DataflowRunner (default), builds the worker image, submits the job to Dataflow,
            and monitors the status. To stop the job, use `axlearn gcp dataflow stop`.
        - If using DirectRunner, builds the worker and runs the job locally. Exiting the script
            terminates the job.

    Stop:
        - Attempts to stop any remote Dataflow job(s) matching job name. This is only useful if the
        job was started with DataflowRunner; for DirectRunner, exiting the script stops the job.

Examples:

    # Simple launch for the wordcount example[1], which implicitly uses DataflowRunner.
    # Flags like project, region, and temp_location will be inferred from settings.
    axlearn gcp dataflow start \
        --name=$USER-dataflow \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=base_image=apache/beam_python3.10_sdk:2.52.0 \
        --bundler_spec=target=dataflow \
        -- "'
        python3 -m apache_beam.examples.wordcount \
            --input=gs://dataflow-samples/shakespeare/kinglear.txt \
            --output=gs://STORAGE_BUCKET/results/outputs \
        '"

    # Launch from a VM. Note that /tmp/output_dir is on the VM. You can also point to gs://.
    #
    # A breakdown of the command:
    # * `axlearn gcp vm` launches everything after `--` on a VM.
    # * `--bundler_spec=extras=dataflow` installs necessary deps on the VM for launching dataflow.
    # * `--dataflow_spec=runner=DirectRunner` runs locally on the VM, rather than in dataflow.
    #
    # Note: If running multiple commands, the quotes "'...'" are necessary to avoid splitting them.
    # Note: To launch on Dataflow, simply remove `--dataflow_spec=runner=DirectRunner` flag.
    #
    axlearn gcp vm start --name=$USER-dataflow --bundler_spec=extras=dataflow --retain_vm -- \
        axlearn gcp dataflow start \
            --name=$USER-dataflow \
            --dataflow_spec=runner=DirectRunner \
            --bundler_spec=dockerfile=Dockerfile \
            --bundler_spec=base_image=apache/beam_python3.10_sdk:2.52.0 \
            --bundler_spec=target=dataflow \
            -- "'
            rm -r /tmp/output_dir; \
            python3 -m apache_beam.examples.wordcount \
                --input=gs://dataflow-samples/shakespeare/kinglear.txt \
                --output=/tmp/output_dir/outputs
            '"

    # Stop the VM (if running on VM).
    axlearn gcp vm stop --name=$USER-dataflow

    # Stop the job (if running on dataflow).
    axlearn gcp dataflow stop --name=$USER-dataflow

Reference [1]:
https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python#run-the-pipeline-on-the-dataflow-service
"""
# pylint: disable=protected-access

import platform
import re
import shlex
import signal
import subprocess
from typing import Any, Dict, List, Sequence, Tuple

from absl import app, flags, logging
from google.auth.credentials import Credentials
from googleapiclient import discovery, errors

from axlearn.cloud.common.bundler import DockerBundler, bundler_flags, get_bundler_config
from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import (
    canonicalize_to_list,
    configure_logging,
    generate_job_name,
    handle_popen,
    parse_action,
    parse_kv_flags,
    send_signal,
)
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import GCPJob
from axlearn.cloud.gcp.utils import catch_auth, common_flags, get_credentials
from axlearn.common.config import REQUIRED, Required, config_class

FLAGS = flags.FLAGS


def launch_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bundler_flags(flag_values=flag_values)
    flag_values.set_default("project", gcp_settings("project", required=False))
    flag_values.set_default("zone", gcp_settings("zone", required=False))
    flag_values.set_default("bundler_type", ArtifactRegistryBundler.TYPE)
    # Note: don't use generate_taskname() here, as the VM may not have $USER.
    flags.DEFINE_string("name", None, "Dataflow job name.", flag_values=flag_values)
    flags.DEFINE_integer("max_tries", 1, "Max attempts to launch the job.", flag_values=flag_values)
    flags.DEFINE_integer(
        "retry_interval", 60, "Interval in seconds between tries.", flag_values=flag_values
    )
    flags.DEFINE_string("vm_type", "n2-standard-2", "Worker VM type.", flag_values=flag_values)
    flags.DEFINE_multi_string(
        "dataflow_spec",
        [],
        "Bundler spec provided as key=value.",
        flag_values=flag_values,
    )


class DataflowJob(GCPJob):
    """Launches a dataflow job from local."""

    @config_class
    class Config(GCPJob.Config):
        # Worker VM type.
        vm_type: Required[str] = REQUIRED
        # Setup command. This is used to prepare the local machine for running `cfg.command`,
        # including installing docker (if not already present) and building the worker image.
        # `cfg.command` will then be run within the worker image, to ensure a consistent build +
        # execute environment.
        setup_command: Required[str] = REQUIRED

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg = super().from_flags(fv, **kwargs)
        cfg.name = cfg.name or generate_job_name()

        # Construct bundler.
        cfg.bundler = get_bundler_config(bundler_type=fv.bundler_type, spec=fv.bundler_spec)
        if not issubclass(cfg.bundler.klass, DockerBundler):
            raise NotImplementedError("Expected a DockerBundler.")
        cfg.bundler.image = cfg.bundler.image or cfg.name

        # Construct bundle command.
        docker_setup_cmd = (
            # Install a docker version with buildkit support.
            # Buildkit is required for actual multi-stage '--target' (without it, docker will
            # attempt to build all stages up to the target).
            # https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script
            "if [[ ! -x $(which docker) ]]; then "
            "curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh;"
            "fi"
        )
        docker_auth_cmd = (
            f"gcloud auth configure-docker {registry_from_repo(cfg.bundler.repo)} --quiet"
        )
        bundle_cmd = " ".join(
            [
                f"python3 -m {bundler.__name__} --name={cfg.name}",
                *_docker_bundler_to_flags(cfg.bundler),
            ]
        )

        # Construct dataflow command.
        dataflow_spec, multi_flags = cls._dataflow_spec_from_flags(cfg, fv)
        dataflow_flags = " ".join(
            sorted(flags.flag_dict_to_args(dataflow_spec, multi_flags=multi_flags))
        )
        cfg.setup_command = f"{docker_setup_cmd} && {docker_auth_cmd} && {bundle_cmd}"
        cfg.command = f"{cfg.command} {dataflow_flags}"
        return cfg

    @classmethod
    def _dataflow_spec_from_flags(
        cls, cfg: Config, fv: flags.FlagValues
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Returns a flag dict and a list of flags considered as 'multi-flags'."""
        # Construct dataflow args, providing some defaults.
        service_account = cfg.service_account or gcp_settings("service_account_email")
        dataflow_spec = {
            "job_name": cfg.name,
            "project": cfg.project,
            "region": cfg.zone.rsplit("-", 1)[0],
            "worker_machine_type": cfg.vm_type,
            "sdk_container_image": f"{cfg.bundler.repo}/{cfg.bundler.image}:{cfg.name}",
            "temp_location": f"gs://{gcp_settings('ttl_bucket')}/tmp/{cfg.name}/",
            "service_account_email": service_account,
            "dataflow_service_options": ["enable_secure_boot", "enable_google_cloud_heap_sampling"],
            "experiments": ["use_network_tags=allow-internet-egress", "use_runner_v2"],
            "no_use_public_ips": None,
            "runner": "DataflowRunner",
        }
        dataflow_spec.update(parse_kv_flags(fv.dataflow_spec, delimiter="="))

        # From https://cloud.google.com/dataflow/docs/reference/pipeline-options#basic_options,
        # these seem to be the only multi-flags. Users can still provide comma-separated strings for
        # other multi-flags, if any.
        multi_flags = ["dataflow_service_options", "experiments"]
        for multi_flag in multi_flags:
            dataflow_spec[multi_flag] = canonicalize_to_list(dataflow_spec[multi_flag])

        # Attempt to infer network settings, if not specified.
        if "network" not in dataflow_spec and "subnetwork" not in dataflow_spec:
            # Following https://cloud.google.com/dataflow/docs/guides/specifying-networks,
            # only --subnetwork is required, and we use the "complete URL" format.
            subnetwork = gcp_settings("subnetwork")
            subnetwork_pat = r"projects/.+/regions/.+/subnetworks/.+"
            if not re.match(subnetwork_pat, subnetwork):
                raise ValueError(
                    f"Expected subnetwork with format {subnetwork_pat}, got {subnetwork}"
                )
            dataflow_spec["subnetwork"] = f"https://www.googleapis.com/compute/v1/{subnetwork}"

        return dataflow_spec, multi_flags

    def _delete(self):
        cfg: DataflowJob.Config = self.config
        # Attempt to stop on dataflow. Note that this is not foolproof, e.g., the command may start
        # multiple jobs.
        _stop_dataflow_job(project=cfg.project, zone=cfg.zone, job_name=cfg.name)

    def _execute(self):
        cfg: DataflowJob.Config = self.config
        # Run the setup command locally, but the launch command via docker.
        # This is to ensure that the launch environment matches the worker environment.
        processor = platform.processor().lower()
        if "arm" in processor:
            # Disable running from docker on Mac M1 chip due to qemu core dump bug.
            # https://github.com/docker/for-mac/issues/5342.
            logging.info(
                (
                    "%s processor detected. "
                    "Skipping docker launch and running from local environment instead."
                ),
                processor,
            )
            cmd = cfg.command
        else:
            cmd = (
                "docker run --rm --entrypoint /bin/bash "
                f"{self._bundler.id(cfg.name)} -c '{cfg.command}'"
            )
        cmd = f"{cfg.setup_command} && {cmd}"
        cmd = f"bash -c {shlex.quote(cmd)}"
        logging.info("Executing in subprocess: %s", cmd)
        with subprocess.Popen(cmd, shell=True, text=True) as proc:
            # Attempt to cleanup the process when exiting.
            def sig_handler(sig: int, _):
                send_signal(proc, sig=sig)

            # SIGTERM for kill, SIGINT for ctrl+c, and SIGHUP for screen quit.
            for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
                signal.signal(sig, sig_handler)

            handle_popen(proc)


def _docker_bundler_to_flags(cfg: DockerBundler.Config) -> List[str]:
    """Converts docker bundler config to a string of flags."""
    # TODO(markblee): Add a config to_spec() method to mirror from_spec().
    spec_flags = [
        f"--bundler_type={cfg.klass.TYPE}",
        f"--bundler_spec=dockerfile={cfg.dockerfile}",
        f"--bundler_spec=image={cfg.image}",
        f"--bundler_spec=repo={cfg.repo}",
    ]
    if cfg.target:
        spec_flags.append(f"--bundler_spec=target={cfg.target}")
    if cfg.platform:
        spec_flags.append(f"--bundler_spec=platform={cfg.platform}")
    spec_flags += [f"--bundler_spec={k}={v}" for k, v in cfg.build_args.items()]
    return spec_flags


def _dataflow_resource(credentials: Credentials):
    # cache_discovery=False to avoid:
    # https://github.com/googleapis/google-api-python-client/issues/299
    dataflow = discovery.build("dataflow", "v1b3", credentials=credentials, cache_discovery=False)
    return dataflow.projects().locations().jobs()


def _get_dataflow_jobs(*, project: str, zone: str, job_name: str) -> List[Dict[str, Any]]:
    """Attempts to retrieve a dataflow job.

    If dataflow job is not found, returns None.
    If job name matches multiple jobs, returns all of them.

    Reference:
    https://cloud.google.com/dataflow/docs/reference/rest/v1b3/projects.jobs#Job
    """
    # Need to use list endpoint to filter by name.
    # Note that job name is user-specified, whereas job ID is generated by dataflow.
    resource = _dataflow_resource(get_credentials())
    resp = resource.list(projectId=project, location=zone, name=job_name).execute()
    return resp.get("jobs", [])


def _stop_dataflow_job(*, project: str, zone: str, job_name: str):
    """Attempts to cancel a dataflow job.

    Reference:
    https://cloud.google.com/dataflow/docs/reference/rest/v1b3/projects.jobs#jobstate
    https://googleapis.github.io/google-api-python-client/docs/dyn/dataflow_v1b3.projects.locations.jobs.html#update
    """
    jobs = _get_dataflow_jobs(project=project, zone=zone, job_name=job_name)
    for job in jobs:
        try:
            # Terminal states cannot be modified. See references above.
            if job.get("currentState") in {
                "JOB_STATE_DONE",
                "JOB_STATE_FAILED",
                "JOB_STATE_STOPPED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_UPDATED",
                "JOB_STATE_DRAINED",
            }:
                logging.info(
                    "Dataflow job %s (id %s) is already terminal with state %s.",
                    job["name"],
                    job["id"],
                    job["currentState"],
                )
                continue
            logging.info(
                "Dataflow job %s (id %s) has state %s, setting it to JOB_STATE_CANCELLED...",
                job["name"],
                job["id"],
                job["currentState"],
            )
            resource = _dataflow_resource(get_credentials())
            resource.update(
                projectId=project,
                location=zone,
                jobId=job["id"],
                body={"requestedState": "JOB_STATE_CANCELLED"},
            ).execute()
        except (ValueError, errors.HttpError) as e:
            logging.warning(
                "Failed to cancel dataflow job: %s. Consider cancelling from the console.", e
            )


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["start", "stop"])

    if action == "start":
        command = " ".join(argv[2:])
        if not command:
            raise app.UsageError("Command is required.")

        # Ensure that command is supplied to `from_flags`.
        cfg = DataflowJob.from_flags(flag_values, command=command)
        if not cfg.bundler.repo or not cfg.bundler.image:
            raise app.UsageError(
                "Worker bundler repo and image are required. "
                f"Instead, got repo={cfg.bundler.repo} image={cfg.bundler.image}."
            )

        job = cfg.instantiate()
        job.execute()
    elif action == "stop":
        job = DataflowJob.from_flags(flag_values, command="").instantiate()
        job._delete()
    else:
        raise ValueError(f"Unsupported action: {action}")


if __name__ == "__main__":
    launch_flags()
    configure_logging(logging.INFO)
    app.run(main)
