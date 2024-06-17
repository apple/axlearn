# Copyright © 2023 Apple Inc.

"""GCP general-purpose utilities."""

import enum
import functools
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Optional, Sequence

import google.auth
from absl import app, flags, logging
from google.auth import exceptions as gauthexceptions
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials

from axlearn.cloud.common.utils import Table, infer_cli_name, subprocess_run
from axlearn.cloud.gcp.scopes import DEFAULT_APPLICATION


def common_flags(**kwargs):
    """Defines common GCP flags. Keyword args will be forwarded to flag definitions."""
    flags.DEFINE_string("project", None, "The GCP project name.", **kwargs)
    flags.DEFINE_string("zone", None, "The GCP zone name.", **kwargs)


def get_credentials(
    *,
    impersonate_account: Optional[str] = None,
    impersonate_scopes: Optional[Sequence[str]] = None,
) -> Credentials:
    """Get gcloud credentials, or exits if unauthenticated.

    Args:
        impersonate_account: Service account to impersonate, if not None.
        impersonate_scopes: Scopes of the impersonation token,
            following https://developers.google.com/identity/protocols/oauth2/scopes.

    Returns:
        An authorized set of credentials.
    """

    try:
        credentials, project_id = google.auth.default()
        logging.log_first_n(logging.INFO, "Using credential for project_id=%s", 1, project_id)
    except (gauthexceptions.RefreshError, gauthexceptions.DefaultCredentialsError):
        logging.error("Please run '%s gcp auth' before this script.", infer_cli_name())
        logging.error("Please also verify if default project id is correct.")
        sys.exit(1)

    if impersonate_account:
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=impersonate_account,
            # If no scope provided, use the same scopes provided by
            # `gcloud auth application-default login`.
            target_scopes=impersonate_scopes or DEFAULT_APPLICATION,
        )

    return credentials


def running_from_vm() -> bool:
    """Check if we're running from GCP VM.

    Reference:
    https://cloud.google.com/compute/docs/instances/detect-compute-engine#use_the_metadata_server_to_detect_if_a_vm_is_running_in
    """
    out = subprocess.run(
        ["curl", "-s", "metadata.google.internal", "-i"],  # Curl silently.
        check=False,
        capture_output=True,
        text=True,
    )
    return (out.returncode == 0) and "Metadata-Flavor: Google" in out.stdout


def running_from_k8s() -> bool:
    """Check if we're running from K8s."""
    return os.environ.get("KUBERNETES_SERVICE_HOST", None) is not None


def validate_resource_name(name: Optional[str]):
    """Validates names (e.g. TPUs, VMs, jobs) to ensure compat with GCP.

    Reference:
    https://cloud.google.com/compute/docs/naming-resources#resource-name-format

    Raises:
        ValueError: If name is invalid.
    """
    if name is None or len(name) > 63 or re.fullmatch(r"^[a-z]([-a-z0-9]*[a-z0-9])?", name) is None:
        raise ValueError(
            f"{name} is not a valid resource name. Please see "
            "https://cloud.google.com/compute/docs/naming-resources#resource-name-format."
        )


def validate_k8s_name(name: str, *, num_workers: int, num_replicas: int):
    """Validates k8s name (e.g. TPUs, VMs, jobs) to ensure compat with GKE.

    Raises:
        ValueError: If name is invalid.
    """
    # K8s job name cannot exceed 63 chars. By default, GKE jobset also appends a suffix
    # "-job-<replica_id>-<host id>-<hash>" to each pod. The hash is typically 5 chars long.
    # https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set
    max_length = 63
    job_name = f"{name}-job-{num_replicas}-{num_workers}-abcde"
    if (excess := len(job_name) - max_length) > 0:
        raise ValueError(f"Job name {job_name} exceeds max ({max_length}) by {excess} chars.")

    # K8s jobset metadata name follows RFC 1123 subdomain regex.
    valid_regex = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
    if not re.fullmatch(valid_regex, name):
        raise ValueError(
            f"Job name {job_name} contains invalid characters. "
            "It should only contain lowercase alphanumerics, hyphens and periods, "
            "and must start and end with alphanumerics."
        )


def catch_auth(fn):
    """Wraps a function by catching google auth or k8s auth errors."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except gauthexceptions.RefreshError:
            logging.error("Please run `%s gcp auth`.", infer_cli_name())
            sys.exit(1)
        except Exception as e:
            try:
                # pylint: disable-next=import-error,import-outside-toplevel
                import kubernetes as k8s  # pytype: disable=import-error

                if (
                    isinstance(e, k8s.client.ApiException)
                    and e.status == 403  # pytype: disable=attribute-error
                    and e.reason == "Forbidden"  # pytype: disable=attribute-error
                ):
                    logging.error(
                        "Encountered error: %s "
                        "Please make sure that gke-gcloud-auth-plugin is installed: "
                        "gcloud components install gke-gcloud-auth-plugin\n",
                        e,
                    )
                    sys.exit(1)
            except (ImportError, ModuleNotFoundError):
                pass
            raise

    return wrapped


def load_kube_config(*, project: str, zone: str, cluster: str):
    """Load kube config.

    This reads or initializes cluster configuration, typically under `~/.kube/config`.

    It's a thin wrapper around `k8s.config.load_kube_config` which attempts to retrieve credentials
    from the cluster automatically if we're running on a VM, else emits a message containing
    instructions for retrieving credentials.
    """
    if project is None or zone is None or cluster is None:
        raise app.UsageError(f"{project=}, {zone=}, and {cluster=} must all be specified.")

    # Avoid introducing a k8s dependency globally.
    # pylint: disable-next=import-error,import-outside-toplevel
    import kubernetes as k8s  # pytype: disable=import-error

    auth_cache = os.path.expanduser("~/.kube/autogenerated/gke_gcloud_auth_plugin_cache")
    try:
        # Seem to need to remove this each time to avoid:
        # "gke_gcloud_auth_plugin_cache file is empty"
        if os.path.exists(auth_cache):
            logging.info("Removing gcloud auth plugin cache: %s", auth_cache)
            os.remove(auth_cache)
    except FileNotFoundError:
        logging.warning("Failed to remove %s, attempting to ignore.", auth_cache)

    region = zone.rsplit("-", 1)[0]
    try:
        k8s.config.load_kube_config(context=f"gke_{project}_{region}_{cluster}")
    except k8s.config.config_exception.ConfigException as e:
        get_credentials_cmd = (
            f"gcloud container clusters get-credentials {cluster} "
            f"--region {region} "
            f"--project {project}"
        )
        # Automatically generate the kube-config on pod or VM.
        if running_from_k8s() or running_from_vm():
            # Use --internal-ip to access internal cluster endpoint.
            subprocess_run(f"{get_credentials_cmd} --internal-ip", check=True)
            k8s.config.load_kube_config(context=f"gke_{project}_{region}_{cluster}")
        else:
            raise app.UsageError(
                f"Failed to load kube-config for cluster {cluster} with: {e}\n"
                "If it's complaining about a missing kube-config file, run: "
                f"gcloud components install gke-gcloud-auth-plugin; {get_credentials_cmd}.\n"
                "Make sure you also activated the config with the associated kubernetes cluster via"
                f" `{infer_cli_name()} gcp config activate`.\n"
                "Activating the config must be done separately for each kube cluster the first \n"
                "time you want to access it, even if you are accessing it through a command that \n"
                "does not normally require you to activate a config first."
            ) from e


def custom_jobset_kwargs() -> Dict[str, str]:
    """Common kwargs needed for CustomObjectsApi JobSets."""
    return dict(group="jobset.x-k8s.io", version="v1alpha2", plural="jobsets")


JOBSET_LABEL = "jobset.sigs.k8s.io/jobset-name"


def list_k8s_jobsets(*, namespace: str) -> Dict[str, list]:
    """Returns a mapping from jobset name to list of K8s jobs."""

    # Avoid introducing a k8s dependency globally.
    # pylint: disable-next=import-error,import-outside-toplevel
    import kubernetes as k8s  # pytype: disable=import-error

    # List k8s jobs. Note the difference between k8s jobset and k8s job:
    # k8s jobset is 1:1 with bastion job, but k8s jobset is 1:many with k8s job.
    ret = k8s.client.BatchV1Api().list_namespaced_job(namespace, watch=False)

    # Group running k8s jobs by jobset. The jobset name corresponds to the bastion job name.
    k8s_jobsets = defaultdict(list)
    for job in ret.items:
        if job_name := (job.metadata.labels or {}).get(JOBSET_LABEL):
            k8s_jobsets[job_name].append(job)

    return k8s_jobsets


def delete_k8s_jobset(name: str, *, namespace: str):
    """Deletes a K8s jobset by name, including all descendant jobs."""

    # Avoid introducing a k8s dependency globally.
    # pylint: disable-next=import-error,import-outside-toplevel
    import kubernetes as k8s  # pytype: disable=import-error

    try:
        k8s.client.CustomObjectsApi().delete_namespaced_custom_object(
            name=name,
            namespace=namespace,
            propagation_policy="Foreground",
            **custom_jobset_kwargs(),
        )
    except k8s.client.ApiException as e:
        if e.status == 404:
            logging.info("Jobset %s does not exist, no need to delete.", name)
            return
        raise


def delete_k8s_job(name: str, *, namespace: str):
    """Deletes a K8s job by name. If the job is managed by a jobset, the job may be recreated."""

    # Avoid introducing a k8s dependency globally.
    # pylint: disable-next=import-error,import-outside-toplevel
    import kubernetes as k8s  # pytype: disable=import-error

    try:
        # Delete all dependents proactively (i.e., the actual pod).
        k8s.client.BatchV1Api().delete_namespaced_job(
            name, namespace=namespace, propagation_policy="Foreground"
        )
    except k8s.client.ApiException as e:
        if e.status == 404:
            logging.info("%s does not exist, no need to delete.", name)
            return
        raise


def k8s_jobset_table(jobsets: Dict[str, list]) -> Table:
    """Produce a tabular view of the jobsets provided.

    Args:
        jobsets: K8s jobsets, e.g. as returned by `list_k8s_jobsets`.

    Returns:
        A table that can be printed.
    """
    rows = []
    for jobset_name, jobs in sorted(jobsets.items(), key=lambda kv: kv[0]):
        statuses = {}
        for job in jobs:
            statuses.update(
                {
                    status: getattr(job.status, status, 0)
                    for status in ["active", "ready", "failed", "succeeded"]
                }
            )
        rows.append([jobset_name, str(statuses)])
    return Table(headings=["JOBSET", "STATUSES"], rows=rows)


class GCPAPI(str, enum.Enum):
    """GCP API to submit resource requests to."""

    QRM = "QRM"
    GKE = "GKE"
