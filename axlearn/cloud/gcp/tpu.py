# Copyright © 2023 Apple Inc.

"""Utilities to create, delete, and list TPU-VMs."""

import dataclasses
import json
import pathlib
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import cloud_tpu_client
from absl import logging
from google.auth.credentials import Credentials
from googleapiclient import discovery, errors
from googleapiclient.http import HttpRequest

from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import format_table
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.scopes import DEFAULT_TPU_SCOPES
from axlearn.cloud.gcp.storage import list_blobs
from axlearn.cloud.gcp.utils import is_valid_resource_name


class TPUCreationError(RuntimeError):
    """An error with TPU creation."""

    pass


class TPUQuotaLimitError(TPUCreationError):
    """An error with TPU creation related to quotas."""

    pass


class TPUDeletionError(RuntimeError):
    """An error with TPU deletion."""

    pass


def create_tpu(
    name: str,
    *,
    tpu_type: str,
    credentials: Credentials,
    bundler_type: str,
    num_slices: int = 1,
    metadata: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None,
):
    """Create TPU.

    Args:
        name: Name of slice.
        tpu_type: Type of each TPU slice.
        credentials: Credentials to use when interacting with GCP.
        bundler_type: Type of bundle intended to be loaded to VM.
        num_slices: The number of slices of type tpu_type to start.
        metadata: Optional metadata for the instance.
        service_account: Service account to execute the TPU creation.

    Raises:
        TPUCreationError: If an exeption is raised on the creation request.
        ValueError: If an invalid name is provided.
    """
    if not is_valid_resource_name(name):
        raise ValueError(
            f"{name} is not a valid resource name. Please see "
            "https://cloud.google.com/compute/docs/naming-resources#resource-name-format."
        )
    # First try the QRM API, and fall-back to legacy API if we detect a quota-related error.
    try:
        _create_multislice_tpu(
            name,
            tpu_type=tpu_type,
            credentials=credentials,
            bundler_type=bundler_type,
            num_slices=num_slices,
            metadata=metadata,
            service_account=service_account,
        )
    except TPUQuotaLimitError as e:
        if num_slices != 1:
            raise TPUCreationError("Unable to create multislice TPU without QRM quota") from e
        _create_legacy_tpu(
            name,
            tpu_type=tpu_type,
            credentials=credentials,
            bundler_type=bundler_type,
            metadata=metadata,
            service_account=service_account,
        )


def _read_error(e: errors.HttpError) -> Dict[str, Any]:
    """Reads error details from HttpError."""
    data = json.loads(e.content.decode("utf-8"))
    if isinstance(data, list):
        data = data[0]
    return data.get("error", {})


def _execute_create_tpu_request(req: HttpRequest):
    """Wraps request execution with quota error checks.

    If TPU creation fails with quota errors, there's usually no value in retrying the create.
    """
    try:
        resp = req.execute()
        # TODO(markblee): Unclear whether code 3 is sufficient for queued resource quota errors.
        # Ryan Day from google suggested checking the error text as well to be safe.
        if (
            resp.get("code") == 3
            and "reservation for request queue" in resp.get("message", "").lower()
        ):
            raise TPUQuotaLimitError(resp["message"])
    except errors.HttpError as e:
        error = _read_error(e)
        # Code 429 is raised if the user lacks a legacy reservation.
        if error.get("code") == 429:
            raise TPUQuotaLimitError(error["message"]) from e
        raise  # Re-raise original.


def _create_legacy_tpu(
    name: str,
    *,
    tpu_type: str,
    credentials: Credentials,
    bundler_type: str,
    metadata: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None,
):
    """Create TPU (using legacy quota).

    TODO(markblee): Deprecate this when QRM becomes the default.

    See `create_tpu` docstring for details.
    """
    project, zone = gcp_settings("project"), gcp_settings("zone")
    resource = _tpu_resource(credentials)
    tpu_path_prefix = f"projects/{project}/locations/{zone}"
    attempt = 0
    boot_timeout = (
        3600  # If we haven't booted after TPU READY + this many seconds, raise exception.
    )
    reserved_tpu = gcp_settings("reserved_tpu", default=False)
    while True:
        node = get_tpu_node(name, resource)
        if node is not None:  # TPU exists.
            node_act = node["acceleratorType"]
            if node_act != tpu_type:
                raise TPUCreationError(f"TPU {name} exists, but has tpu_type: {node_act}")
            status = get_tpu_node_status(name, node=node)
            if status["state"] == "READY":
                logging.info("TPU %s is READY, checking health.", name)
                # Wait for TPU to become healthy.
                client = cloud_tpu_client.Client(name, project=project, zone=zone)
                while (health := client.health()) != "HEALTHY":
                    logging.info("TPU %s is not HEALTHY (%s), waiting until HEALTHY.", name, health)
                    time.sleep(10)
                logging.info("TPU %s is READY and HEALTHY.", name)
                # Now check for when boot script is complete.
                num_vms = infer_tpu_workers(tpu_type)
                logging.info("Checking for boot status on %d TPU-VM endpoints.", num_vms)
                booted_monitor_start = time.perf_counter()
                while (status := get_tpu_node_status(name, node=node))["num_booted"] < num_vms:
                    if time.perf_counter() - booted_monitor_start >= boot_timeout:
                        raise TPUCreationError(
                            f"Timed out after {boot_timeout}s waiting for {num_vms} to boot."
                        )
                    logging.info("%d/%d are now booted.", status["num_booted"], num_vms)
                    time.sleep(10)
                logging.info("All endpoints READY, HEALTHY and booted for TPU %s", name)
                return
            if status["state"] == "PREEMPTED":
                logging.info("TPU %s is PREEMPTED, will delete and restart.", name)
                _delete_legacy_tpu(name, credentials=credentials)
                continue
            # TPU not ready, wait and check again.
            logging.info("TPU %s showing %s, waiting for READY.", name, status["state"])
            time.sleep(10)
        else:  # TPU does not exist.
            if attempt:
                # Exponential backoff capped at 512s.
                backoff_for = 2 ** min(attempt, 9)
                logging.info(
                    "Attempt %d to create TPU failed, backoff for %ds.", attempt, backoff_for
                )
                time.sleep(backoff_for)
            try:
                attempt += 1
                request_body = _tpu_body(
                    name,
                    tpu_type=tpu_type,
                    bundler_type=bundler_type,
                    metadata=metadata,
                    service_account=service_account,
                )
                # Specify schedulingConfig only for non-QRM requests, as it will break QRM.
                # For QRM, we specify a different field to use reserved quota -- see `_qrm_body`.
                request_body["schedulingConfig"] = {
                    "preemptible": not reserved_tpu,
                    "reserved": reserved_tpu,
                }
                request = resource.create(
                    parent=tpu_path_prefix,
                    nodeId=name,
                    body=request_body,
                )
                _execute_create_tpu_request(request)
            except TPUQuotaLimitError:
                raise  # Re-raise.
            except (errors.HttpError, Exception) as e:
                raise TPUCreationError("Failed to create TPU-VM") from e


def get_tpu_node_status(name: str, *, node: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    """Get the status from the given TPU node.

    For possible states, see:
    https://cloud.google.com/tpu/docs/reference/rest/v2alpha1/projects.locations.nodes#Node.State
    For possible health statuses:
    https://cloud.google.com/tpu/docs/reference/rest/v1/projects.locations.nodes#health

    Args:
        name: Name of the node.
        node: Node as returned by `get_queued_tpu_node`.

    Returns:
        A dict with keys:
        * "state": The current node state.
        * "num_booted": How many VMs have booted.
    """
    state = node["state"]
    num_booted = 0
    if state == "READY":
        # Check for boot script status.
        ready_flags_base_path = (
            f"gs://{gcp_settings('ttl_bucket')}/axlearn/jobs/{name}/tpu_vm_ready_flags"
        )
        ready_flags_path = f"{ready_flags_base_path}/{node['metadata']['create_request_time']}/"
        num_booted = len(list_blobs(ready_flags_path))
    return dict(state=state, num_booted=num_booted)


def delete_tpu(name: str, *, credentials: Credentials, wait: bool = True):
    """Delete TPU.

    Args:
        name: Name of TPU to delete.
        credentials: Credentials to use when interacting with GCP.
        wait: Whether to wait for completion.

    Raises:
        TPUDeletionError: If an exeption is raised on the deletion request.
    """
    # Try QRM quota first, then legacy quota.
    _delete_multislice_tpu(name, credentials=credentials, wait=wait)
    # TODO(markblee, tom_gunter): Remove when we can rely on QRM being the default option.
    _delete_legacy_tpu(name, credentials=credentials, wait=wait)


def _delete_legacy_tpu(name: str, *, credentials: Credentials, wait: bool = True):
    """Delete TPU (using legacy quota).

    See `delete_tpu` docstring for details.
    """
    resource = _tpu_resource(credentials)
    node = get_tpu_node(name, resource)
    if node is None:
        logging.info("TPU %s doesn't exist, no need to delete.", name)
        return
    try:
        logging.info("Deleting TPU %s.", name)
        resource.delete(name=node["name"]).execute()
    except (errors.HttpError, Exception) as e:
        raise TPUDeletionError("Failed to delete TPU-VM") from e

    # Confirm deleted.
    while wait:
        if get_tpu_node(name, resource) is None:
            logging.info("Deleted TPU %s.", name)
            return
        logging.info("Waiting for confirmed deletion of TPU %s", name)
        time.sleep(10)


def list_tpu(credentials: Credentials) -> List[str]:
    """List running TPUs.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        List of running TPU names.
    """
    info = list_tpu_info(credentials)
    return [el.name for el in info]


def _delete_multislice_tpu(name: str, *, credentials: Credentials, wait: bool = True):
    """Delete multislice TPU.

    See `delete_tpu` docstring for details.
    """
    resource = _qrm_resource(credentials)
    node = get_queued_tpu_node(name, resource)
    if node is None:
        logging.info("Multislice TPU %s doesn't exist, no need to delete.", name)
        return

    def _try_delete():
        try:
            logging.info("Deleting multislice TPU %s", name)
            # N.B. force=True ensures that we tear down the associated TPU resources too.
            resource.delete(name=node["name"], force=True).execute()
        except (errors.HttpError, Exception) as e:
            # pytype: disable=attribute-error
            if isinstance(e, errors.HttpError) and e.resp.status == 404:
                # Sometimes the API will return 404 instead. We're already happy in this case.
                return
            # pytype: enable=attribute-error
            raise TPUDeletionError("Failed to delete multislice TPU") from e

    _try_delete()

    # Confirm deleted.
    while wait:
        node = get_queued_tpu_node(name, resource)
        if node is None:
            logging.info("Deleted multislice TPU %s.", name)
            return
        elif node["state"]["state"] in {"FAILED", "SUSPENDED"}:
            # Issue another delete attempt. Due to bugs in GCP API, resources can get stuck in
            # FAILED or SUSPENDED otherwise.
            _try_delete()
        logging.info(
            "Waiting for confirmed deletion of multislice TPU %s (with state %s)",
            name,
            node["state"]["state"],
        )
        time.sleep(10)


def _create_multislice_tpu(
    name: str,
    *,
    tpu_type: str,
    credentials: Credentials,
    bundler_type: str,
    num_slices: int = 1,
    metadata: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None,
):
    """Create multislice TPU.

    See `create_tpu` docstring for details.
    """
    project, zone = gcp_settings("project"), gcp_settings("zone")
    resource = _qrm_resource(credentials)
    attempt = 0
    # If we haven't booted all slices after READY + this many seconds, raise exception.
    boot_timeout = 3600
    while True:
        node = get_queued_tpu_node(name, resource)
        if node is None:
            # Multi-slice doesn't exist.
            if attempt:
                # Exponential backoff capped at 512s.
                backoff_for = 2 ** min(attempt, 9)
                logging.info(
                    "Attempt %d to create TPU failed, backoff for %ds.", attempt, backoff_for
                )
                time.sleep(backoff_for)
            try:
                attempt += 1
                request = resource.create(
                    parent=f"projects/{project}/locations/{zone}",
                    queuedResourceId=name,
                    body=_qrm_body(
                        name,
                        num_slices=num_slices,
                        tpu_body=_tpu_body(
                            name,
                            tpu_type=tpu_type,
                            bundler_type=bundler_type,
                            metadata=metadata,
                            service_account=service_account,
                        ),
                    ),
                )
                _execute_create_tpu_request(request)
            except TPUQuotaLimitError:
                raise  # Re-raise.
            except (errors.HttpError, Exception) as e:
                # Workaround a GCP bug that throws 409 "Resource already exists" even for resources
                # that were just created.
                error = _read_error(e)
                if error.get("code") == 409:
                    logging.warning(
                        "Got 409 even though resource was just created. Attempting to ignore..."
                    )
                    continue
                raise TPUCreationError("Failed to create queued resource") from e
            continue
        # Else multi-slice does exist.
        status = get_queued_tpu_node_status(name, node=node)
        state = status["state"]
        logging.info("Queued resource %s is in state %s.", name, state)
        if state in ["ACCEPTED", "PROVISIONING", "WAITING_FOR_RESOURCES", "CREATING"]:
            # Either trying to find capacity or provisioning capacity.
            time.sleep(60)
        elif state == "ACTIVE":
            logging.info("Slice %s is ACTIVE.", name)
            num_vms = num_slices * infer_tpu_workers(tpu_type)
            booted_monitor_start = time.perf_counter()
            while (status := get_queued_tpu_node_status(name, node=node))["num_booted"] < num_vms:
                if time.perf_counter() - booted_monitor_start >= boot_timeout:
                    raise TPUCreationError(
                        f"Timed out after {boot_timeout}s waiting for {num_vms} to boot."
                    )
                logging.info("%s are now booted.", f"{status['num_booted']} / {num_vms}")
                time.sleep(10)
            logging.info("All endpoints READY, HEALTHY and booted for multislice TPU %s", name)
            return
        elif state in ["FAILED", "SUSPENDING", "SUSPENDED"]:
            logging.info("Deleting multislice TPU.")
            # By default, blocks until deleted.
            _delete_multislice_tpu(name, credentials=credentials)
        elif state != "DELETING":
            # Poll until state change. Easier to track consecutive "unknown" count within this
            # block.
            for _ in range(10):
                logging.warning("Unknown TPU state: %s. Will see if it resolves itself...", state)
                time.sleep(60)
                node = get_queued_tpu_node(name, resource)
                if get_queued_tpu_node_status(name, node=node)["state"] != state:
                    break
            else:
                raise TPUCreationError(f"TPU appears to be stuck in unknown state {state}.")


def get_queued_tpu_node_status(name: str, *, node: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    """Get the status from the given queued TPU node.

    For possible states, see:
    https://cloud.google.com/tpu/docs/reference/rest/v2alpha1/projects.locations.queuedResources#State

    Args:
        name: Name of the node (without multi-slice suffix).
        node: Node as returned by `get_queued_tpu_node`.

    Returns:
        A dict with keys:
        * "state": The current node state.
        * "num_booted": How many VMs have booted.
    """
    state = node["state"]["state"]
    num_booted = 0
    if state == "ACTIVE":
        # Startup script has (in theory) begun running on all nodes in slice.
        # TODO(markblee,tom_gunter): Proper health checks for queued resources.
        ready_flags_base_path = (
            f"gs://{gcp_settings('ttl_bucket')}/axlearn/jobs/{name}/tpu_vm_ready_flags"
        )
        # Only one node spec is permitted (even though it's a list).
        create_request_time = node["tpu"]["nodeSpec"][0]["node"]["metadata"]["create_request_time"]
        ready_flags_path = f"{ready_flags_base_path}/{create_request_time}/"
        num_booted = len(list_blobs(ready_flags_path))
    return dict(state=state, num_booted=num_booted)


@dataclass
class TpuInfo:
    """Information associated with a TPU instance."""

    name: str
    accelerator_type: str
    state: str
    metadata: Dict[str, Any]


def list_tpu_info(credentials: Credentials) -> List[TpuInfo]:
    """Collect info for running TPUs.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        list of TPU info for running TPUs.
    """
    resource = _tpu_resource(credentials)
    result = resource.list(
        parent=f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}"
    ).execute()
    info = [
        TpuInfo(
            name=el.get("name", "").split("/")[-1],
            accelerator_type=el.get("acceleratorType"),
            state=el.get("state"),
            metadata=el.get("metadata", {}),
        )
        for el in result.get("nodes", [])
    ]
    return info


@dataclass
class QueuedResourceInfo(TpuInfo):
    """Information associated with a QueuedResource instance."""

    num_slices: int
    reserved: bool


def list_queued_resource_info(credentials: Credentials) -> List[QueuedResourceInfo]:
    """Collect info for live queued resources.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        List of info for running TPUs.
    """
    resource = _qrm_resource(credentials)
    result = resource.list(
        parent=f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}"
    ).execute()
    info = []
    for el in result.get("queuedResources", {}):
        node_spec = el.get("tpu", {}).get("nodeSpec", {})
        info.append(
            QueuedResourceInfo(
                name=el.get("name", "").split("/")[-1],
                accelerator_type=node_spec[0].get("node", {}).get("acceleratorType"),
                state=el.get("state", {}).get("state"),
                metadata=el.get("metadata", {}),
                num_slices=len(node_spec),
                reserved=el.get("guaranteed", {}).get("reserved", False),
            )
        )
    return info


def _qrm_resource(credentials: Credentials) -> discovery.Resource:
    """Build gcloud TPU v2alpha1 QueuedResource API resource.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        discovery.Resource object for the TPU v2alpha1 QueuedResource API.
    """
    resource = (
        discovery.build("tpu", "v2alpha1", credentials=credentials, cache_discovery=False)
        .projects()
        .locations()
        .queuedResources()
    )
    return resource


def _tpu_resource(credentials: Credentials) -> discovery.Resource:
    """Build gcloud TPU v2alpha1 API resource.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        discovery.Resource object for the TPU v2alpha1 API.
    """
    resource = (
        discovery.build("tpu", "v2alpha1", credentials=credentials, cache_discovery=False)
        .projects()
        .locations()
        .nodes()
    )
    return resource


def _tpu_body(
    name: str,
    *,
    tpu_type: str,
    bundler_type: str,
    metadata: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None,
) -> Dict[str, Any]:
    """Create configuration object for starting a TPU."""

    # TPU-VM configuration.
    dir_path = pathlib.Path(__file__).parent
    startup_script_path = dir_path / "scripts" / "start_tpu.sh"
    with open(startup_script_path, "r", encoding="utf8") as of:
        startup_script_contents = of.read()
    docker_repo = gcp_settings("docker_repo", required=False)

    if tpu_type.startswith("v4") or tpu_type.startswith("v5lite"):
        runtime_version = "tpu-ubuntu2204-base"
    elif tpu_type.startswith("v5p"):
        runtime_version = "v2-alpha-tpuv5"
    else:
        raise ValueError(f"Unknown TPU-VM runtime version for {tpu_type}.")

    body = {}
    if metadata is not None and "enable_ici_resiliency" in metadata:
        # If we don't set this, the platform (GCP) decides whether or not to enable it.
        body["ici_resilience_config"] = {
            # If True, the TPU will continue with degraded ICI throughput in the event of
            # some forms of hardware failure.
            "enable_ici_resiliency": metadata.pop("enable_ici_resiliency"),
        }

    body.update(
        {
            "acceleratorType": tpu_type,
            "metadata": {
                "bundle_bucket": gcp_settings("ttl_bucket"),
                "enable-oslogin": "false",
                "job_name": name,
                "startup-script": startup_script_contents,
                "zone": gcp_settings("zone"),
                "tpu_type": tpu_type,
                "docker_registry": registry_from_repo(docker_repo) if docker_repo else "",
                "bundler_type": bundler_type,
                "create_request_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
                **(metadata or {}),
            },
            "networkConfig": {
                "enableExternalIps": False,
                "network": gcp_settings("network"),
                "subnetwork": gcp_settings("subnetwork"),
            },
            "runtimeVersion": runtime_version,
            "serviceAccount": {
                "email": service_account or gcp_settings("service_account_email"),
                "scope": DEFAULT_TPU_SCOPES,
            },
            "tags": ["allow-internet-egress"],
        }
    )

    return body


def _qrm_body(name: str, *, num_slices: int, tpu_body: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration object for starting a multislice TPU.

    Reference: https://cloud.google.com/tpu/docs/queued-resources
    """
    node_spec = {
        "parent": f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}",
        "node": tpu_body,
    }
    if num_slices == 1:
        # Required iff a single node.
        node_spec["node_id"] = name
    else:
        # Required iff multiple nodes.
        node_spec["multi_node_params"] = {
            "node_count": num_slices,
            "node_id_prefix": name,
        }
    body = {
        "guaranteed": {"reserved": gcp_settings("reserved_tpu")},
        "tpu": {
            "node_spec": [node_spec],  # List, but only a single node spec is supported right now.
        },
    }
    return body


def get_tpu_node(name: str, resource: discovery.Resource) -> Optional[Dict[str, Any]]:
    """Gets information about a TPU node.

    Args:
        name: Name of TPU node.
        resource: discovery.Resource object. See also `_tpu_resource`.

    Returns:
        The node with the given name, or None if it doesn't exist.
    """
    # Get node if one exists.
    try:
        return resource.get(
            name=f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}/nodes/{name}"
        ).execute()
    except errors.HttpError as e:
        if e.resp.status == 404:
            return None
        raise  # Re-raise.


def get_queued_tpu_node(name: str, resource: discovery.Resource) -> Optional[Dict[str, Any]]:
    """Gets information about a QueuedResource.

    Args:
        name: Name of QueuedResource.
        resource: discovery.Resource object. See also `_qrm_resource`.

    Returns:
        The QueuedResource with the given name, or None if it doesn't exist.
    """
    # Get node if one exists.
    while True:
        try:
            return resource.get(
                # pylint: disable-next=line-too-long
                name=f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}/queuedResources/{name}"
            ).execute()
        except errors.HttpError as e:
            if e.resp.status == 404:
                return None
            logging.info("HttpError getting queued resource, retrying after backoff: %s", e)
            time.sleep(10)
            continue


_TPU_VERSIONS = ("v3", "v4", "v5litepod", "v5p")


def infer_tpu_version(tpu_type: str) -> str:
    """Infer TPU version from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred TPU version string.

    Raises:
        ValueError: if the TPU version string is unknown.
    """
    tpu_version = tpu_type.rsplit("-", 1)[0]  # split from the last occurance of '-'
    if tpu_version not in _TPU_VERSIONS:
        raise ValueError(f"Unknown TPU version {tpu_version}. Expected one of {_TPU_VERSIONS}")
    return tpu_version


def infer_tpu_cores(tpu_type: str) -> int:
    """Infer the number of TPU cores from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred number of TPU cores.
    """
    return int(tpu_type.rsplit("-", 1)[1])


def infer_tpu_workers(tpu_type: str) -> int:
    """Infer the number of worker processes for the given TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred number of TPU workers.
    """
    tpu_pattern = r"(.+)*-(\d+)"
    match = re.search(tpu_pattern, tpu_type)
    try:
        if match is not None:
            tpu_version, tpu_cores = match.groups()
            if tpu_version in {"v3", "v4", "v5p"}:
                return int(tpu_cores) // 8
            if tpu_version in {"v5litepod"}:
                return int(tpu_cores) // 4
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Failed to parse tpu_type %s: %s", tpu_type, e)
    raise NotImplementedError(tpu_type)


# TODO(markblee): Dedup with other format_* utils.
def format_tpu_info(tpus: List[TpuInfo], metadata: Optional[Sequence[str]] = None) -> str:
    """Produce a tabular string view of the TPU infos provided.

    Args:
        tpus: Information on TPUs, e.g. as returned by `list_tpu_info`.
        metadata: An optional sequence of metadata fields to include in the result. By default,
            metadata fields are dropped, since the output can become noisy (e.g. with startup
            scripts).

    Returns:
        A table-formatted string.
    """
    headings = [field.name for field in dataclasses.fields(TpuInfo)]
    if metadata:
        # Filter requested metadata fields.
        tpus = [
            dataclasses.replace(tpu, metadata={k: tpu.metadata.get(k, None) for k in metadata})
            for tpu in tpus
        ]
    else:
        headings.remove("metadata")
    return format_table(
        headings=headings,
        rows=[
            [str(v) for k, v in info.__dict__.items() if k in headings]
            for info in sorted(tpus, key=lambda x: x.name)
        ],
    )


# TODO(markblee): Dedup with other format_* utils.
def format_queued_resource_info(
    queued_resources: List[QueuedResourceInfo], metadata: Optional[Sequence[str]] = None
) -> str:
    """Produce a tabular string view of the queued resource infos provided.

    Args:
        queued_resources: Information on queued resources, e.g. as returned by
            `list_queued_resource_info`.
        metadata: An optional sequence of metadata fields to include in the result. By default,
            metadata fields are dropped, since the output can become noisy (e.g. with startup
            scripts).

    Returns:
        A table-formatted string.
    """
    headings = [field.name for field in dataclasses.fields(QueuedResourceInfo)]
    if metadata:
        # Filter requested metadata fields.
        queued_resources = [
            dataclasses.replace(
                resource, metadata={k: resource.metadata.get(k, None) for k in metadata}
            )
            for resource in queued_resources
        ]
    else:
        headings.remove("metadata")
    return format_table(
        headings=headings,
        rows=[
            [str(v) for k, v in info.__dict__.items() if k in headings]
            for info in sorted(queued_resources, key=lambda x: x.name)
        ],
    )
