# Copyright © 2023 Apple Inc.

"""Utilities to create, delete, and list TPU-VMs."""

import dataclasses
import json
import pathlib
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import cloud_tpu_client
from absl import logging
from googleapiclient import discovery, errors
from googleapiclient.http import HttpRequest
from oauth2client.client import GoogleCredentials

from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import format_table
from axlearn.cloud.gcp.config import gcp_settings
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
    credentials: GoogleCredentials,
    bundler_type: str,
    num_slices: int = 1,
    metadata: Optional[Dict[str, str]] = None,
):
    """Create TPU.

    Args:
        name: Name of slice.
        tpu_type: Type of each TPU slice.
        credentials: Credentials to use when interacting with GCP.
        bundler_type: Type of bundle intended to be loaded to VM.
        num_slices: The number of slices of type tpu_type to start.
        metadata: Optional metadata for the instance.

    Raises:
        TPUCreationError: If an exeption is raised on the creation request.
        ValueError: If an invalid name is provided.
    """
    if not is_valid_resource_name(name):
        raise ValueError(f"{name} is not a valid resource name.")
    # First try the QRM API, and fall-back to legacy API if we detect a quota-related error.
    try:
        _create_multislice_tpu(
            name,
            tpu_type=tpu_type,
            credentials=credentials,
            bundler_type=bundler_type,
            num_slices=num_slices,
            metadata=metadata,
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
        )


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
        data = json.loads(e.content.decode("utf-8"))
        if isinstance(data, list):
            data = data[0]
        # Code 429 is raised if the user lacks a legacy reservation.
        if data.get("error", {}).get("code") == 429:
            raise TPUQuotaLimitError(data["error"]["message"]) from e
        raise  # Re-raise original.


def _create_legacy_tpu(
    name: str,
    *,
    tpu_type: str,
    credentials: GoogleCredentials,
    bundler_type: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """Create TPU (using legacy quota).

    TODO(markblee): Deprecate this when QRM becomes the default.

    See `create_tpu` docstring for details.
    """
    resource = _tpu_resource(credentials)
    tpu_path_prefix = f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}"
    attempt = 0
    boot_timeout = (
        3600  # If we haven't booted after TPU READY + this many seconds, raise exception.
    )
    while True:
        node = get_tpu_node(name, resource)
        if node is not None:  # TPU exists.
            node_act = node["acceleratorType"]
            if node_act != tpu_type:
                raise TPUCreationError(f"TPU {name} exists, but has tpu_type: {node_act}")
            state = node["state"]
            if state == "READY":
                logging.info("TPU %s is READY, checking health.", name)
                client = cloud_tpu_client.Client(
                    name, project=gcp_settings("project"), zone=gcp_settings("zone")
                )
                while client.health() == "UNHEALTHY_MAINTENANCE":
                    logging.info("TPU %s is not HEALTHY, waiting until HEALTHY.", name)
                    time.sleep(10)
                logging.info("TPU %s is READY and HEALTHY.", name)
                # Now check for when boot script is complete:
                total_vms = infer_tpu_workers(tpu_type)
                ready_flags_base_path = (
                    f"gs://{gcp_settings('ttl_bucket')}/axlearn/tasks/{name}/tpu_vm_ready_flags"
                )
                node = resource.get(name=f"{tpu_path_prefix}/nodes/{name}").execute()
                ready_flags_path = (
                    f"{ready_flags_base_path}/{node['metadata']['create_request_time']}/"
                )
                logging.info(
                    "Checking for boot status on %d TPU-VM endpoints at %s.",
                    total_vms,
                    ready_flags_path,
                )
                booted_monitor_start = time.perf_counter()
                while len(list_blobs(ready_flags_path)) < total_vms:
                    if time.perf_counter() - booted_monitor_start >= boot_timeout:
                        raise TPUCreationError(
                            f"Timed out after {boot_timeout}s waiting for {total_vms} to boot."
                        )
                    logging.info(
                        "%d/%d are now booted.", len(list_blobs(ready_flags_path)), total_vms
                    )
                    time.sleep(10)
                logging.info("All endpoints READY, HEALTHY and booted for TPU %s", name)
                return
            if state == "PREEMPTED":
                logging.info("TPU %s is PREEMPTED, will delete and restart.", name)
                _delete_legacy_tpu(name, credentials=credentials)
                continue
            # TPU not ready, wait and check again.
            logging.info("TPU %s showing %s, waiting for READY.", name, state)
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
                request = resource.create(
                    parent=tpu_path_prefix,
                    nodeId=name,
                    body=_tpu_body(
                        name, tpu_type=tpu_type, bundler_type=bundler_type, metadata=metadata
                    ),
                )
                _execute_create_tpu_request(request)
            except TPUQuotaLimitError:
                raise  # Re-raise.
            except (errors.HttpError, Exception) as e:
                raise TPUCreationError("Failed to create TPU-VM") from e


def delete_tpu(name: str, *, credentials: GoogleCredentials, wait: bool = True):
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


def _delete_legacy_tpu(name: str, *, credentials: GoogleCredentials, wait: bool = True):
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


def list_tpu(credentials: GoogleCredentials) -> List[str]:
    """List running TPUs.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        List of running TPU names.
    """
    info = list_tpu_info(credentials)
    return [el.name for el in info]


def _delete_multislice_tpu(name: str, *, credentials: GoogleCredentials, wait: bool = True):
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
    credentials: GoogleCredentials,
    bundler_type: str,
    num_slices: int = 1,
    metadata: Optional[Dict[str, str]] = None,
):
    """Create multislice TPU.

    See `create_tpu` docstring for details.
    """
    resource = _qrm_resource(credentials)
    attempt = 0
    boot_timeout = (
        3600  # If we haven't booted all slices after READY + this many seconds, raise exception.
    )
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
                    parent=f"projects/{gcp_settings('project')}/locations/{gcp_settings('zone')}",
                    queuedResourceId=name,
                    body=_qrm_body(
                        name,
                        num_slices=num_slices,
                        tpu_body=_tpu_body(
                            name, tpu_type=tpu_type, bundler_type=bundler_type, metadata=metadata
                        ),
                    ),
                )
                _execute_create_tpu_request(request)
            except TPUQuotaLimitError:
                raise  # Re-raise.
            except (errors.HttpError, Exception) as e:
                raise TPUCreationError("Failed to create queued resource") from e
            continue
        # Else multi-slice does exist.
        state = node["state"]["state"]
        logging.info("Queued resource %s is in state %s.", name, state)
        if state in ["ACCEPTED", "PROVISIONING"]:
            # Either trying to find capacity or provisioning capacity.
            time.sleep(60)
        elif state == "ACTIVE":
            # Startup script has (in theory) begun running on all nodes in slice.
            logging.info("Slice %s is ACTIVE.", name)
            # TODO(markblee,tom_gunter): Proper health checks for queued resources.
            num_vms = num_slices * infer_tpu_workers(tpu_type)
            ready_flags_base_path = (
                f"gs://{gcp_settings('ttl_bucket')}/axlearn/tasks/{name}/tpu_vm_ready_flags"
            )
            # Only one node spec is permitted (even though it's a list).
            create_request_time = node["tpu"]["nodeSpec"][0]["node"]["metadata"][
                "create_request_time"
            ]
            ready_flags_path = f"{ready_flags_base_path}/{create_request_time}/"
            logging.info(
                "Checking for boot status on %d TPU-VM endpoints at %s.",
                num_vms,
                ready_flags_path,
            )
            booted_monitor_start = time.perf_counter()
            while len(list_blobs(ready_flags_path)) < num_vms:
                if time.perf_counter() - booted_monitor_start >= boot_timeout:
                    raise TPUCreationError(
                        f"Timed out after {boot_timeout}s waiting for {num_vms} to boot."
                    )
                logging.info(
                    "%s are now booted.", f"{len(list_blobs(ready_flags_path))} / {num_vms}"
                )
                time.sleep(10)
            logging.info("All endpoints READY, HEALTHY and booted for multislice TPU %s", name)
            return
        elif state in ["FAILED", "SUSPENDING", "SUSPENDED"]:
            logging.info("Deleting multislice TPU.")
            _delete_multislice_tpu(name, credentials=credentials)
        elif state != "DELETING":
            raise TPUCreationError(f"Unknown TPU state value: {state}")


@dataclass
class TpuInfo:
    """Information associated with a TPU instance."""

    name: str
    accelerator_type: str
    state: str
    metadata: Dict[str, Any]


def list_tpu_info(credentials: GoogleCredentials) -> List[TpuInfo]:
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


def list_queued_resource_info(credentials: GoogleCredentials) -> List[QueuedResourceInfo]:
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


def _qrm_resource(credentials: GoogleCredentials) -> discovery.Resource:
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


def _tpu_resource(credentials: GoogleCredentials) -> discovery.Resource:
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
    name: str, *, tpu_type: str, bundler_type: str, metadata: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Create configuration object for starting a TPU."""

    # TPU-VM configuration.
    dir_path = pathlib.Path(__file__).parent
    startup_script_path = dir_path / "scripts" / "start_tpu.sh"
    with open(startup_script_path, "r", encoding="utf8") as of:
        startup_script_contents = of.read()
    docker_repo = gcp_settings("docker_repo", required=False)

    if tpu_type.startswith("v4"):
        runtime_version = "tpu-vm-v4-base"
    elif tpu_type.startswith("v5lite"):
        runtime_version = "v2-alpha-tpuv5-lite"
    else:
        # TODO(markblee): Verify whether this is still applicable on tf>2.8.0.
        runtime_version = "tpu-vm-tf-2.8.0"

    body = {
        "acceleratorType": tpu_type,
        "metadata": {
            "bundle_bucket": gcp_settings("ttl_bucket"),
            "enable-oslogin": "false",
            "taskname": name,
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
            "email": gcp_settings("service_account_email"),
            "scope": ["https://www.googleapis.com/auth/cloud-platform"],
        },
        "tags": ["allow-internet-egress"],
    }
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


def infer_tpu_version(tpu_type: str) -> str:
    """Infer TPU version from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred TPU version string.
    """
    return tpu_type.rsplit("-", 1)[0]  # split from the last occurance of '-'


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
            if tpu_version in {"v3", "v4"}:
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
