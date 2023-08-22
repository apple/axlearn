# Copyright © 2023 Apple Inc.

"""Utilities to create, delete, and list VMs."""

import dataclasses
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from absl import logging
from googleapiclient import discovery, errors
from oauth2client.client import GoogleCredentials

from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import format_table
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import infer_cli_name, is_valid_resource_name


class VMCreationError(RuntimeError):
    """An error with VM creation."""

    pass


class VMDeletionError(RuntimeError):
    """An error with VM deletion."""

    pass


# pylint: disable-next=too-many-branches
def create_vm(
    name: str,
    *,
    vm_type: str,
    disk_size: int,
    credentials: GoogleCredentials,
    bundler_type: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """Create VM.

    Args:
        name: Name of VM.
        vm_type: What gcloud machine type to boot.
        disk_size: Size of disk to provision (in GB).
        credentials: Credentials to use when interacting with GCP.
        bundler_type: Type of bundle intended to be loaded to VM.
        metadata: Optional metadata for the instance.

    Raises:
        VMCreationError: If an exeption is raised on the creation request.
        ValueError: If an invalid name is provided.
    """
    if not is_valid_resource_name(name):
        raise ValueError(f"{name} is not a valid resource name.")
    resource = _compute_resource(credentials)
    attempt = 0
    while True:
        node = _get_vm_node(name, resource)
        if node is None:  # VM doesn't exist.
            if attempt:
                # Exponential backoff capped at 512s.
                backoff_for = 2 ** min(attempt, 9)
                logging.info(
                    "Attempt %d to create VM failed, backoff for %ds. "
                    "Check https://console.cloud.google.com/home/activity?project=%s for errors.",
                    attempt,
                    backoff_for,
                    gcp_settings("project"),
                )
                time.sleep(backoff_for)
            try:
                images = list_disk_images(credentials)
                if not images:
                    raise VMCreationError("Could not find valid disk image.")
                resource.instances().insert(
                    project=gcp_settings("project"),
                    zone=gcp_settings("zone"),
                    body=_vm_config(
                        name,
                        vm_type=vm_type,
                        disk_size=disk_size,
                        disk_image=images[0],
                        bundler_type=bundler_type,
                        metadata=metadata,
                    ),
                ).execute()
                attempt += 1
            except (errors.HttpError, Exception) as e:
                raise VMCreationError("Couldn't create VM") from e
        else:  # VM exists.
            status = node["status"]
            if status != "RUNNING":
                logging.info("VM %s showing %s, waiting for RUNNING.", name, status)
                time.sleep(10)
                continue
            if "labels" in node:
                # Boot script may have completed.
                boot_status = node["labels"].get("boot_status")
                if boot_status == "done":
                    logging.info("VM %s is running and booted.", name)
                    logging.info("SSH to VM with: %s gcp sshvm %s", infer_cli_name(), name)
                    return
                logging.info(
                    "VM %s RUNNING, waiting for boot to complete "
                    "(which usually takes a few minutes): %s",
                    name,
                    boot_status,
                )
                time.sleep(10)


def delete_vm(name: str, *, credentials: GoogleCredentials):
    """Delete VM.

    Args:
        name: Name of VM to delete.
        credentials: Credentials to use when interacting with GCP.

    Raises:
        VMDeletionError: If an exeption is raised on the deletion request.
    """
    resource = _compute_resource(credentials)
    node = _get_vm_node(name, resource)
    if node is None:  # VM doesn't exist.
        logging.info("VM %s doesn't exist.", name)
        return
    try:
        response = (
            resource.instances()
            .delete(project=gcp_settings("project"), zone=gcp_settings("zone"), instance=name)
            .execute()
        )
        while True:
            logging.info("Waiting for deletion of VM %s to complete.", name)
            zone_op = (
                resource.zoneOperations()
                .get(
                    project=gcp_settings("project"),
                    zone=gcp_settings("zone"),
                    operation=response["name"],
                )
                .execute()
            )
            if zone_op.get("status") == "DONE":
                if "error" in zone_op:
                    raise VMDeletionError(zone_op["error"])
                logging.info("Deletion of VM %s is complete.", name)
                return
            time.sleep(10)
    except (errors.HttpError, Exception) as e:
        raise VMDeletionError("Failed to delete VM") from e


@dataclass
class VmInfo:
    """Information associated with a VM instance."""

    name: str
    metadata: Dict[str, Any]


def list_vm_info(credentials: GoogleCredentials) -> List[VmInfo]:
    """List running VMs for the given project and zone.

    Args:
        credentials: gcloud credentials used by googleapiclient.discovery.

    Returns:
        list of up-VMs.
    """
    resource = _compute_resource(credentials)
    result = (
        resource.instances()
        .list(project=gcp_settings("project"), zone=gcp_settings("zone"))
        .execute()
    )
    results = []
    for vm in result.get("items", []):
        results.append(
            VmInfo(
                name=vm["name"],
                metadata={item["key"]: item["value"] for item in vm["metadata"]["items"]},
            )
        )
    return results


def _compute_resource(credentials: GoogleCredentials) -> discovery.Resource:
    """Build gcloud compute v1 API resource.

    Args:
        credentials: gcloud credentials used by googleapiclient.discovery.

    Returns:
       discovery.Resource object for the compute v1 API.
    """
    return discovery.build("compute", "v1", credentials=credentials, cache_discovery=False)


def _vm_config(
    name: str,
    *,
    vm_type: str,
    disk_size: int,
    disk_image: str,
    bundler_type: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Produce VM config for <name>.

    Args:
        name: Name of VM.
        vm_type: VM type.
        disk_size: Size of disk in GB.
        disk_image: Name of disk image to load into VM.
        bundler_type: Type of bundle intended to be loaded to VM.
        metadata: Optional metadata for the instance.

    Returns:
        Dictionary describing the VM configuration.
    """
    dir_path = pathlib.Path(__file__).parent
    startup_script_path = dir_path / "scripts" / "start_vm.sh"
    with open(startup_script_path, "r", encoding="utf8") as of:
        startup_script_contents = of.read()
    docker_repo = gcp_settings("docker_repo", required=False)
    metadata = {
        "items": [
            {"key": "bundle_bucket", "value": gcp_settings("ttl_bucket")},
            {"key": "enable-oslogin", "value": "false"},
            {"key": "taskname", "value": name},
            {"key": "startup-script", "value": startup_script_contents},
            {"key": "zone", "value": gcp_settings("zone")},
            {
                "key": "docker_registry",
                "value": registry_from_repo(docker_repo) if docker_repo else "",
            },
            {"key": "bundler_type", "value": bundler_type},
        ]
        + [{"key": k, "value": v} for k, v in (metadata or {}).items()]
    }
    config = {
        "canIpForward": False,
        "confidentialInstanceConfig": {"enableConfidentialCompute": False},
        "deletionProtection": False,
        "description": "AXLearn VM",
        "disks": [
            {
                "autoDelete": True,
                "boot": True,
                "deviceName": "data",
                "initializeParams": {
                    "diskSizeGb": str(disk_size),
                    "diskType": (
                        f"projects/{gcp_settings('project')}/zones/{gcp_settings('zone')}/"
                        "diskTypes/pd-standard"
                    ),
                    "labels": {},
                    "sourceImage": disk_image,
                },
                "mode": "READ_WRITE",
                "type": "PERSISTENT",
            }
        ],
        "displayDevice": {"enableDisplay": False},
        "guestAccelerators": [],
        "labels": {},
        "machineType": (
            f"projects/{gcp_settings('project')}/zones/{gcp_settings('zone')}/"
            f"machineTypes/{vm_type}"
        ),
        "metadata": metadata,
        "name": name,
        "networkInterfaces": [
            {
                "subnetwork": gcp_settings("subnetwork"),
            }
        ],
        "reservationAffinity": {"consumeReservationType": "ANY_RESERVATION"},
        "scheduling": {
            "automaticRestart": True,
            "onHostMaintenance": "MIGRATE",
            "preemptible": False,
        },
        "serviceAccounts": [
            {
                "email": gcp_settings("service_account_email"),
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ],
        "shieldedInstanceConfig": {
            "enableIntegrityMonitoring": True,
            "enableSecureBoot": True,
            "enableVtpm": True,
        },
        "tags": {"items": ["allow-internet-egress"]},
        "zone": f"projects/{gcp_settings('project')}/zones/{gcp_settings('zone')}",
    }
    return config


def _get_vm_node(name: str, resource: discovery.Resource) -> Optional[Dict[str, Any]]:
    """Gets information about a VM node.

    Args:
        name: Name of VM.
        resource: discovery.Resource object. See also `_compute_resource`.

    Returns:
        The VM with the given name, or None if it doesn't exist.
    """
    up_nodes = (
        resource.instances()
        .list(
            project=gcp_settings("project"),
            zone=gcp_settings("zone"),
        )
        .execute()
    )
    nodes = [el for el in up_nodes.get("items", {}) if el.get("name", "").split("/")[-1] == name]
    return None if not nodes else nodes.pop()


def list_disk_images(creds: GoogleCredentials) -> List[str]:
    """Lists available disk images in the configured `image_project`.

    Args:
        creds: Gcloud credentials.

    Returns:
        The list of images.
    """
    resource = _compute_resource(creds)
    image_project = gcp_settings("image_project")
    images = (
        resource.images()
        .list(
            project=image_project,
            orderBy="creationTimestamp desc",
            maxResults=50,
        )
        .execute()
    )
    image_names = [el["name"] for el in images["items"] if "ubuntu-2004" in el["name"]]
    return [f"projects/{image_project}/global/images/{name}" for name in image_names]


def format_vm_info(vms: List[VmInfo], metadata: Optional[Sequence[str]] = None) -> str:
    """Formats vm information into a table.

    Args:
        vms: A list of VmInfos, e.g. as returned by `list_vm_info`.
        metadata: An optional sequence of metadata fields to include in the result. By default,
            metadata fields are dropped, since the output can become noisy (e.g. with startup
            scripts).

    Returns:
        A string produced by `format_table`, which can be printed.
    """
    headings = [field.name for field in dataclasses.fields(VmInfo)]
    if metadata:
        # Filter requested metadata fields.
        vms = [
            dataclasses.replace(vm, metadata={k: vm.metadata.get(k, None) for k in metadata})
            for vm in vms
        ]
    else:
        headings.remove("metadata")
    return format_table(
        headings=headings,
        rows=[
            [str(v) for k, v in info.__dict__.items() if k in headings]
            for info in sorted(vms, key=lambda x: x.name)
        ],
    )
