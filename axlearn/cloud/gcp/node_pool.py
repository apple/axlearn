# Copyright © 2024 Apple Inc.

"""Node pool utilities"""
import enum
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from absl import flags, logging
from google.auth.credentials import Credentials
from googleapiclient import discovery, errors

from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import get_credentials

FLAGS = flags.FLAGS

# This label should be added to the node pools. It can be used to identify the node pools
# created by the same pre-provisioner.
PRE_PROVISIONER_LABEL = "pre-provisioner-id"


class NodePoolCreationError(RuntimeError):
    """An error with NodePool creation."""

    pass


class NodePoolValidationError(RuntimeError):
    """An error with NodePool body validation."""

    pass


class NodePoolDeletionError(RuntimeError):
    """An error with NodePool deletion."""

    pass


class NodePoolStatus(enum.Enum):
    """Node Pool Status.

    See also:
    https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1beta1/projects.locations.clusters.nodePools#NodePool.Status

    Attributes:
        STATUS_UNSPECIFIED: Status not set.
        PROVISIONING: The node pool is being created.
        RUNNING: The node pool has been created and is fully usable.
        RUNNING_WITH_ERROR: The node pool has been created and is partially usable.
        RECONCILING: Some work is actively being done on the node pool, such as software upgrade.
        STOPPING: The node pool is being deleted.
        ERROR: The node pool may be unusable.

        NOT_EXIST: Node pool does not exist. Not part of official status. Added for convenience.
        UNKNOWN: Unknown status. Not part of official status. Added for forwards compatibility.

    """

    STATUS_UNSPECIFIED = "STATUS_UNSPECIFIED"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    RUNNING_WITH_ERROR = "RUNNING_WITH_ERROR"
    RECONCILING = "RECONCILING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"
    NOT_EXIST = "NOT_EXIST"
    UNKNOWN = "UNKNOWN"


def _node_pool_body(
    *,
    name: str,
    pre_provisioner_id: Optional[str] = None,
    zone: str,
    num_nodes: int,
    machine_type: str,
    topology: Optional[str] = None,
    use_spot_vm: Optional[bool] = None,
    reservation: Optional[str] = None,
    location_hint: Optional[str] = None,
    enable_tpu_ici_resiliency: Optional[bool] = None,
    disk_size: int = 100,
    service_account_email: Optional[str] = None,
    additional_labels: Optional[dict[str, str]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Node pool request body.

    See also:
    https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#create

    Args:
        name: Name of the node pool.
        pre_provisioner_id: ID of the pre-provisioner.
            It will be added to the node pool labels and can be used to
            identify the node pools created by the same pre-provisioner.
        zone: GCP zone.
        num_nodes: Number of nodes in the node pool.
        machine_type: What gcloud machine type to boot.
        topology: TPU topology.
        use_spot_vm: Whether to use spot VMs. It is mutually exclusive with reservation.
        reservation: Name of TPU reservation. It is mutually exclusive with use_spot_vm.
        location_hint: Location of TPU reservation.
        enable_tpu_ici_resiliency: Whether to enable TPU ICI resiliency.
        disk_size: Size of disk to provision (in GB).
        service_account_email: Optional service account email to use.
        additional_labels: Additional labels to add.
        metadata: Optional metadata for the instance.

    Returns:
        The request body for v1 node pool API call.

    Raises:
        NodePoolCreationError: When use_spot_vm is True and reservation is not None.
    """

    if use_spot_vm and reservation is not None:
        raise NodePoolValidationError("use_spot_vm and reservation are mutually exclusive.")

    reservation_conf = {}

    if use_spot_vm is not None:
        reservation_conf = {"spot": use_spot_vm}

    if reservation is not None:
        reservation_conf.update(
            {
                "reservationAffinity": {
                    "consumeReservationType": "SPECIFIC_RESERVATION",
                    "key": "compute.googleapis.com/reservation-name",
                    "values": [reservation],
                }
            }
        )

    if service_account_email is None:
        service_account_email = gcp_settings("service_account_email")

    placement_policy = {}
    if topology is not None:
        placement_policy = {
            "placementPolicy": {
                "tpuTopology": topology,
                "type": "COMPACT",
            }
        }

    labels = {}

    if pre_provisioner_id is not None:
        labels.update({PRE_PROVISIONER_LABEL: pre_provisioner_id})

    if location_hint is not None:
        labels.update({"cloud.google.com/gke-location-hint": str(location_hint).lower()})

    if enable_tpu_ici_resiliency is not None:
        labels.update(
            {"cloud.google.com/gke-tpu-ici-resiliency": str(enable_tpu_ici_resiliency).lower()}
        )

    if additional_labels is not None:
        labels.update(additional_labels)

    return {
        "nodePool": {
            "name": name,
            "locations": [zone],
            "initialNodeCount": num_nodes,
            "autoscaling": {
                "autoprovisioned": False,
                "enabled": False,
            },
            "management": {
                "autoRepair": True,
                # Disable node auto upgrade to avoid interruption to workloads. AutoUpgrade can
                # only be disabled when GKE cluster is not enrolled in a Release channel.
                # Otherwise node pool creation will fail.
                # https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-upgrades#disable
                "autoUpgrade": False,
            },
            "queuedProvisioning": {
                "enabled": False,  # Not QRM specific.
            },
            "config": {
                "diskSizeGb": disk_size,
                "labels": labels,
                "machineType": machine_type,
                "metadata": metadata or {},
                "preemptible": False,
                "serviceAccount": service_account_email,
                "shieldedInstanceConfig": {
                    "enableIntegrityMonitoring": True,
                    "enableSecureBoot": True,
                },
                "tags": ["allow-internet-egress"],
                "taints": [
                    {
                        "effect": "NO_SCHEDULE",
                        "key": "google.com/tpu",
                        "value": "present",
                    },
                ],
                **reservation_conf,
            },
            **placement_policy,
        },
    }


def _node_pool_resource(credentials: Credentials) -> discovery.Resource:
    """Builds gcloud v1 node pool API resource.

    Args:
        credentials: Gcloud credentials used by googleapiclient.discovery.

    Returns:
        discovery.Resource object for the v1 node pool API.
    """
    return (
        discovery.build("container", "v1", credentials=credentials, cache_discovery=False)
        .projects()
        .locations()
        .clusters()
        .nodePools()
    )


def _create_node_pool(resource: discovery.Resource, *, parent: str, body: Any):
    # https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#create
    return resource.create(parent=parent, body=body).execute()


def _get_node_pool(resource: discovery.Resource, *, name: str):
    # https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#get
    return resource.get(name=name).execute()


def _delete_node_pool(resource: discovery.Resource, *, name: str):
    # https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#delete
    return resource.delete(name=name).execute()


def _list_node_pools(resource: discovery.Resource, *, parent: str):
    # https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#list
    return resource.list(parent=parent).execute()


def _node_pool_parent(*, project: str, zone: str, cluster: str) -> str:
    region = zone.rsplit("-", 1)[0]
    return f"projects/{project}/locations/{region}/clusters/{cluster}"


def _node_pool_canonical_id(*, project: str, zone: str, cluster: str, name: str) -> str:
    return f"{_node_pool_parent(project=project, zone=zone, cluster=cluster)}/nodePools/{name}"


def delete_node_pool(
    name: str,
    *,
    project: str,
    zone: str,
    cluster: str,
    fire_and_forget: bool = False,
) -> Any:
    """Delete a node pool.

    See also:
    https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#delete

    Args:
        name: Node pool name.
        project: GCP Project name.
        zone: GCP zone name.
        cluster: K8s cluster.
        fire_and_forget: If True, execute deletion and return immediately regardless of failures.
            If False, raise an exception if node pool deletion fails.

    Returns:
        Response of the v1 node pool deletion API call.

    Raises:
        Exception: If node pool deletion fails and fire_and_forget is False.
    """

    node_pool_cid = _node_pool_canonical_id(project=project, zone=zone, cluster=cluster, name=name)

    try:
        resource = _node_pool_resource(get_credentials())
        return _delete_node_pool(resource, name=node_pool_cid)
    except Exception as e:  # pylint: disable=broad-except
        if not fire_and_forget:
            raise e
        logging.warning(
            "Failed to delete node pool %s; Exception %s. Ignoring due to fire_and_forget",
            node_pool_cid,
            e,
        )


def delete_node_pools(
    names: list[str],
    *,
    project: str,
    zone: str,
    cluster: str,
    retry_interval: int = 30,
    wait_timeout: int = 0,
):
    """Delete node pools.

    See also:
    https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#delete

    Args:
        names: List of node pools to delete.
        project: GCP Project name.
        zone: GCP zone name.
        cluster: K8s cluster.
        retry_interval: Time in seconds between retries.
        wait_timeout: Seconds to wait for node pools to delete.
            This function doesn't wait if wait_timeout <= 0.

    Raises:
        NodePoolDeletionError: If node pools are not deleted
            within wait_timeout if wait_timeout > 0.
    """

    start_time = time.perf_counter()
    while True:
        # Wait until all node pools are deleted, or timeout.
        num_node_pools_deleted = 0
        for node_pool_name in names:
            node_pool_status = get_node_pool_status(
                node_pool_name,
                project=project,
                zone=zone,
                cluster=cluster,
            )
            if node_pool_status == NodePoolStatus.NOT_EXIST:
                logging.info("Node pool %s is deleted", node_pool_name)
                num_node_pools_deleted += 1
            else:
                logging.info("Deleting node pool %s", node_pool_name)

                res = delete_node_pool(
                    node_pool_name,
                    project=project,
                    zone=zone,
                    cluster=cluster,
                    fire_and_forget=True,
                )
                logging.debug("Node pool deletion response: %s", res)
                logging.warning("Node pool %s is %s", node_pool_name, node_pool_status)

        elapsed_time = time.perf_counter() - start_time

        if num_node_pools_deleted == len(names):
            logging.info(
                "All %s node pools deletion took %s seconds", num_node_pools_deleted, elapsed_time
            )
            break

        if wait_timeout <= 0:
            logging.info(
                "Skip waiting for node pool deletion " "since wait_timeout (%s) is non-positive",
                wait_timeout,
            )
            break

        elif elapsed_time > wait_timeout:
            raise NodePoolDeletionError(
                f"Timed out after {wait_timeout}s waiting for {names} node pools"
                f" to delete. {elapsed_time} elapsed"
            )

        time.sleep(retry_interval)


def create_node_pool(
    name: str,
    *,
    project: str,
    zone: str,
    cluster: str,
    pre_provisioner_id: Optional[str] = None,
    num_nodes_per_pool: int,
    machine_type: str,
    topology: Optional[str] = None,
    use_spot_vm: Optional[bool] = None,
    reservation: Optional[str] = None,
    location_hint: Optional[str] = None,
    enable_tpu_ici_resiliency: Optional[bool] = None,
    service_account_email: Optional[str] = None,
    additional_labels: Optional[dict[str, str]] = None,
    fire_and_forget: bool = False,
) -> Any:
    """Create a node pool.

    See also:
    https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#create

    Args:
        name: Node pool name.
        project: GCP Project name.
        zone: GCP zone name.
        cluster: K8s cluster.
        pre_provisioner_id: ID of the pre-provisioner.
            It will be added to the node pool labels and can be used to
            identify the node pools created by the same pre-provisioner.
        num_nodes_per_pool: Number of node pools to create.
        machine_type: Machine type of nodes in the node pools.
        topology: TPU topology.
        use_spot_vm: Whether to use spot VMs. It is mutually exclusive with reservation.
        reservation: Name of TPU reservation. It is mutually exclusive with use_spot_vm.
        location_hint: Location of TPU reservation.
        enable_tpu_ici_resiliency: Whether to enable TPU ICI resiliency.
        service_account_email: Service account email.
        additional_labels: Additional labels attached to the node pool.
        fire_and_forget: If True, execute creation and return immediately regardless of failures.
            If False, raise an exception if node pool creation fails.

    Returns:
        Response of the v1 node pool creation API call.

    Raises:
        Exception: If node pool creation fails and fire_and_forget is False.
    """

    parent = _node_pool_parent(project=project, zone=zone, cluster=cluster)

    body = _node_pool_body(
        name=name,
        pre_provisioner_id=pre_provisioner_id,
        zone=zone,
        num_nodes=num_nodes_per_pool,
        machine_type=machine_type,
        topology=topology,
        use_spot_vm=use_spot_vm,
        reservation=reservation,
        location_hint=location_hint,
        enable_tpu_ici_resiliency=enable_tpu_ici_resiliency,
        service_account_email=service_account_email,
        additional_labels=additional_labels,
    )

    # Create the node pool
    try:
        resource = _node_pool_resource(get_credentials())
        return _create_node_pool(resource, parent=parent, body=body)
    except Exception as e:  # pylint: disable=broad-except
        if not fire_and_forget:
            raise e
        logging.warning(
            "Failed to create node pool %s; Exception %s. Ignoring due to fire_and_forget",
            name,
            e,
        )


def create_node_pools(
    names: list[str],
    *,
    project: str,
    zone: str,
    cluster: str,
    pre_provisioner_id: Optional[str] = None,
    num_nodes_per_pool: int,
    machine_type: str,
    topology: Optional[str] = None,
    use_spot_vm: Optional[bool] = None,
    reservation: Optional[str] = None,
    location_hint: Optional[str] = None,
    enable_tpu_ici_resiliency: Optional[bool] = None,
    service_account_email: Optional[str] = None,
    additional_labels_list: Optional[list[dict[str, str]]] = None,
    retry_interval: int = 30,
    wait_timeout: int = 0,
):
    """Create node pools.

    See also:
    https://googleapis.github.io/google-api-python-client/docs/dyn/container_v1.projects.locations.clusters.nodePools.html#create

    Args:
        names: List of node pools to delete.
        project: GCP Project name.
        zone: GCP zone name.
        cluster: K8s cluster.
        pre_provisioner_id: ID of the pre-provisioner.
            It will be added to the node pool labels and can be used to
            identify the node pools created by the same pre-provisioner.
        num_nodes_per_pool: Number of node pools to create.
        machine_type: Machine type of nodes in the node pools.
        topology: TPU topology.
        use_spot_vm: Whether to use spot VMs. It is mutually exclusive with reservation.
        reservation: Name of TPU reservation. It is mutually exclusive with use_spot_vm.
        location_hint: Location of TPU reservation.
        enable_tpu_ici_resiliency: Whether to enable TPU ICI resiliency.
        service_account_email: Service account email.
        additional_labels_list: Additional labels attached to each node pool.
        retry_interval: Time in seconds between retries.
        wait_timeout: Seconds to wait for node pools to delete.
            This function doesn't wait if wait_timeout <= 0.

    Raises:
        NodePoolCreationError: If node pools are not deleted
            within wait_timeout if wait_timeout > 0.
    """

    start_time = time.perf_counter()
    while True:
        # Wait until all node pools are running, or timeout.
        num_node_pools_running = 0

        for i in range(len(names)):
            node_pool_name = names[i]
            additional_labels = (
                additional_labels_list[i] if additional_labels_list is not None else {}
            )

            node_pool_status = get_node_pool_status(
                project=project, zone=zone, cluster=cluster, name=node_pool_name
            )

            if node_pool_status == NodePoolStatus.RUNNING:
                logging.info("Node pool %s is running; skip creating it", node_pool_name)
                num_node_pools_running += 1
            else:
                logging.info("Creating node pool %s", node_pool_name)
                create_node_pool(
                    node_pool_name,
                    project=project,
                    zone=zone,
                    cluster=cluster,
                    pre_provisioner_id=pre_provisioner_id,
                    num_nodes_per_pool=num_nodes_per_pool,
                    machine_type=machine_type,
                    topology=topology,
                    use_spot_vm=use_spot_vm,
                    reservation=reservation,
                    location_hint=location_hint,
                    enable_tpu_ici_resiliency=enable_tpu_ici_resiliency,
                    service_account_email=service_account_email,
                    additional_labels=additional_labels,
                    fire_and_forget=True,
                )
                logging.warning("Node pool %s is %s", node_pool_name, node_pool_status)

        elapsed_time = time.perf_counter() - start_time

        if num_node_pools_running == len(names):
            logging.info(
                "All %s node pools for %s creation took %s seconds",
                len(names),
                pre_provisioner_id,
                elapsed_time,
            )
            break

        if wait_timeout <= 0:
            logging.info(
                "Skip waiting for node pool creation " "since wait_timeout (%s) is non-positive",
                wait_timeout,
            )
            break

        elif elapsed_time > wait_timeout:
            raise NodePoolCreationError(
                f"Timed out after {wait_timeout}s waiting for {len(names)} "
                f"node pools to launch. {elapsed_time} elapsed"
            )

        time.sleep(retry_interval)


def list_node_pools_by_label_key(
    *,
    project: str,
    zone: str,
    cluster: str,
    label_key: str,
) -> Dict[str, list]:
    """List all node pools with a label matching `label_key`.

    Args:
        project: GCP project name.
        zone: GCP zone name.
        cluster: GKE cluster name.
        label_key: The label to search for.

    Returns:
        A dict of label value -> list of node pool objects.
    """
    node_pool_parent = _node_pool_parent(project=project, zone=zone, cluster=cluster)
    resource = _node_pool_resource(get_credentials())
    result = _list_node_pools(resource, parent=node_pool_parent)

    logging.debug("List node pools: %s", result)

    node_pool_dict = defaultdict(list)

    for node_pool in (result or {}).get("nodePools", []):
        labels = node_pool.get("config", {}).get("labels", {})

        label_val = labels.get(label_key, None)

        if label_val is not None:
            node_pool_dict[label_val].append(node_pool)

    return node_pool_dict


def get_node_pool_status(
    name: str,
    *,
    project: str,
    zone: str,
    cluster: str,
) -> NodePoolStatus:
    """Get node pool status.

    Args:
        name: Node pool name.
        project: GCP project name.
        zone: GCP zone name.
        cluster: GKE cluster name.

    Returns:
        Node pool status.
    """

    node_pool_cid = _node_pool_canonical_id(project=project, zone=zone, cluster=cluster, name=name)
    try:
        resource = _node_pool_resource(get_credentials())
        res = _get_node_pool(resource, name=node_pool_cid)
        logging.debug("Node Pool %s status: %s", node_pool_cid, res)
        if "status" in res:
            status: str = res["status"]
            return NodePoolStatus[status.upper()]
        return NodePoolStatus.UNKNOWN
    except errors.HttpError as e:
        if e.resp.status == 404:
            logging.warning("Node pool %s does not exist.", node_pool_cid)
            return NodePoolStatus.NOT_EXIST
        raise


def construct_node_pool_name(*, jobset_namespace: str, jobset_name: str, index: int) -> str:
    """Construct the node pool name.

    Node pool name should not exceed 40 characters. Use job name and node pool index
    to construct a unique name while satisfying this condition.
    First calculate the hash value of jobset_namespace/jobset_name-index, then construct a
    node pool name as {first 34 chars of jobset_name}-{first 5 chars of hash}.

    Args:
        jobset_namespace: The jobset namespace.
        jobset_name: The jobset name.
        index: The node pool index.

    Returns:
        A unique node pool name not longer than 40 characters.
    """

    long_name = f"{jobset_namespace}/{jobset_name}-{index}"
    hash_val = hashlib.md5(long_name.encode("utf-8")).hexdigest()
    return f"{jobset_name[:34]}-{hash_val[:5]}"
