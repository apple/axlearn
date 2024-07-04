# Copyright © 2024 Apple Inc.

"""Tests node_pool module."""
from functools import partial
from typing import Optional, Type
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp import node_pool as node_pool_utils
from axlearn.cloud.gcp.node_pool import (
    NodePoolStatus,
    NodePoolValidationError,
    _node_pool_body,
    construct_node_pool_name,
    create_node_pool,
    create_node_pools,
    delete_node_pool,
    delete_node_pools,
    get_node_pool_status,
    list_node_pools_by_label_key,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings


class NodePoolUtilsTest(parameterized.TestCase):
    """Tests node pool utils."""

    @parameterized.parameters(
        dict(node_pools={}, label="test", expected={}),
        dict(node_pools={"nodePools": []}, label="test", expected={}),
        dict(
            node_pools={
                "nodePools": [
                    {
                        "name": "pool0",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                    {
                        "name": "pool1",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                    {
                        "name": "pool2",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                    {
                        "name": "pool3",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                ]
            },
            label="provisioner-nodepool-id",
            expected={
                "auto-provisioner-1": [
                    {
                        "name": "pool0",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                    {
                        "name": "pool1",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                ]
            },
        ),
        dict(
            node_pools={
                "nodePools": [
                    {
                        "name": "pool0",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                    {
                        "name": "pool1",
                        "config": {"labels": {"provisioner-nodepool-id": "auto-provisioner-1"}},
                    },
                    {
                        "name": "pool2",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                    {
                        "name": "pool3",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                ]
            },
            label=node_pool_utils.PRE_PROVISIONER_LABEL,
            expected={
                "pre-provisioner-2": [
                    {
                        "name": "pool2",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                    {
                        "name": "pool3",
                        "config": {"labels": {"pre-provisioner-id": "pre-provisioner-2"}},
                    },
                ],
            },
        ),
    )
    def test_list_node_pools_by_label(self, node_pools, label, expected):
        with mock.patch.multiple(
            node_pool_utils.__name__,
            _list_node_pools=mock.Mock(return_value=node_pools),
            get_credentials=mock.DEFAULT,
        ):
            self.assertEqual(
                expected,
                list_node_pools_by_label_key(
                    project="test-project",
                    zone="test-region-zone",
                    cluster="test-cluster",
                    label_key=label,
                ),
            )

    @parameterized.product(
        use_spot_vm=[None, True, False],
        reservation=[None, "reservation-1"],
        location_hint=[None, "location-hint-1"],
        topology=[None, "test-topology"],
        additional_labels=[{"label1": "value1"}, {"label1": "value1", "label2": "value2"}, None],
    )
    def test_node_pool_body(
        self,
        use_spot_vm: Optional[bool] = None,
        reservation: Optional[str] = None,
        location_hint: Optional[str] = None,
        topology: Optional[str] = None,
        additional_labels: Optional[dict[str, str]] = None,
    ):
        mock_settings = {
            "service_account_email": "settings-service_account_email",
        }
        with mock_gcp_settings([node_pool_utils.__name__], mock_settings):
            fv = flags.FlagValues()
            fv.mark_as_parsed()

            test_fn = partial(
                _node_pool_body,
                name="node_pool0",
                zone="test-zone",
                num_nodes=1,
                machine_type="test-machine-type",
                use_spot_vm=use_spot_vm,
                reservation=reservation,
                location_hint=location_hint,
                topology=topology,
                additional_labels=additional_labels,
            )

            if use_spot_vm and reservation is not None:
                with self.assertRaises(NodePoolValidationError):
                    test_fn()
            else:
                body = test_fn()

                labels_in_body = body["nodePool"]["config"]["labels"]

                if reservation is not None:
                    reservation_in_body = body["nodePool"]["config"]["reservationAffinity"][
                        "values"
                    ][0]
                    self.assertEqual(reservation, reservation_in_body)

                if use_spot_vm is not None:
                    spot_in_body = body["nodePool"]["config"]["spot"]
                    self.assertEqual(use_spot_vm, spot_in_body)

                self.assertEqual(
                    location_hint, labels_in_body.get("cloud.google.com/gke-location-hint", None)
                )

                if topology is not None:
                    topology_in_body = body["nodePool"]["placementPolicy"]["tpuTopology"]
                    self.assertEqual(topology, topology_in_body)

                if additional_labels is not None:
                    for key, value in additional_labels.items():
                        self.assertEqual(value, labels_in_body.get(key, None))

    @parameterized.parameters(
        dict(fire_and_forget=True, exception=None, expected=None),
        dict(fire_and_forget=False, exception=RuntimeError, expected=RuntimeError),
        dict(fire_and_forget=True, exception=RuntimeError, expected=None),
        dict(fire_and_forget=False, exception=None, expected=None),
    )
    def test_delete_node_pool_fire_and_forget(
        self, fire_and_forget, exception, expected: Optional[Type[Exception]]
    ):
        with mock.patch.multiple(
            node_pool_utils.__name__,
            _node_pool_body=mock.DEFAULT,
            _delete_node_pool=mock.Mock(side_effect=exception),
            get_credentials=mock.DEFAULT,
        ):
            test_fn = partial(
                delete_node_pool,
                "node_pool0",
                project="test-project",
                zone="test-region-zone",
                cluster="test-cluster",
                fire_and_forget=fire_and_forget,
            )

            if expected is not None:
                with self.assertRaises(expected):
                    test_fn()
            else:
                test_fn()

    @parameterized.parameters(
        dict(names=["node_pool0"], wait_timeout=0, expected_delete_call_count=1),
        dict(names=["node_pool0", "node_pool1"], wait_timeout=0, expected_delete_call_count=2),
        dict(names=["node_pool0"], wait_timeout=2, expected_delete_call_count=2),
        dict(names=["node_pool0", "node_pool1"], wait_timeout=2, expected_delete_call_count=4),
    )
    def test_delete_node_pools(
        self, names: list[str], wait_timeout: int, expected_delete_call_count: int
    ):
        get_node_pool_status_side_effect = []
        for status in [NodePoolStatus.RUNNING, NodePoolStatus.RUNNING, NodePoolStatus.NOT_EXIST]:
            for _ in range(len(names)):
                get_node_pool_status_side_effect.append(status)

        mock_get_node_pool_status = mock.Mock(side_effect=get_node_pool_status_side_effect)
        mock_delete_node_pool = mock.Mock()

        with mock.patch.multiple(
            node_pool_utils.__name__,
            get_node_pool_status=mock_get_node_pool_status,
            get_credentials=mock.DEFAULT,
            delete_node_pool=mock_delete_node_pool,
        ):
            delete_node_pools(
                names,
                project="test-project",
                zone="test-zone",
                cluster="test-cluster",
                retry_interval=0,
                wait_timeout=wait_timeout,
            )

            self.assertEqual(expected_delete_call_count, mock_delete_node_pool.call_count)

    @parameterized.parameters(
        dict(fire_and_forget=True, exception=None, expected=None),
        dict(fire_and_forget=False, exception=RuntimeError, expected=RuntimeError),
        dict(fire_and_forget=True, exception=RuntimeError, expected=None),
        dict(fire_and_forget=False, exception=None, expected=None),
    )
    def test_create_node_pool_fire_and_forget(
        self, fire_and_forget, exception, expected: Optional[Type[Exception]]
    ):
        with mock.patch.multiple(
            node_pool_utils.__name__,
            _node_pool_body=mock.DEFAULT,
            _create_node_pool=mock.Mock(side_effect=exception),
            get_credentials=mock.DEFAULT,
        ):
            test_fn = partial(
                create_node_pool,
                "node_pool0",
                project="test-project",
                zone="test-region-zone",
                cluster="test-cluster",
                num_nodes_per_pool=1,
                machine_type="test-type",
                fire_and_forget=fire_and_forget,
            )

            if expected is not None:
                with self.assertRaises(expected):
                    test_fn()
            else:
                test_fn()

    @parameterized.parameters(
        dict(names=["node_pool0"], wait_timeout=0, expected_create_call_count=1),
        dict(names=["node_pool0", "node_pool1"], wait_timeout=0, expected_create_call_count=2),
        dict(names=["node_pool0"], wait_timeout=2, expected_create_call_count=2),
        dict(names=["node_pool0", "node_pool1"], wait_timeout=2, expected_create_call_count=4),
    )
    def test_create_node_pools(
        self, names: list[str], wait_timeout: int, expected_create_call_count: int
    ):
        get_node_pool_status_side_effect = []
        for status in [
            NodePoolStatus.NOT_EXIST,
            NodePoolStatus.PROVISIONING,
            NodePoolStatus.RUNNING,
        ]:
            for _ in range(len(names)):
                get_node_pool_status_side_effect.append(status)

        mock_get_node_pool_status = mock.Mock(side_effect=get_node_pool_status_side_effect)
        mock_create_node_pool = mock.Mock()

        with mock.patch.multiple(
            node_pool_utils.__name__,
            get_node_pool_status=mock_get_node_pool_status,
            get_credentials=mock.DEFAULT,
            create_node_pool=mock_create_node_pool,
        ):
            create_node_pools(
                names,
                project="test-project",
                zone="test-zone",
                cluster="test-cluster",
                num_nodes_per_pool=1,
                machine_type="test-type",
                retry_interval=0,
                wait_timeout=wait_timeout,
            )

            self.assertEqual(expected_create_call_count, mock_create_node_pool.call_count)

    @parameterized.parameters(
        dict(
            name="node_pool0",
            node_pool={"name": "node_pool0", "status": "running"},
            expected=NodePoolStatus.RUNNING,
        ),
        dict(
            name="node_pool0",
            node_pool={"name": "node_pool0", "status": "provisioning"},
            expected=NodePoolStatus.PROVISIONING,
        ),
        dict(
            name="node_pool0",
            node_pool={"name": "node_pool0", "status": "error"},
            expected=NodePoolStatus.ERROR,
        ),
        dict(
            name="node_pool0",
            node_pool={"name": "node_pool0", "status": "not_exist"},
            expected=NodePoolStatus.NOT_EXIST,
        ),
    )
    def test_get_node_pool_status(self, name: str, node_pool: dict, expected: NodePoolStatus):
        with mock.patch.multiple(
            node_pool_utils.__name__,
            _get_node_pool=mock.Mock(return_value=node_pool),
            get_credentials=mock.DEFAULT,
        ):
            self.assertEqual(
                expected,
                get_node_pool_status(
                    name,
                    project="test-project",
                    zone="test-region-zone",
                    cluster="test-cluster",
                ),
            )

    @parameterized.parameters(
        # Node pool 0 with a short jobset name.
        dict(
            jobset_namespace="default",
            jobset_name="test-job-0",
            index=0,
            expected="test-job-0-3702e",
        ),
        # Node pool 1 with a short jobset name.
        dict(
            jobset_namespace="default",
            jobset_name="test-job-0",
            index=1,
            expected="test-job-0-249eb",
        ),
        # Node pool 0 with a very long jobset name.
        dict(
            jobset_namespace="default",
            jobset_name="test-a-very-very-very-long-job-name-job-0",
            index=0,
            expected="test-a-very-very-very-long-job-nam-7bee7",
        ),
        # Node pool 1 with a very long jobset name.
        dict(
            jobset_namespace="default",
            jobset_name="test-a-very-very-very-long-job-name-job-0",
            index=1,
            expected="test-a-very-very-very-long-job-nam-a2963",
        ),
    )
    def test_construct_node_pool_name(
        self, jobset_namespace: str, jobset_name: str, index: int, expected: str
    ):
        res = construct_node_pool_name(
            jobset_namespace=jobset_namespace, jobset_name=jobset_name, index=index
        )
        self.assertEqual(expected, res)
