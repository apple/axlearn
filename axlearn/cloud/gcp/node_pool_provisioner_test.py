# Copyright © 2024 Apple Inc.

"""Tests node_pool_provisioner module."""

import contextlib
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp import bundler, job, node_pool_provisioner
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler
from axlearn.cloud.gcp.job import TPUGKEJob
from axlearn.cloud.gcp.test_utils import mock_gcp_settings


class TPUNodePoolProvisionerTest(parameterized.TestCase):
    """Tests TPUNodePoolProvisioner."""

    @property
    def _mock_settings(self):
        return {
            "project": "settings-project",
            "zone": "settings-zone",
            "gke_cluster": "settings-cluster",
            "service_account_email": "settings-service-account-email",
            "docker_repo": "settings-repo",
            "default_dockerfile": "settings-dockerfile",
        }

    @contextlib.contextmanager
    def _mock_configs(
        self,
        num_replicas: int,
    ):
        with mock_gcp_settings(
            [node_pool_provisioner.__name__, job.__name__, bundler.__name__], self._mock_settings
        ):
            fv = flags.FlagValues()
            TPUGKEJob.define_flags(fv)
            fv.set_default("num_replicas", num_replicas)
            fv.mark_as_parsed()
            job_cfg = TPUGKEJob.from_flags(fv)
            job_cfg.bundler = ArtifactRegistryBundler.from_spec([], fv=fv).set(image="test-image")
            job_cfg.accelerator.instance_type = "tpu-v4-8"

            provisioner_cfg = node_pool_provisioner.TPUNodePoolProvisioner.from_flags(fv)

            yield job_cfg, provisioner_cfg

    @parameterized.parameters(
        dict(num_replicas=1),
        dict(num_replicas=2),
    )
    def test_create_for(self, num_replicas: int):
        mock_create_node_pools = mock.Mock()
        mock_construct_node_pool_name = mock.Mock()

        mock_utils = mock.patch.multiple(
            node_pool_provisioner.__name__,
            create_node_pools=mock_create_node_pools,
            construct_node_pool_name=mock_construct_node_pool_name,
        )

        with self._mock_configs(num_replicas) as [job_cfg, provisioner_cfg], mock_utils:
            tpu_gke_job = job_cfg.instantiate()
            provisioner = provisioner_cfg.set(name="pre-provisioner-0").instantiate()

            provisioner.create_for(tpu_gke_job)

            self.assertEqual(num_replicas, mock_construct_node_pool_name.call_count)
            self.assertEqual(1, mock_create_node_pools.call_count)

    @parameterized.parameters(
        dict(num_replicas=1),
        dict(num_replicas=2),
    )
    def test_delete_for(self, num_replicas: int):
        mock_delete_node_pools = mock.Mock()
        mock_construct_node_pool_name = mock.Mock()

        mock_utils = mock.patch.multiple(
            node_pool_provisioner.__name__,
            delete_node_pools=mock_delete_node_pools,
            construct_node_pool_name=mock_construct_node_pool_name,
        )

        with self._mock_configs(num_replicas) as [job_cfg, provisioner_cfg], mock_utils:
            tpu_gke_job = job_cfg.instantiate()
            provisioner = provisioner_cfg.set(name="pre-provisioner-0").instantiate()

            provisioner.delete_for(tpu_gke_job)

            self.assertEqual(num_replicas, mock_construct_node_pool_name.call_count)
            self.assertEqual(1, mock_delete_node_pools.call_count)
