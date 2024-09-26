# Copyright © 2024 Apple Inc.

"""Tests node_pool_provisioner module."""

import contextlib
import io
from datetime import datetime
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bastion import (
    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR,
    new_jobspec,
    serialize_jobspec,
)
from axlearn.cloud.common.types import JobMetadata
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
        enable_tpu_smart_repair: bool = False,
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
            job_cfg.enable_tpu_smart_repair = enable_tpu_smart_repair

            provisioner_cfg = node_pool_provisioner.TPUNodePoolProvisioner.from_flags(fv)

            yield job_cfg, provisioner_cfg

    def _create_serialized_job_spec(self, job_priority):
        test_spec = new_jobspec(
            name="test_job",
            command="test command",
            metadata=JobMetadata(
                user_id="test_id",
                project_id="test_project",
                # Make sure str timestamp isn't truncated even when some numbers are 0.
                creation_time=datetime(1900, 1, 1, 0, 0, 0, 0),
                resources={"test": 8},
                priority=job_priority,
            ),
        )
        serialized_jobspec = io.StringIO()
        serialize_jobspec(test_spec, serialized_jobspec)
        return serialized_jobspec.getvalue()

    @parameterized.parameters(
        dict(num_replicas=1, job_priority=None, enable_tpu_smart_repair=False),
        dict(num_replicas=2, job_priority=1, enable_tpu_smart_repair=True),
    )
    def test_create_for(self, num_replicas: int, job_priority: int, enable_tpu_smart_repair: bool):
        env = {}
        if job_priority is not None:
            env.update(
                {
                    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR: self._create_serialized_job_spec(
                        job_priority
                    )
                }
            )  # pytype: disable=attribute-error

        mock_create_node_pools = mock.Mock()
        mock_construct_node_pool_name = mock.Mock()

        mock_utils = mock.patch.multiple(
            node_pool_provisioner.__name__,
            create_node_pools=mock_create_node_pools,
            construct_node_pool_name=mock_construct_node_pool_name,
        )

        with self._mock_configs(num_replicas, enable_tpu_smart_repair) as [
            job_cfg,
            provisioner_cfg,
        ], mock_utils, mock.patch.dict("os.environ", env):
            tpu_gke_job = job_cfg.instantiate()
            provisioner = provisioner_cfg.set(name="pre-provisioner-0").instantiate()

            provisioner.create_for(tpu_gke_job)

            self.assertEqual(num_replicas, mock_construct_node_pool_name.call_count)
            self.assertEqual(1, mock_create_node_pools.call_count)

            additional_labels_list = mock_create_node_pools.call_args.kwargs[
                "additional_labels_list"
            ]
            self.assertEqual(num_replicas, len(additional_labels_list))

            for additional_labels in additional_labels_list:
                if job_priority is None:
                    self.assertNotIn("job-priority", additional_labels.keys())
                else:
                    self.assertIn("job-priority", additional_labels.keys())
                    self.assertEqual(str(job_priority), additional_labels.get("job-priority"))

                if enable_tpu_smart_repair:
                    self.assertEqual(
                        "true", additional_labels.get("cloud.google.com/gke-tpu-auto-restart", None)
                    )
                else:
                    self.assertNotIn("cloud.google.com/gke-tpu-auto-restart", additional_labels)

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
