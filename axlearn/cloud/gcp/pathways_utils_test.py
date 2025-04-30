# Copyright © 2025 Apple Inc.

"""Tests Pathways utilities."""
import contextlib

from absl import flags

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, jobset_utils, pathways_utils
from axlearn.cloud.gcp.bundler import CloudBuildBundler
from axlearn.cloud.gcp.pathways_utils import _PATHWAYS_HEAD_NODE_POOL_NAME, _PATHWAYS_SERVER_IMAGE
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.test_utils import TestCase


class PathwaysReplicatedJobTest(TestCase):
    """Tests PathwaysReplicatedJob."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], **kwargs):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            cfg = pathways_utils.PathwaysReplicatedJob.default_config().set(
                inner=jobset_utils.TPUReplicatedJob.default_config()
            )
            define_flags(cfg, fv)

            fv.set_default("name", "fake-name")
            fv.set_default("instance_type", "tpu-v5p-16")
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = from_flags(cfg, fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            print("debug: cfg: ", type(cfg))
            yield cfg, bundler_cfg

    def test_build_pathways_head_pod(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="test",
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())
            # pylint: disable-next=protected-access
            pod = builder._build_pathways_head_pod()
            pod_spec = pod["spec"]

            self.assertEqual(len(pod_spec["containers"]), 1)
            self.assertEqual(len(pod_spec["initContainers"]), 2)
            node_selector = pod_spec["nodeSelector"]
            self.assertEqual(
                _PATHWAYS_HEAD_NODE_POOL_NAME, node_selector.get("cloud.google.com/gke-nodepool")
            )

    def test_build_pathways_worker_pod(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="test",
                command="test_command",
                output_dir="FAKE",
                service_account="test-service-account",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())
            # pylint: disable-next=protected-access
            pod = builder._build_pathways_worker_pod()
            pod_spec = pod["spec"]

            host_alias = pod_spec["hostAliases"]
            self.assertEqual(1, len(host_alias))
            self.assertEqual(pod_spec.get("hostNetwork"), True)
            self.assertEqual(pod_spec.get("dnsPolicy"), "ClusterFirstWithHostNet")
            container = pod_spec.get("containers")[0]
            self.assertEqual(container["image"], _PATHWAYS_SERVER_IMAGE)
            annotations = pod["metadata"]["annotations"]
            self.assertEqual(
                "test-service-account@test-project.iam.gserviceaccount.com",
                annotations.get("tpu-provisioner.cloud.google.com/node-service-account", None),
            )

    def test_replicated_job(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="test",
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())

            replicated_jobs = builder()

            for replicated_job in replicated_jobs:
                replicated_job_name = replicated_job["name"]

                job_spec = replicated_job["template"]

                annotations = job_spec["metadata"]["annotations"]
                # Annotation to create load balancer.
                self.assertEqual(
                    f"test-{replicated_job_name}-service",
                    annotations.get("axlearn/replicatedjob-load-balancer-service-name", {}),
                )
                self.assertEqual(
                    "9000",
                    annotations.get("axlearn/replicatedjob-load-balancer-target-port", {}),
                )
                self.assertEqual(
                    "80",
                    annotations.get("axlearn/replicatedjob-load-balancer-port", {}),
                )
