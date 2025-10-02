# Copyright © 2025 Apple Inc.

"""Tests Pathways utilities."""
import contextlib

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import bundler, jobset_utils, lws_utils, pathways_utils
from axlearn.cloud.gcp.bundler import CloudBuildBundler
from axlearn.cloud.gcp.pathways_utils import (
    _PATHWAYS_HEAD_NODE_POOL_SELECTOR_KEY,
    _PATHWAYS_HEAD_NODE_POOL_SELECTOR_VALUE,
    _PATHWAYS_PROXY_CONTAINER_NAME,
    _PATHWAYS_RESOURCE_MANAGER_CONTAINER_NAME,
    _PATHWAYS_SERVER_IMAGE,
    get_megascale_options,
    get_xla_options,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.compiler_options import default_xla_options, xla_flags_from_options
from axlearn.common.test_utils import TestCase


class HelperFunctionTest(TestCase):
    def test_round_up_to_power_of_2(self):
        with self.assertRaises(AssertionError):
            pathways_utils.round_up_to_power_of_2(-1)
        with self.assertRaises(AssertionError):
            pathways_utils.round_up_to_power_of_2(0)
        with self.assertRaises(AssertionError):
            pathways_utils.round_up_to_power_of_2(2.2)
        self.assertEqual(pathways_utils.round_up_to_power_of_2(2), 2)
        self.assertEqual(pathways_utils.round_up_to_power_of_2(7), 8)
        self.assertEqual(pathways_utils.round_up_to_power_of_2(8), 8)
        self.assertEqual(pathways_utils.round_up_to_power_of_2(9), 16)
        self.assertEqual(pathways_utils.round_up_to_power_of_2(10), 16)
        # ct5p-hightpu-4t host memory
        self.assertEqual(pathways_utils.round_up_to_power_of_2(448 // 4), 128)
        # ct6e-standard-4t host memory
        self.assertEqual(pathways_utils.round_up_to_power_of_2(720 // 4), 256)


class SplitXLAMXLAFlagsTest(TestCase):
    """Test the splitting of XLA and Megascale flags."""

    def test_v6e_default_options_split_megascale_and_xla(self):
        default_options = default_xla_options(
            instance_type="tpu-v6e-512", num_slices=2, backend="tpu"
        )
        megascale_options = get_megascale_options(default_options)
        xla_options = get_xla_options(default_options)
        self.assertEqual(len(megascale_options) + len(xla_options), len(default_options))


class PathwaysReplicatedJobTest(TestCase):
    """Tests PathwaysReplicatedJob."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], instance_type: str = "tpu-v5p-16", **kwargs):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            cfg = pathways_utils.PathwaysReplicatedJob.default_config().set(
                inner=jobset_utils.TPUReplicatedJob.default_config()
            )
            define_flags(cfg, fv)

            fv.set_default("name", "fake-name")
            fv.set_default("instance_type", instance_type)
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = from_flags(cfg, fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.parameters(dict(instance_type="tpu-v5p-16"), dict(instance_type="tpu-v5p-256"))
    def test_build_pathways_head_pod(self, instance_type):
        with (
            self._job_config(
                CloudBuildBundler,
                instance_type,
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
            # pathways-proxy, pathways-rm and output-uploader
            self.assertEqual(len(pod_spec["initContainers"]), 3)
            node_selector = pod_spec["nodeSelector"]
            self.assertEqual(
                _PATHWAYS_HEAD_NODE_POOL_SELECTOR_VALUE,
                node_selector.get(_PATHWAYS_HEAD_NODE_POOL_SELECTOR_KEY),
            )

            head_container = pod_spec["containers"][0]
            env_vars = set()
            for env_pair in head_container["env"]:
                env_vars.add(env_pair["name"])
                # pylint: disable=line-too-long
                if env_pair["name"] == "NUM_REPLICAS":
                    self.assertEqual(
                        env_pair["valueFrom"],
                        {
                            "fieldRef": {
                                "fieldPath": "metadata.annotations['jobset.sigs.k8s.io/replicatedjob-replicas']"
                            }
                        },
                    )
                if env_pair["name"] == "IFRT_PROXY_LARGE_TRANSFER_THRESHOLD":
                    self.assertEqual(env_pair["value"], "1")
                if env_pair["name"] == "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY":
                    self.assertEqual(env_pair["value"], "/tmp/ifrt_proxy")

            self.assertTrue(
                {
                    "NUM_REPLICAS",
                    "REPLICA_ID",
                    "IFRT_PROXY_LARGE_TRANSFER_THRESHOLD",
                    "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY",
                }.issubset(env_vars)
            )

            # Check pathways-proxy container args for XLA flags.
            proxy_container = None
            rm_container = None
            for container in pod_spec["initContainers"]:
                if container["name"] == _PATHWAYS_PROXY_CONTAINER_NAME:
                    proxy_container = container
                if container["name"] == _PATHWAYS_RESOURCE_MANAGER_CONTAINER_NAME:
                    rm_container = container
            self.assertIsNotNone(proxy_container, "Pathways proxy container not found.")
            self.assertIsNotNone(rm_container, "Pathways rm container not found.")

            # pylint: disable-next=protected-access
            xla_arg_flags = xla_flags_from_options(builder._xla_options).split()
            self.assertTrue(xla_arg_flags, "XLA flags should be present")
            for flag in xla_arg_flags:
                self.assertIn(flag, proxy_container["args"])

            # Check that instance_type and expected_instances are set
            if instance_type == "tpu-v5p-16":
                self.assertIn("--instance_count=1", rm_container["args"])
                self.assertIn("--instance_type=tpuv5:2x2x2", rm_container["args"])
            if instance_type == "tpu-v5p-256":
                self.assertIn("--instance_count=1", rm_container["args"])
                self.assertIn("--instance_type=tpuv5:4x4x8_untwisted", rm_container["args"])

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
            worker_container = pod_spec.get("containers")[0]
            self.assertEqual(worker_container["image"], _PATHWAYS_SERVER_IMAGE)
            annotations = pod["metadata"]["annotations"]
            self.assertEqual(
                "test-service-account@test-project.iam.gserviceaccount.com",
                annotations.get("tpu-provisioner.cloud.google.com/node-service-account", None),
            )
            self.assertIn("--tpu_pinned_host_allocation_recycle=true", worker_container["args"])
            # 128GiB
            self.assertIn("--tpu_premapped_buffer_size=137438953472", worker_container["args"])

            # Check worker container args for Megascale (MXLA) flags.
            # pylint: disable-next=protected-access
            mxla_arg_flags = xla_flags_from_options(builder._mxla_options).split()
            # MXLA flags are generally only present in multi-slice jobs.
            for flag in mxla_arg_flags:
                self.assertIn(flag, worker_container["args"])

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

    def test_pathways_xla_flags_processing(self):
        """Tests processing of pathways_xla_flags, including overrides and new flags."""
        flag_to_override_key = "xla_tpu_enable_latency_hiding_scheduler"
        override_value_str = "false"
        expected_override_value_parsed = "false"

        new_xla_flag_key = "xla_a_brand_new_one"
        new_xla_flag_value_str = "12345"
        expected_new_xla_value_parsed = 12345

        new_mxla_flag_key = "megascale_another_setting"
        new_mxla_flag_value_str = "a_string_value"
        expected_new_mxla_value_parsed = new_mxla_flag_value_str

        pathways_xla_flags_input = [
            f"{flag_to_override_key}={override_value_str}",
            f"{new_xla_flag_key}={new_xla_flag_value_str}",
            f"--{new_mxla_flag_key}={new_mxla_flag_value_str}",
        ]

        initial_default_options = default_xla_options(
            instance_type="tpu-v5p-16", num_slices=1, backend="tpu"
        )
        # Ensure the flag we intend to override actually exists in the defaults.
        self.assertIn(flag_to_override_key, initial_default_options)

        with self._job_config(
            CloudBuildBundler,
            pathways_xla_flags=pathways_xla_flags_input,
        ) as (cfg, bundler_cfg):
            cfg.inner.set(
                project="test-project",
                name="test-inner-name",
                command="test-inner-command",
                output_dir="test-inner-output",
                service_account="test-service-account",
            )
            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())

            # pylint: disable=protected-access
            actual_xla_options = builder._xla_options
            actual_mxla_options = builder._mxla_options
            # pylint: enable=protected-access

            # Check overridden flag.
            self.assertIn(flag_to_override_key, actual_xla_options)
            self.assertEqual(
                actual_xla_options[flag_to_override_key], expected_override_value_parsed
            )

            # Check new XLA flag.
            self.assertIn(new_xla_flag_key, actual_xla_options)
            self.assertEqual(actual_xla_options[new_xla_flag_key], expected_new_xla_value_parsed)

            # Check new Megascale flag.
            self.assertIn(new_mxla_flag_key, actual_mxla_options)
            self.assertEqual(actual_mxla_options[new_mxla_flag_key], expected_new_mxla_value_parsed)

    def test_validate_head_name(self):
        with self._job_config(CloudBuildBundler) as (cfg, bundler_cfg):
            cfg.inner.set(
                project="test-project",
                name="a" * 40,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            with self.assertRaisesRegex(
                ValueError, r"pathways-head-1-1-abcde exceeds max \(63\) by 1 chars."
            ):
                _ = cfg.instantiate(bundler=bundler_cfg.instantiate())

    def test_validate_worker_name(self):
        with self._job_config(CloudBuildBundler) as (cfg, bundler_cfg):
            cfg.inner.set(
                project="test-project",
                name="a" * 38,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            with self.assertRaisesRegex(
                ValueError, r"pathways-worker-1-2-abcde exceeds max \(63\) by 1 chars."
            ):
                _ = cfg.instantiate(bundler=bundler_cfg.instantiate())


class PathwaysMultiheadReplicatedJobTest(TestCase):
    """Tests PathwaysMultiheadReplicatedJob."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], num_replicas: int, **kwargs):
        with mock_gcp_settings([jobset_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            cfg = pathways_utils.PathwaysMultiheadReplicatedJob.default_config().set(
                inner=jobset_utils.TPUReplicatedJob.default_config()
            )
            define_flags(cfg, fv)

            fv.set_default("name", "fake-name")
            fv.set_default("instance_type", "tpu-v5p-16")
            fv.set_default("num_replicas", num_replicas)
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = from_flags(cfg, fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.parameters([1, 2])
    def test_replicated_job(self, num_replicas):
        with (self._job_config(CloudBuildBundler, num_replicas) as (cfg, bundler_cfg),):
            cfg.inner.set(
                project="test-project",
                name="test",
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())

            replicated_jobs = builder()

            self.assertEqual(len(replicated_jobs), num_replicas + 1)

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

                if replicated_job_name.startswith("pathways-head"):
                    self.assertEqual(replicated_job["replicas"], num_replicas)
                elif replicated_job_name.startswith("pathways-worker"):
                    self.assertEqual(replicated_job["replicas"], 1)

    def test_validate_head_name(self):
        with self._job_config(CloudBuildBundler, 2) as (cfg, bundler_cfg):
            cfg.inner.set(
                project="test-project",
                name="a" * 40,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

        with self.assertRaisesRegex(
            ValueError, r"pathways-head-1-1-abcde exceeds max \(63\) by 1 chars."
        ):
            _ = cfg.instantiate(bundler=bundler_cfg.instantiate())

    def test_validate_worker_name(self):
        with self._job_config(CloudBuildBundler, 2) as (cfg, bundler_cfg):
            cfg.inner.set(
                project="test-project",
                name="a" * 36,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

        with self.assertRaisesRegex(
            ValueError, r"pathways-worker-2-0-2-abcde exceeds max \(63\) by 1 chars."
        ):
            _ = cfg.instantiate(bundler=bundler_cfg.instantiate())


class PathwaysLeaderWorkerTemplateTest(TestCase):
    """Test PathwaysLeaderWorkerTemplate."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], **kwargs):
        with mock_gcp_settings([lws_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            cfg = pathways_utils.PathwaysLeaderWorkerTemplate.default_config()
            define_flags(cfg, fv)
            fv.set_default("name", "fake-name")
            fv.set_default("instance_type", "tpu-v6e-16")
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = from_flags(cfg, fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            print("debug: cfg: ", type(cfg))
            yield cfg, bundler_cfg

    def test_build_leader_pod(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="a" * 36,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = builder.build_leader_pod()
            pod_spec = pod["spec"]

            self.assertEqual(len(pod_spec["containers"]), 3)

    def test_build_worker_pod(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="a" * 36,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())
            pod = builder.build_worker_pod()
            pod_spec = pod["spec"]
            container = pod_spec.get("containers")[0]
            self.assertEqual(container["image"], _PATHWAYS_SERVER_IMAGE)
            self.assertEqual(len(container["args"]), 3)

    def test_leader_worker_template(self):
        with (
            self._job_config(
                CloudBuildBundler,
            ) as (cfg, bundler_cfg),
        ):
            cfg.inner.set(
                project="test-project",
                name="a" * 36,
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())

            builder = cfg.instantiate(bundler=bundler_cfg.instantiate())
            lws = builder()
            leader_template = lws["leaderTemplate"]
            worker_template = lws["workerTemplate"]

            self.assertEqual(lws["size"], 5)
            self.assertEqual(len(leader_template["spec"]["containers"]), 3)
            self.assertEqual(len(worker_template["spec"]["containers"]), 1)
