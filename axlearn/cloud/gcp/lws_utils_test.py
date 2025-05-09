"""Tests LWS utilities."""

import contextlib
from typing import Optional

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.gcp import bundler, lws_utils
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.system_characteristics import USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.test_utils import TestCase


class TPULeaderWorkerTemplateTest(TestCase):
    """Test TPULeaderWorkerTemplate."""

    @contextlib.contextmanager
    def _job_config(self, bundler_cls: type[Bundler], **kwargs):
        with mock_gcp_settings([lws_utils.__name__, bundler.__name__]):
            fv = flags.FlagValues()
            lws_utils.TPULeaderWorkerTemplate.define_flags(fv)
            fv.set_default("name", "test-name")
            fv.set_default("instance_type", "tpu-v6e-16")
            for key, value in kwargs.items():
                if value is not None:
                    setattr(fv, key, value)
            fv.mark_as_parsed()
            cfg = lws_utils.TPULeaderWorkerTemplate.from_flags(fv)
            bundler_cfg = bundler_cls.from_spec([], fv=fv).set(image="test-image")
            yield cfg, bundler_cfg

    @parameterized.product(
        [
            dict(
                reservation=None,
                reservation_project=None,
            ),
            dict(
                reservation="test-reservation",
                reservation_project=None,
            ),
            dict(
                reservation="test-reservation",
                reservation_project="test-reservation-project",
            ),
        ],
        bundler_cls=[ArtifactRegistryBundler, CloudBuildBundler],
    )
    def test_build_pod(
        self,
        bundler_cls: type[Bundler],
        reservation: Optional[str] = None,
        reservation_project: Optional[str] = None,
    ):
        with (
            self._job_config(
                bundler_cls,
            ) as (cfg, bundler_cfg),
        ):
            gke_lws: lws_utils.TPULeaderWorkerTemplate = cfg.set(
                reservation=reservation,
                reservation_project=reservation_project,
                name="test",
                command="test_command",
                output_dir="FAKE",
            ).instantiate(bundler=bundler_cfg.instantiate())
            # pylint: disable-next=protected-access
            pod = gke_lws._build_pod()
            pod_spec = pod["spec"]
            node_selector = pod_spec["nodeSelector"]
            if reservation is not None:
                self.assertEqual(
                    reservation, node_selector.get("cloud.google.com/reservation-name", None)
                )
                if reservation_project is not None:
                    self.assertEqual(
                        reservation_project,
                        node_selector.get("cloud.google.com/reservation-project", None),
                    )
                else:
                    self.assertNotIn("cloud.google.com/reservation-project", node_selector)
            else:
                self.assertNotIn("cloud.google.com/reservation-name", node_selector)

            self.assertEqual(len(pod_spec["containers"]), 1)
            container = pod_spec["containers"][0]

            self.assertIn("test_command", container["command"])

            resources = container["resources"]
            self.assertIn("limits", resources)
            tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
            tpu_characteristics = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[tpu_type]

            self.assertIn("google.com/tpu", resources["limits"])
            self.assertEqual(
                node_selector.get("cloud.google.com/gke-tpu-accelerator"),
                tpu_characteristics.gke_accelerator,
            )
            self.assertEqual(
                node_selector.get("cloud.google.com/gke-tpu-topology"), tpu_characteristics.topology
            )

    def test_leaderworker_template(self):
        with self._job_config(
            CloudBuildBundler,
        ) as (cfg, bundler_cfg):
            leaderworker_template: lws_utils.TPULeaderWorkerTemplate = cfg.set(
                name="test", command="test_command", output_dir="FAKE"
            ).instantiate(bundler=bundler_cfg.instantiate())

            leaderworker_template_spec = leaderworker_template()
            self.assertEqual(leaderworker_template_spec["size"], 4)
