"""Tests k8s service module."""

from absl import flags

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import k8s_service
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class GKELWSService(TestCase):
    """Tests GKEService with LWS(TPU)."""

    def _service_config(
        self,
        *,
        command: str,
        **kwargs,
    ) -> k8s_service.LWSService.Config:
        fv = flags.FlagValues()
        cfg = k8s_service.LWSService.default_config().set()

        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                # Use setattr rather than set_default to set flags.
                setattr(fv, key, value)
        fv.name = "fake-name"
        fv.project = "fake-project"
        fv.mark_as_parsed()
        cfg = from_flags(cfg, fv, command=command)
        # Test that retries are configured on fv by default.
        self.assertIsNotNone(fv["name"])
        self.assertIsNotNone(fv["project"])
        return cfg

    def test_instantiate(
        self,
    ):
        cfg = self._service_config(
            command="test-command",
            project="fake-project",
            ports=[9000],
        )
        self.assertIsInstance(cfg, k8s_service.LWSService.Config)
        self.assertEqual(cfg.project, "fake-project")
        gke_lws_service = cfg.set().instantiate()
        self.assertEqual(cfg.name + "-service", gke_lws_service.name)
        self.assertEqual(cfg.ports, gke_lws_service.ports)
