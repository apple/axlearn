# Copyright © 2026 Apple Inc.

"""Tests for MetricsCollection and PodMonitoring configurables."""

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp.k8s_metrics_collection import MetricsCollection, PodMonitoring


def _pm_flags(**overrides):
    fv = flags.FlagValues()
    PodMonitoring.define_flags(fv)
    fv.mark_as_parsed()
    for name, value in overrides.items():
        setattr(fv, name, value)
    return fv


def _pm(**overrides):
    defaults = dict(name="x", port=9090)
    defaults.update(overrides)
    return PodMonitoring.default_config().set(**defaults).instantiate()


def _prebuilt_pm_cfg():
    return PodMonitoring.default_config().set(
        name="pm",
        namespace="default",
        selector_labels={"k": "v"},
        port=0,  # placeholder; from_flags overwrites when scaling_metrics_port is set
    )


class _StubMetricsCollection(MetricsCollection):
    """Concrete stub of the abstract base — exercises the default
    `compose_metric_name` behavior without depending on a real K8s impl.
    """

    def execute(self, *, owner=None):
        pass


class MetricsCollectionComposeTest(parameterized.TestCase):
    """Tests for MetricsCollection.compose_metric_name (default identity)."""

    def test_base_is_identity(self):
        stub = _StubMetricsCollection.default_config().set(name="x").instantiate()
        self.assertEqual(stub.compose_metric_name("test_metric"), "test_metric")


class PodMonitoringComposeTest(parameterized.TestCase):
    """Tests for PodMonitoring.compose_metric_name."""

    @parameterized.parameters(
        dict(overrides={}, expected="prometheus.googleapis.com|test_metric|gauge"),
        dict(overrides=dict(metric_name_prefix=""), expected="test_metric|gauge"),
        dict(overrides=dict(metric_name_kind=""), expected="prometheus.googleapis.com|test_metric"),
        dict(
            overrides=dict(metric_name_prefix="", metric_name_kind=""),
            expected="test_metric",
        ),
    )
    def test_composition(self, overrides, expected):
        self.assertEqual(_pm(**overrides).compose_metric_name("test_metric"), expected)

    def test_pre_composed_returned_unchanged(self):
        # Names that already contain `|` are assumed pre-composed.
        self.assertEqual(_pm().compose_metric_name("a|b|c"), "a|b|c")


class PodMonitoringFromFlagsTest(parameterized.TestCase):
    """Tests for PodMonitoring.from_flags."""

    def test_returns_prebuilt_when_no_port(self):
        cfg = PodMonitoring.from_flags(_pm_flags(), prebuilt_cfg=_prebuilt_pm_cfg())
        # Port unchanged; the cfg is passed through verbatim.
        self.assertEqual(cfg.port, 0)

    def test_populates_port_path_interval_and_composition(self):
        fv = _pm_flags(
            scaling_metrics_port="9090",
            scaling_metrics_path="/custom",
            scaling_metrics_interval="15s",
            scaling_metric_prefix="my.prefix",
            scaling_metric_kind="counter",
        )
        cfg = PodMonitoring.from_flags(fv, prebuilt_cfg=_prebuilt_pm_cfg())
        self.assertEqual(cfg.port, 9090)
        self.assertEqual(cfg.path, "/custom")
        self.assertEqual(cfg.interval, "15s")
        self.assertEqual(cfg.metric_name_prefix, "my.prefix")
        self.assertEqual(cfg.metric_name_kind, "counter")

    def test_named_port(self):
        # Non-numeric port value falls back to a named port string.
        fv = _pm_flags(scaling_metrics_port="metrics-port")
        cfg = PodMonitoring.from_flags(fv, prebuilt_cfg=_prebuilt_pm_cfg())
        self.assertEqual(cfg.port, "metrics-port")
