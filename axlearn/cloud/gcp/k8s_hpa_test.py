# Copyright © 2026 Apple Inc.

"""Tests for HPA and HPAMetric configurables."""

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.gcp.k8s_hpa import HPA, DescribedObject, HPAMetric


def _metric(**overrides):
    defaults = dict(
        type="Pods",
        metric_name="test_metric",
        target_type="AverageValue",
        target_value="100",
    )
    defaults.update(overrides)
    return HPAMetric.default_config().set(**defaults).instantiate()


def _hpa_flags(**overrides):
    """Build a FlagValues with HPA-owned flags pre-set."""
    fv = flags.FlagValues()
    HPA.define_flags(fv)
    fv.mark_as_parsed()
    for name, value in overrides.items():
        setattr(fv, name, value)
    return fv


def _prebuilt_hpa_cfg():
    return HPA.default_config().set(
        name="hpa",
        namespace="default",
        scale_target_api_version="apps/v1",
        scale_target_kind="Deployment",
        scale_target_name="dep",
    )


class HPAMetricTest(parameterized.TestCase):
    """Tests for HPAMetric.build_identifier and HPAMetric.build_spec."""

    def test_build_identifier_without_selector(self):
        self.assertEqual(_metric().build_identifier(), {"name": "test_metric"})

    def test_build_identifier_with_selector(self):
        self.assertEqual(
            _metric(selector={"k": "v"}).build_identifier(),
            {"name": "test_metric", "selector": {"matchLabels": {"k": "v"}}},
        )

    @parameterized.parameters(
        dict(
            kwargs=dict(),
            expected={
                "type": "Pods",
                "pods": {
                    "metric": {"name": "test_metric"},
                    "target": {"type": "AverageValue", "averageValue": "100"},
                },
            },
        ),
        dict(
            kwargs=dict(selector={"workload": "x"}),
            expected_selector={"matchLabels": {"workload": "x"}},
        ),
        dict(
            kwargs=dict(
                type="Resource",
                metric_name="cpu",
                target_type="Utilization",
                target_value="80",
            ),
            expected={
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {"type": "Utilization", "averageUtilization": 80},
                },
            },
        ),
        dict(
            kwargs=dict(type="External"),
            expected_type="External",
            expected_top_key="external",
        ),
        dict(
            kwargs=dict(
                type="Object",
                described_object=DescribedObject.default_config().set(
                    api_version="networking.k8s.io/v1",
                    kind="Ingress",
                    name="main-route",
                ),
            ),
            expected_type="Object",
            expected_top_key="object",
        ),
        dict(
            kwargs=dict(target_type="Value", target_value="42"),
            expected_target={"type": "Value", "value": "42"},
        ),
    )
    def test_build_spec(
        self,
        kwargs,
        expected=None,
        expected_selector=None,
        expected_type=None,
        expected_top_key=None,
        expected_target=None,
    ):
        spec = _metric(**kwargs).build_spec()
        if expected is not None:
            self.assertEqual(spec, expected)
        if expected_selector is not None:
            self.assertEqual(spec["pods"]["metric"]["selector"], expected_selector)
        if expected_type is not None:
            self.assertEqual(spec["type"], expected_type)
        if expected_top_key is not None:
            self.assertIn(expected_top_key, spec)
        if expected_target is not None:
            self.assertEqual(spec["pods"]["target"], expected_target)

    def test_build_spec_unsupported_type_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported HPA metric type"):
            _metric(type="Bogus").build_spec()

    def test_object_includes_described_object(self):
        spec = _metric(
            type="Object",
            described_object=DescribedObject.default_config().set(
                api_version="networking.k8s.io/v1",
                kind="Ingress",
                name="main-route",
            ),
        ).build_spec()
        self.assertEqual(
            spec["object"]["describedObject"],
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "name": "main-route",
            },
        )

    def test_object_missing_described_object_raises(self):
        with self.assertRaisesRegex(ValueError, "described_object"):
            _metric(type="Object").build_spec()


class HPAFromFlagsTest(parameterized.TestCase):
    """Tests for HPA.from_flags."""

    def test_returns_prebuilt_when_no_metric_name(self):
        cfg = HPA.from_flags(_hpa_flags(), prebuilt_cfg=_prebuilt_hpa_cfg())
        self.assertEqual(cfg.metrics, [])

    def test_populates_metrics_and_bounds(self):
        fv = _hpa_flags(
            scaling_metric_name=["m1", "m2"],
            scaling_metric_target_value=["100", "200"],
            scaling_metric_type=["Pods", "External"],
            scaling_scale_up_stabilization_seconds=60,
        )
        cfg = HPA.from_flags(fv, prebuilt_cfg=_prebuilt_hpa_cfg())
        self.assertEqual(len(cfg.metrics), 2)
        self.assertEqual(cfg.metrics[0].type, "Pods")
        self.assertEqual(cfg.metrics[0].metric_name, "m1")
        self.assertEqual(cfg.metrics[0].target_value, "100")
        self.assertEqual(cfg.metrics[1].type, "External")
        self.assertEqual(cfg.metrics[1].metric_name, "m2")
        self.assertEqual(cfg.scale_up_stabilization_seconds, 60)
        # Metric names are passed through verbatim — the runner is
        # responsible for adapter-specific transforms.
        self.assertNotIn("|", cfg.metrics[0].metric_name)

    @parameterized.parameters(
        dict(
            overrides=dict(
                scaling_metric_name=["m1", "m2"],
                scaling_metric_target_value=["100"],
            ),
            expected_regex="target_value count",
        ),
        dict(
            overrides=dict(
                scaling_metric_name=["m1", "m2"],
                scaling_metric_target_value=["100", "200"],
                scaling_metric_type=["Pods"],
            ),
            expected_regex="metric_type count",
        ),
        dict(
            overrides=dict(
                scaling_metric_name=["m1", "m2"],
                scaling_metric_target_value=["100", "200"],
                scaling_metric_target_type=["AverageValue"],
            ),
            expected_regex="target_type count",
        ),
    )
    def test_length_mismatch_raises(self, overrides, expected_regex):
        with self.assertRaisesRegex(ValueError, expected_regex):
            HPA.from_flags(_hpa_flags(**overrides), prebuilt_cfg=_prebuilt_hpa_cfg())
