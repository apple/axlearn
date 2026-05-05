# Copyright © 2025 Apple Inc.

"""Tests for the ReadinessProbe and LivenessProbe classes."""

from absl import flags
from absl.testing import absltest

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp.k8s_readiness_probe import LivenessProbe, ReadinessProbe


def _make_probe(**kwargs) -> ReadinessProbe:
    return ReadinessProbe.default_config().set(**kwargs).instantiate()


def _make_liveness_probe(**kwargs) -> LivenessProbe:
    return LivenessProbe.default_config().set(**kwargs).instantiate()


def _flags_config(**flag_overrides) -> ReadinessProbe.Config:
    fv = flags.FlagValues()
    cfg = ReadinessProbe.default_config()
    define_flags(cfg, fv)
    for key, value in flag_overrides.items():
        fv[key].value = value
    fv.mark_as_parsed()
    return from_flags(cfg, fv)


class BuildReadinessProbeTest(absltest.TestCase):
    def test_grpc_port_only(self):
        self.assertEqual(
            _make_probe(grpc_port=8080).build_readiness_probe(), {"grpc": {"port": 8080}}
        )

    def test_grpc_with_service(self):
        self.assertEqual(
            _make_probe(grpc_port=8080, grpc_service="svc").build_readiness_probe(),
            {"grpc": {"port": 8080, "service": "svc"}},
        )

    def test_grpc_all_fields(self):
        result = _make_probe(
            grpc_port=8080,
            grpc_service="my-svc",
            initial_delay_seconds=5,
            period_seconds=10,
            timeout_seconds=3,
            success_threshold=1,
            failure_threshold=3,
        ).build_readiness_probe()
        self.assertEqual(
            result,
            {
                "grpc": {"port": 8080, "service": "my-svc"},
                "initialDelaySeconds": 5,
                "periodSeconds": 10,
                "timeoutSeconds": 3,
                "successThreshold": 1,
                "failureThreshold": 3,
            },
        )

    def test_http_port_only(self):
        self.assertEqual(
            _make_probe(http_port=8080).build_readiness_probe(), {"httpGet": {"port": 8080}}
        )

    def test_http_all_fields(self):
        result = _make_probe(
            http_port=9090,
            http_path="/healthz",
            http_scheme="HTTPS",
            initial_delay_seconds=10,
            period_seconds=15,
            timeout_seconds=5,
            success_threshold=2,
            failure_threshold=4,
        ).build_readiness_probe()
        self.assertEqual(
            result,
            {
                "httpGet": {"port": 9090, "path": "/healthz", "scheme": "HTTPS"},
                "initialDelaySeconds": 10,
                "periodSeconds": 15,
                "timeoutSeconds": 5,
                "successThreshold": 2,
                "failureThreshold": 4,
            },
        )

    def test_none_fields_omitted(self):
        result = _make_probe(grpc_port=8080, period_seconds=10).build_readiness_probe()
        self.assertIn("periodSeconds", result)
        self.assertNotIn("initialDelaySeconds", result)
        self.assertNotIn("timeoutSeconds", result)
        self.assertNotIn("successThreshold", result)
        self.assertNotIn("failureThreshold", result)

    def test_both_ports_raises(self):
        with self.assertRaises(ValueError):
            _make_probe(grpc_port=8080, http_port=8080).build_readiness_probe()

    def test_neither_port_raises(self):
        with self.assertRaises(ValueError):
            _make_probe().build_readiness_probe()


class IsConfiguredTest(absltest.TestCase):
    def test_grpc(self):
        self.assertTrue(_make_probe(grpc_port=8080).is_configured())

    def test_http(self):
        self.assertTrue(_make_probe(http_port=9000).is_configured())

    def test_none(self):
        self.assertFalse(_make_probe().is_configured())


class DefineFlagsTest(absltest.TestCase):
    def test_registers_without_error(self):
        fv = flags.FlagValues()
        cfg = ReadinessProbe.default_config()
        define_flags(cfg, fv)
        fv.mark_as_parsed()
        self.assertIn("readiness_probe_grpc_port", fv)
        self.assertIn("readiness_probe_http_port", fv)
        self.assertIn("readiness_probe_http_path", fv)

    def test_from_flags_grpc_port(self):
        self.assertEqual(_flags_config(readiness_probe_grpc_port=8080).grpc_port, 8080)

    def test_from_flags_http_fields(self):
        cfg = _flags_config(readiness_probe_http_port=9000, readiness_probe_http_path="/health")
        self.assertEqual(cfg.http_port, 9000)
        self.assertEqual(cfg.http_path, "/health")

    def test_from_flags_all_none(self):
        cfg = _flags_config()
        self.assertIsNone(cfg.grpc_port)
        self.assertIsNone(cfg.grpc_service)
        self.assertIsNone(cfg.http_port)
        self.assertIsNone(cfg.http_path)
        self.assertIsNone(cfg.http_scheme)
        self.assertIsNone(cfg.initial_delay_seconds)
        self.assertIsNone(cfg.period_seconds)
        self.assertIsNone(cfg.timeout_seconds)
        self.assertIsNone(cfg.success_threshold)
        self.assertIsNone(cfg.failure_threshold)


# ===========================================================================
# LivenessProbe tests
# ===========================================================================


def _liveness_flags_config(**flag_overrides) -> LivenessProbe.Config:
    fv = flags.FlagValues()
    cfg = LivenessProbe.default_config()
    define_flags(cfg, fv)
    for key, value in flag_overrides.items():
        fv[key].value = value
    fv.mark_as_parsed()
    return from_flags(cfg, fv)


class BuildLivenessProbeTest(absltest.TestCase):
    def test_grpc_port_only(self):
        self.assertEqual(
            _make_liveness_probe(grpc_port=8080).build_liveness_probe(), {"grpc": {"port": 8080}}
        )

    def test_grpc_with_service(self):
        self.assertEqual(
            _make_liveness_probe(grpc_port=8080, grpc_service="svc").build_liveness_probe(),
            {"grpc": {"port": 8080, "service": "svc"}},
        )

    def test_grpc_all_fields(self):
        result = _make_liveness_probe(
            grpc_port=8080,
            grpc_service="my-svc",
            initial_delay_seconds=5,
            period_seconds=10,
            timeout_seconds=3,
            success_threshold=1,
            failure_threshold=3,
        ).build_liveness_probe()
        self.assertEqual(
            result,
            {
                "grpc": {"port": 8080, "service": "my-svc"},
                "initialDelaySeconds": 5,
                "periodSeconds": 10,
                "timeoutSeconds": 3,
                "successThreshold": 1,
                "failureThreshold": 3,
            },
        )

    def test_http_port_only(self):
        self.assertEqual(
            _make_liveness_probe(http_port=8080).build_liveness_probe(),
            {"httpGet": {"port": 8080}},
        )

    def test_http_all_fields(self):
        result = _make_liveness_probe(
            http_port=9090,
            http_path="/healthz",
            http_scheme="HTTPS",
            initial_delay_seconds=10,
            period_seconds=15,
            timeout_seconds=5,
            success_threshold=2,
            failure_threshold=4,
        ).build_liveness_probe()
        self.assertEqual(
            result,
            {
                "httpGet": {"port": 9090, "path": "/healthz", "scheme": "HTTPS"},
                "initialDelaySeconds": 10,
                "periodSeconds": 15,
                "timeoutSeconds": 5,
                "successThreshold": 2,
                "failureThreshold": 4,
            },
        )

    def test_none_fields_omitted(self):
        result = _make_liveness_probe(grpc_port=8080, period_seconds=10).build_liveness_probe()
        self.assertIn("periodSeconds", result)
        self.assertNotIn("initialDelaySeconds", result)
        self.assertNotIn("timeoutSeconds", result)
        self.assertNotIn("successThreshold", result)
        self.assertNotIn("failureThreshold", result)

    def test_both_ports_raises(self):
        with self.assertRaises(ValueError):
            _make_liveness_probe(grpc_port=8080, http_port=8080).build_liveness_probe()

    def test_neither_port_raises(self):
        with self.assertRaises(ValueError):
            _make_liveness_probe().build_liveness_probe()


class IsConfiguredLivenessTest(absltest.TestCase):
    def test_grpc(self):
        self.assertTrue(_make_liveness_probe(grpc_port=8080).is_configured())

    def test_http(self):
        self.assertTrue(_make_liveness_probe(http_port=9000).is_configured())

    def test_none(self):
        self.assertFalse(_make_liveness_probe().is_configured())


class DefineFlagsLivenessTest(absltest.TestCase):
    def test_registers_without_error(self):
        fv = flags.FlagValues()
        cfg = LivenessProbe.default_config()
        define_flags(cfg, fv)
        fv.mark_as_parsed()
        self.assertIn("liveness_probe_grpc_port", fv)
        self.assertIn("liveness_probe_http_port", fv)
        self.assertIn("liveness_probe_http_path", fv)

    def test_from_flags_grpc_port(self):
        self.assertEqual(_liveness_flags_config(liveness_probe_grpc_port=8080).grpc_port, 8080)

    def test_from_flags_http_fields(self):
        cfg = _liveness_flags_config(
            liveness_probe_http_port=9000, liveness_probe_http_path="/health"
        )
        self.assertEqual(cfg.http_port, 9000)
        self.assertEqual(cfg.http_path, "/health")

    def test_from_flags_all_none(self):
        cfg = _liveness_flags_config()
        self.assertIsNone(cfg.grpc_port)
        self.assertIsNone(cfg.http_port)
        self.assertIsNone(cfg.http_path)
        self.assertIsNone(cfg.initial_delay_seconds)
        self.assertIsNone(cfg.failure_threshold)


if __name__ == "__main__":
    absltest.main()
