# Copyright © 2024 Apple Inc.
"""Tests for compiler_options.py."""

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common import compiler_options, test_utils
from axlearn.common.utils import Tensor


class CompilerOptionsTest(test_utils.TestCase):
    @pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU-only test.")
    def test_default_xla_options(self):
        @jax.jit
        def f(x: Tensor) -> Tensor:
            return 3 * x

        f_compiled = f.lower(jnp.asarray(2)).compile(
            compiler_options=compiler_options.default_xla_options(
                instance_type="tpu-v5p-8", num_slices=1, backend="tpu"
            )
        )
        self.assertEqual(f_compiled(5), 15)

    def atest_xla_flags_from_options(self):
        options = dict(a="true", b="false", c=True, d=False, long_option_name=True)
        result = compiler_options.xla_flags_from_options(options)
        self.assertEqual(result, "--a=true --b=false --c=1 --d=0 --long_option_name=1")

    def test_xsc_compiler_options(self):
        options = compiler_options.infer_xsc_compiler_options(
            halt_on_detection=False, repeat_count=2, device_kind="TPU v6e"
        )
        expected_options = dict(
            xla_tpu_enable_sdc_checker=True,
            xla_tpu_sdc_check_repeat_count=2,
            xla_tpu_sdc_check_halt_on_detection=False,
            xla_tpu_sdc_replicate_llo=True,
            xla_tpu_sdc_checker_alternate_megacore_cores=True,
            xla_tpu_ici_sdc_test_run_on_program_start=True,
            xla_tpu_ici_sdc_test_max_distance=1,
            xla_tpu_ici_sdc_test_pipeline_depth=4,
            xla_tpu_ici_sdc_test_buffer_size_chunks=32,
            xla_tpu_ici_sdc_test_packet_size_chunks=4,
            xla_tpu_ici_sdc_test_iterations=10,
            xla_tpu_enable_log_recorder=False,
        )
        for name, option in options.items():
            self.assertEqual(option, expected_options[name])

    @parameterized.parameters(
        dict(tpu_type="v5e-16", expected="v5litepod"),
    )
    def test_tpu_version_alias(self, tpu_type: str, expected: str):
        self.assertEqual(expected, compiler_options.infer_tpu_version(tpu_type))

    def test_xla_performance_flags(self):
        self.assertEqual(
            {},
            compiler_options.infer_xla_performance_flags(
                mesh_shape=[4, 4], mesh_axis_names=("data", "fsdp"), device_kind="TPU v6 lite"
            ),
        )
        self.assertNotEqual(
            {},
            compiler_options.infer_xla_performance_flags(
                mesh_shape=[64, 4], mesh_axis_names=("fsdp", "model"), device_kind="TPU v6 lite"
            ),
        )
        self.assertNotEqual(
            {},
            compiler_options.infer_xla_performance_flags(
                mesh_shape=[32, 8], mesh_axis_names=("fsdp", "model"), device_kind="TPU v6 lite"
            ),
        )
        self.assertEqual(
            {},
            compiler_options.infer_xla_performance_flags(
                mesh_shape=[64, 4], mesh_axis_names=("data", "fsdp"), device_kind="TPU v5p"
            ),
        )
