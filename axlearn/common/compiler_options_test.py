# Copyright © 2024 Apple Inc.
"""Tests for compiler_options.py."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import compiler_options, test_utils
from axlearn.common.utils import Tensor


class CompilerOptionsTest(test_utils.TestCase):
    def setUp(self):
        self._original_env = dict(os.environ)

    def tearDown(self):
        # Reset environment variables.
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_default_xla_options(self):
        if jax.default_backend() != "tpu":
            self.skipTest("TPU-only test.")

        @jax.jit
        def f(x: Tensor) -> Tensor:
            return 3 * x

        f_compiled = f.lower(jnp.asarray(2)).compile(
            compiler_options=compiler_options.default_xla_options(
                instance_type="tpu-v5p-8", num_slices=1, backend="tpu"
            )
        )
        self.assertEqual(f_compiled(5), 15)

    @parameterized.parameters(
        dict(xla_option_override=None, expected_value="true"),
        dict(xla_option_override="", expected_value="true"),
        dict(
            xla_option_override="xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction=false",
            expected_value="false",
        ),
    )
    def test_default_xla_options_override(
        self,
        xla_option_override: Optional[str],
        expected_value: str,
    ):
        if xla_option_override is not None:
            os.environ["XLA_OPTIONS_OVERRIDE"] = xla_option_override
        xla_options = compiler_options.default_xla_options(
            instance_type="tpu-v6e-32",
            num_slices=2,
            backend="tpu",
        )
        self.assertEqual(
            xla_options["xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction"],
            expected_value,
        )

    @parameterized.parameters(
        dict(xla_option_override="megascale_graph_hang_threshold=60m", expected_value="60m"),
        dict(xla_option_override="megascale_graph_hang_threshold=300s", expected_value="300s"),
    )
    def test_default_xla_options_override_time_values(
        self,
        xla_option_override: str,
        expected_value: str,
    ):
        os.environ["XLA_OPTIONS_OVERRIDE"] = xla_option_override
        xla_options = compiler_options.default_xla_options(
            instance_type="tpu-v6e-32",
            num_slices=2,
            backend="tpu",
        )
        self.assertEqual(
            xla_options["megascale_graph_hang_threshold"],
            expected_value,
        )

    def test_xla_flags_from_options(self):
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
            xla_tpu_ici_sdc_test_run_on_program_start=False,
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
        dict(tpu_type="v6e-8-1", expected="v6e"),
    )
    def test_tpu_version_alias(self, tpu_type: str, expected: str):
        self.assertEqual(expected, compiler_options.infer_tpu_version(tpu_type))

    def test_xla_performance_flags(self):
        def sc_offload_enabled(flags: dict[str, str]) -> bool:
            return flags.get("xla_tpu_enable_sparse_core_collective_offload_all_gather") == "true"

        self.assertFalse(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[4, 4], mesh_axis_names=("data", "fsdp"), device_kind="TPU v6 lite"
                )
            ),
        )
        self.assertTrue(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[64, 4], mesh_axis_names=("fsdp", "model"), device_kind="TPU v6 lite"
                )
            )
        )
        self.assertTrue(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[32, 8], mesh_axis_names=("fsdp", "model"), device_kind="TPU v6 lite"
                )
            ),
        )
        self.assertFalse(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[64, 4], mesh_axis_names=("data", "fsdp"), device_kind="TPU v5p"
                )
            ),
        )
        self.assertTrue(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[32, 8, 1],
                    mesh_axis_names=("fsdp", "track", "model"),
                    device_kind="TPU v6 lite",
                )
            ),
        )
        self.assertTrue(
            sc_offload_enabled(
                compiler_options.infer_xla_performance_flags(
                    mesh_shape=[16, 8, 1],
                    mesh_axis_names=("fsdp", "track", "model"),
                    device_kind="TPU v6 lite",
                )
            ),
        )


if __name__ == "__main__":
    absltest.main()
