# Copyright © 2023 Apple Inc.

"""Tests for test_utils.py."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import test_utils
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import ConfigOr
from axlearn.common.layers import Linear


class BindTest(test_utils.TestCase):
    """Tests trainer config utils."""

    LAYER_CFG = Linear.default_config().set(input_dim=5, output_dim=7)

    def test_bind_module(self):
        with test_utils.bind_module(
            self.LAYER_CFG, state={"weight": jnp.ones((5, 7)), "bias": jnp.ones(7)}
        ) as instantiated_layer:
            result = instantiated_layer(jnp.ones(5))  # pylint:disable=not-callable
        self.assertNestedAllClose(result, jnp.ones(7) * 6)

    @parameterized.parameters(LAYER_CFG, LAYER_CFG.clone(name="tmp").instantiate(parent=None))
    def test_bind_layer(self, layer: ConfigOr[BaseLayer]):
        with test_utils.bind_layer(layer) as instantiated_layer:
            result = instantiated_layer(jnp.ones(5))
        self.assertEqual(result.shape, (7,))

        with test_utils.bind_layer(
            layer, state={"weight": jnp.ones((5, 7)), "bias": jnp.ones(7)}
        ) as instantiated_layer:
            result = instantiated_layer(jnp.ones(5))
        self.assertNestedAllClose(result, jnp.ones(7) * 6)


class CleanHLOTest(parameterized.TestCase):
    """Tests trainer config utils."""

    def test_clean_hlo_real_hlo(self):
        @jax.jit
        def f():
            return 5

        hlo = f.lower().compile().as_text()
        hlo = test_utils.clean_hlo(hlo)
        self.assertNotIn("metadata", hlo)
        self.assertNotIn("source_file", hlo)
        self.assertNotIn("source_line", hlo)

    @parameterized.parameters(
        r"""metadata={op_name="jit(f)/jit(main)/mul" source_file="/my/f.py" source_line=15}""",
        r"""metadata={op_name="jit(f)/jit(main)/mul" source_file="/my/\"f.py" source_line=15}""",
        r"""metadata={op_name="jit(f)/jit(main)/mul" source_file="/my/}f.py" source_line=15}""",
        r"""metadata={op_name="jit(f)/jit(main)/mul" source_file="/my/}\"f.py" source_line=15}""",
    )
    def test_clean_hlo_regex(self, hlo: str):
        hlo = "before" + hlo + "after"
        hlo = test_utils.clean_hlo(hlo)
        self.assertEqual(hlo, "beforeafter")


if __name__ == "__main__":
    absltest.main()
