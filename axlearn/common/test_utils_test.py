# Copyright © 2023 Apple Inc.

"""Tests for test_utils.py."""

import jax
from absl.testing import absltest, parameterized

from axlearn.common import test_utils


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
