# Copyright © 2024 Apple Inc.

"""Tests orbax checkpointer.

See also checkpointer_test.py for common checkpointing tests.
"""

# pylint: disable=protected-access

import os
import tempfile
from typing import Sequence

import jax
import orbax.checkpoint as ocp
from jax import numpy as jnp
from jax.experimental import mesh_utils

from axlearn.common import test_utils
from axlearn.common.checkpointer import read_index_file
from axlearn.common.checkpointer_orbax import OrbaxCheckpointer


def _mesh(mesh_shape: Sequence[int]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ("data", "model"))


class OrbaxCheckpointerTest(test_utils.TestCase):
    def test_index(self):
        """Tests that index files saved with orbax can be read with `read_index_file`."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape), tempfile.TemporaryDirectory() as temp_dir:
            ckpt = (
                OrbaxCheckpointer.default_config()
                .set(name="test", dir=temp_dir)
                .instantiate(parent=None)
            )
            step = 123
            state = dict(x=jnp.ones([3, 2]))
            ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            ref_index = read_index_file(os.path.join(temp_dir, "step_00000123", "index"))
            test_index = ckpt._manager.restore(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(ckpt._get_spec(step=step, state=state))
                ),
            )
            self.assertEqual(ref_index, test_index["index"])
