# Copyright © 2024 Apple Inc.

"""The main health check program to run.

This health check runs a matmul of a column-parallel matrix and a row-parallel matrix, followed
by an scalar add. An allreduce should be inserted by compiler to produce non-sharded output.
"""
import functools
import sys

import jax
import jax.distributed
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def main():
    device_count = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), axis_names=("i",))
    dim = device_count * 2
    x = jnp.eye(dim, dim, dtype=jnp.float32)
    y = jnp.eye(dim, dim, dtype=jnp.float32)

    @functools.partial(jax.jit, out_shardings=NamedSharding(mesh, P(None, None)))
    def jit_fn(x_dist, y_dist):
        return (x_dist @ y_dist) + 1

    return np.all(
        np.asarray(
            jit_fn(
                jax.device_put(x, NamedSharding(mesh, P(None, "i"))),
                jax.device_put(y, NamedSharding(mesh, P("i", None))),
            )
        )
        == np.eye(dim, dim, dtype=np.float32) + 1
    )


if __name__ == "__main__":
    jax.distributed.initialize()
    result = main()
    jax.distributed.shutdown()
    if not result:
        logging.error("TPU matmul produced unexpected result!")
        sys.exit(-1)
