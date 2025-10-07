# Copyright © 2023 Apple Inc.

"""A command-line tool to perform AoT (ahead-of-time) compilation on CPU using the JAX TPU library.

And it prints useful information.
Note: jax[tpu] doesn't support MacOS (as of 2025/03/05).
Note: If --topology=cpu-<digit> (e.g. cpu-1024) is used, installing the JAX TPU library is not
    required. AOT is performed using the JAX CPU library instead.

pip install 'jax[tpu]==0.4.28' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

XLA_FLAGS=--xla_dump_to=/tmp/aot_xla_dump \
python -m axlearn.experiments.run_aot_compilation \
    --module=axlearn.experiments.text.gpt.c4_trainer \
    --config=fuji-1B-v3 \
    --topology=v4-1024 1> /tmp/aot_stdout

For CPU fallback,
XLA_FLAGS=--xla_dump_to=/tmp/aot_xla_dump \
python -m axlearn.experiments.run_aot_compilation \
    --module=axlearn.experiments.text.gpt.c4_trainer \
    --config=fuji-1B-v3 \
    --topology=cpu-1024 1> /tmp/aot_stdout

For TPU slices,
python -m axlearn.experiments.run_aot_compilation \
    --module=axlearn.experiments.text.gpt.c4_trainer \
    --config=fuji-1B-v3 \
    --topology=v5e-256 --topology_num_slices=4 1> /tmp/aot_stdout

Reference: https://jax.readthedocs.io/en/latest/aot.html
"""

import pickle
from typing import Optional

import chex
import jax
from absl import app, flags, logging
from jax.experimental.serialize_executable import serialize

from axlearn.common import aot_compilation, compiler_options
from axlearn.common.config import TrainerConfigFn, get_named_trainer_config
from axlearn.common.trainer import SpmdTrainer, aot_model_analysis, select_mesh_config
from axlearn.common.utils import set_data_dir
from axlearn.common.utils_spmd import setup

flags.DEFINE_string("module", None, "The trainer config module.", required=True)
flags.DEFINE_string("config", None, "The trainer config name.", required=True)
flags.DEFINE_string("topology", None, "The TPU topology.", required=True)
flags.DEFINE_integer("topology_num_slices", 1, "The number of TPU slices.")
flags.DEFINE_string(
    "data_dir", "FAKE", "Sets the environment variable `DATA_DIR` to the given `data_dir`."
)

FLAGS = flags.FLAGS


def _mesh_selector(topology: str) -> str:
    slice_type = topology.replace("v5e", "v5litepod")
    return f"tpu-{slice_type}"


def _compile_and_dump_programs(
    trainer_config: SpmdTrainer.Config,
    *,
    compile_topology: Optional[str],
    compile_topology_num_slices: int = 1,
):
    if compile_topology is not None:
        xla_options = compiler_options.default_xla_options(
            instance_type=f"tpu-{compile_topology}",
            num_slices=compile_topology_num_slices,
            backend="tpu",
        )
    else:
        xla_options = None

    # Remove XLA options that are not supported by JAX 0.4.38.
    # TODO(kelvin-zou): Remove this when we upgrade JAX.
    if xla_options is not None:
        xla_options.pop("megascale_graph_within_launch_hang_threshold", None)
        xla_options.pop("megascale_graph_hang_threshold", None)
        xla_options.pop("megascale_grpc_enable_xor_tracer", None)
        xla_options.pop("megascale_error_reporter_abort_on_hang", None)
        xla_options.pop("megascale_grpc_premap_memory_bytes", None)
        xla_options.pop("megascale_error_reporter_abort_on_error", None)

    programs = aot_compilation.compile_trainer_programs(
        trainer_config,
        topology=compile_topology,
        topology_num_slices=compile_topology_num_slices,
        compiler_options=xla_options,
    )
    for program_name, program in programs.items():
        print(f"== Text: {program_name} ==")
        print(program.as_text())
        print()
        print(aot_model_analysis(program))
        print()

        # Serialization does not work for CPU devices:
        #     UNIMPLEMENTED: Not an XLA Runtime executable
        if compile_topology is not None:
            serialized_compiled, _, _ = serialize(program)
            serialized_compiled_output_path = f"/tmp/aot_compiled_{program_name}"
            with open(serialized_compiled_output_path, "wb") as f:
                pickle.dump(serialized_compiled, f)
            logging.info("Wrote serialized %s to %s", program_name, serialized_compiled_output_path)


def main(_):
    with set_data_dir(FLAGS.data_dir):
        setup(jax_backend="cpu")
        trainer_config_fn: TrainerConfigFn = get_named_trainer_config(
            FLAGS.config,
            config_module=FLAGS.module,
        )
        cfg: SpmdTrainer.Config = trainer_config_fn()
        if FLAGS.topology.startswith("cpu-"):
            jax.config.update("jax_threefry_partitionable", True)
            n_cpus = FLAGS.topology.split("-")[1]
            if not n_cpus.isdigit():
                raise ValueError(f"{FLAGS.topology} must be `cpu-digit` format for CPU fallback.")
            chex.set_n_cpu_devices(int(n_cpus))
            compile_topology = None
        else:
            select_mesh_config(cfg, mesh_selector=_mesh_selector(FLAGS.topology))
            compile_topology = FLAGS.topology
        _compile_and_dump_programs(
            cfg,
            compile_topology=compile_topology,
            compile_topology_num_slices=FLAGS.topology_num_slices,
        )


if __name__ == "__main__":
    app.run(main)
