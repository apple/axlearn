# Copyright © 2023 Apple Inc.

"""A command-line tool to perform AoT (ahead-of-time) compilation.

pip install 'jax[tpu]==0.4.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

XLA_FLAGS=--xla_dump_to=/tmp/aot_xla_dump \
python axlearn/experiments/run_aot_compilation.py \
    --module=axlearn.experiments.text.gpt.c4_trainer \
    --config=fuji-7B \
    --topology=v4-1024 1> /tmp/aot_stdout

Reference: https://jax.readthedocs.io/en/latest/aot.html
"""

import pickle
from typing import Optional

from absl import app, flags, logging
from jax.experimental.serialize_executable import serialize

from axlearn.common.aot_compilation import compile_trainer_programs
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import set_data_dir
from axlearn.common.utils_spmd import setup
from axlearn.experiments import TrainerConfigFn, get_named_trainer_config

flags.DEFINE_string("module", None, "The trainer config module.", required=True)
flags.DEFINE_string("config", None, "The trainer config name.", required=True)
flags.DEFINE_string("topology", None, "The TPU topology.")
flags.DEFINE_integer("topology_num_slices", 1, "The number of TPU slices.")

FLAGS = flags.FLAGS


def _compile_and_dump_programs(
    trainer_config: SpmdTrainer.Config,
    *,
    compile_topology: Optional[str],
    compile_topology_num_slices: int = 1,
):
    with set_data_dir("FAKE"):
        programs = compile_trainer_programs(
            trainer_config,
            topology=compile_topology,
            topology_num_slices=compile_topology_num_slices,
        )
    for program_name, program in programs.items():
        print(f"== Text: {program_name} ==")
        print(program.as_text())
        print(f"== Cost analysis {program_name} ==")
        print(program.cost_analysis())
        print(f"== Memeory analysis {program_name} ==")
        print(program.memory_analysis())

        # Serialization does not work for CPU devices:
        #     UNIMPLEMENTED: Not an XLA Runtime executable
        if compile_topology is not None:
            serialized_compiled, _, _ = serialize(program)
            serialized_compiled_output_path = f"/tmp/aot_compiled_{program_name}"
            with open(serialized_compiled_output_path, "wb") as f:
                pickle.dump(serialized_compiled, f)
            logging.info("Wrote serialized %s to %s", program_name, serialized_compiled_output_path)


def main(_):
    setup(jax_backend="cpu")
    trainer_config_fn: TrainerConfigFn = get_named_trainer_config(
        FLAGS.config,
        config_module=FLAGS.module,
    )
    _compile_and_dump_programs(
        trainer_config_fn(),
        compile_topology=FLAGS.topology,
        compile_topology_num_slices=FLAGS.topology_num_slices,
    )


if __name__ == "__main__":
    app.run(main)
