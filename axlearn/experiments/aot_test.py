# Copyright © 2023 Apple Inc.

"""AoT (ahead-of-time) compilation config tests.

pip install 'jax[tpu]==0.4.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

export TPU_SKIP_MDS_QUERY=1
python axlearn/experiments/aot_test.py

Reference:
https://docs.google.com/document/d/1Y5IdmvAZA7UtMHAWkRh8k2PscVoG5FvMH9-E6hygsyY/
"""
import pickle
from typing import Optional

from absl.testing import absltest
from jax.experimental.serialize_executable import serialize

from axlearn.common import test_utils
from axlearn.common.aot_compilation import compile_trainer_programs
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.text.gpt import c4_trainer


class AoTCompilationTest(test_utils.TrainerConfigTestCase):
    """Tests ahead-of-time (AoT) compilation."""

    def _jax_backend(self) -> Optional[str]:
        return "cpu"

    def _test_aot(
        self,
        trainer_config: SpmdTrainer.Config,
        *,
        compile_topology: Optional[str],
        compile_topology_num_slices: int = 1,
    ):
        programs = compile_trainer_programs(
            trainer_config,
            topology=compile_topology,
            topology_num_slices=compile_topology_num_slices,
        )
        compiled_train_step = programs["train_step"]
        self.assertIsNotNone(compiled_train_step)
        print("== Help ==")
        print(help(compiled_train_step))
        print("== Text ==")
        print(compiled_train_step.as_text())
        print("== Cost analysis ==")
        print(compiled_train_step.cost_analysis())
        print("== Memeory analysis ==")
        print(compiled_train_step.memory_analysis())

        # Serialization does not work for CPU devices:
        #     UNIMPLEMENTED: Not an XLA Runtime executable
        if compile_topology is not None:
            serialized_compiled, in_tree, out_tree = serialize(compiled_train_step)
            with open("/tmp/aot_compiled", "wb") as f:
                pickle.dump(serialized_compiled, f)
            print(serialized_compiled)

    def test_fuji_7B(self):
        self._test_aot(
            c4_trainer.named_trainer_configs()["fuji-7B"](),
            compile_topology=None,
        )


if __name__ == "__main__":
    absltest.main()
