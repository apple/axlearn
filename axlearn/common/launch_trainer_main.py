# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

from absl import app, flags

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function
from pathwaysutils.elastic import manager


def main(_):
    elastic_manager = manager.Manager()
    while True:
        try:
            measurement.initialize(flags.FLAGS)
            launch.setup()
            trainer_config = launch_trainer.get_trainer_config()
            trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
            measurement.start_monitoring()
            launch_trainer.run_trainer(trainer_config)
            break
        except jax.errors.JaxRuntimeError as error:
          if not elastic_manager._is_error_due_to_slice_down(error):
            raise
          ten_minutes = 10 * 60
          elastic_manager.wait_for_slices(timeout=ten_minutes)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
