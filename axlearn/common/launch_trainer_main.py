# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import sys

from absl import app, flags, logging

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function


def main(_):
    logging.info("Command line: %s", " ".join(sys.argv))
    measurement.initialize(flags.FLAGS)
    launch.setup()
    trainer_config = launch_trainer.get_trainer_config()
    trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
    launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
