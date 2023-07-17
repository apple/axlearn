# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""
from absl import app

from axlearn.common import launch, launch_trainer


def main(_):
    launch.setup()
    trainer_config = launch_trainer.get_trainer_config()
    launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    app.run(main)
