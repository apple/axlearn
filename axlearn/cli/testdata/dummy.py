# Copyright © 2023 Apple Inc.

"""A dummy absl program."""

from absl import app, flags

flags.DEFINE_string("required", None, "A required flag.", required=True)
flags.DEFINE_string("optional", None, "An optional flag.")
flags.DEFINE_string("root_default", None, "A required flag defaulted at a parent.", required=True)

FLAGS = flags.FLAGS


def main(_):
    print(
        f"required: {FLAGS.required}, optional: {FLAGS.optional}, "
        f"root_default: {FLAGS.root_default}"
    )


if __name__ == "__main__":
    app.run(main)
