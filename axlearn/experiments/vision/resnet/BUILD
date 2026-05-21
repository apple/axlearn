load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//:golden_config.bzl", "golden_config_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":imagenet_trainer",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:learner",
        "//axlearn/common:optimizers",
        "//axlearn/common:param_init",
        "//axlearn/common:schedule",
        "//axlearn/vision:image_classification",
    ],
)

py_library(
    name = "imagenet_trainer",
    srcs = ["imagenet_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:schedule",
        "//axlearn/common:trainer",
        "//axlearn/experiments/vision/imagenet:__init__",
        "//axlearn/vision:resnet",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        "//axlearn/common:config",
        "//axlearn/common:schedule",
        "//axlearn/common:test_utils",
        "//axlearn/vision:resnet",
        "@axlearn_pip//absl_py",
    ],
)

golden_config_test(
    name = "imagenet_trainer_golden_config_test",
    checks = [
        "check",
        "check_init",
        "check_regularizer",
    ],
    data = ["//axlearn/experiments/testdata:golden_files"],
    module = ":imagenet_trainer",
)

py_test(
    name = "imagenet_trainer_test",
    srcs = ["imagenet_trainer_test.py"],
    deps = [
        ":imagenet_trainer",
        "//axlearn/experiments:test_utils",
    ],
)
