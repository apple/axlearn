load("@rules_python//python:defs.bzl", "py_library")
load("//:golden_config.bzl", "golden_config_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "ax",
    srcs = ["ax.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:base_model",
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_grain",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:layers",
        "//axlearn/common:learner",
        "//axlearn/common:module",
        "//axlearn/common:optimizers",
        "//axlearn/common:schedule",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "tutorial",
    srcs = ["tutorial.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":ax",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

golden_config_test(
    name = "tutorial_golden_config_test",
    module = ":tutorial",
    data = ["//axlearn/experiments/testdata:golden_files"],
    checks = ["check", "check_init"],
)
