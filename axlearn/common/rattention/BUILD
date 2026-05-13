load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "rattention",
    srcs = ["rattention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:ein_ops",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/common/rattention/kernels:linear_attention_kernels",
        "//axlearn/common/rattention/kernels:utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "rattention_test",
    size = "enormous",
    srcs = ["rattention_test.py"],
    shard_count = 16,
    deps = [
        ":rattention",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/common/rattention/kernels:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)
