load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "tpu",
    values = {"define": "tpu=true"},
)

py_library(
    name = "online_softmax_log_probs",
    srcs = ["online_softmax_log_probs.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "online_softmax_log_probs_test",
    srcs = ["online_softmax_log_probs_test.py"],
    deps = [
        ":online_softmax_log_probs",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_binary(
    name = "online_softmax_log_probs_benchmark",
    srcs = ["online_softmax_log_probs_benchmark.py"],
    deps = [
        ":online_softmax_log_probs",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ] + select({
        ":tpu": ["@axlearn_pip//libtpu"],
        "//conditions:default": [],
    }),
)
