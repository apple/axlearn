load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "quantized_dot_general",
    srcs = [
        "__init__.py",
        "activation_clipping.py",
        "layers.py",
        "utils.py",
    ],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//aqtp",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        ":activation_clipping",
        ":utils",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//aqtp",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@axlearn_pip//aqtp",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "activation_clipping",
    srcs = ["activation_clipping.py"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "fp8_ops",
    srcs = ["fp8_ops.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//flax",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "layers_test",
    size = "small",
    srcs = ["layers_test.py"],
    deps = [
        ":activation_clipping",
        ":layers",
        "//axlearn/common:decoder",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//aqtp",
        "@axlearn_pip//jax",
    ],
)
