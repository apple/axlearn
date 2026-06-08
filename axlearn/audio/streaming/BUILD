load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "streaming_base",
    srcs = ["streaming_base.py"],
    deps = [
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "convolutions",
    srcs = ["convolutions.py"],
    deps = [
        ":streaming_base",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:ein_ops",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        ":streaming_base",
        "//axlearn/common:ein_ops",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "streaming_base_test",
    srcs = ["streaming_base_test.py"],
    deps = [
        ":streaming_base",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "convolutions_test",
    srcs = ["convolutions_test.py"],
    deps = [
        ":convolutions",
        ":test_utils",
        "//axlearn/common:convolution",
        "//axlearn/common:ein_ops",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)
