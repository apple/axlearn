load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ctc_aligner",
    srcs = ["ctc_aligner.py"],
    deps = [
        "//axlearn/common:ein_ops",
        "//axlearn/common:module",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "ctc_aligner_test",
    size = "large",
    srcs = ["ctc_aligner_test.py"],
    deps = [
        ":ctc_aligner",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "ctc_aligner_test_lib",
    testonly = True,
    srcs = ["ctc_aligner_test.py"],
    visibility = ["//axlearn:__subpackages__"],
    deps = [
        ":ctc_aligner",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)
