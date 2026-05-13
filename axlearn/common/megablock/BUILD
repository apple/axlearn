load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "megablock",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":gmm_gpu",
        ":ops",
    ],
)

py_library(
    name = "gmm_gpu",
    srcs = ["gmm_gpu.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":gmm_gpu",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "gmm_gpu_test",
    srcs = ["gmm_gpu_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":gmm_gpu",
        ":ops",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)
