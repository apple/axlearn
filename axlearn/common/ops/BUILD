load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_optimization_barrier",
    ],
)

py_library(
    name = "_optimization_barrier",
    srcs = ["_optimization_barrier.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "_optimization_barrier_test",
    srcs = ["_optimization_barrier_test.py"],
    deps = [
        ":__init__",
        ":_optimization_barrier",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)
