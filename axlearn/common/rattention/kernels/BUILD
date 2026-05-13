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
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "linear_attention_kernels",
    srcs = ["linear_attention_kernels.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//axlearn/common:utils",
        "@axlearn_pip//einops",
        "@axlearn_pip//jax",
    ],
)

# manual: requires TPU backend — pytest.skip() at module level if not TPU.
py_test(
    name = "linear_attention_kernels_test",
    srcs = ["linear_attention_kernels_test.py"],
    tags = ["manual"],
    deps = [
        ":linear_attention_kernels",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)
