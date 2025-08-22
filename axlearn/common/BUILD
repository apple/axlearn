load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//attrs",
    ],
)

py_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//numpy",
        "@pip//pytest",
        "@pip//wrapt",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":param_init",
    ],
)

py_library(
    name = "param_init",
    srcs = ["param_init.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
    ],
)

py_library(
    name = "state_builder",
    srcs = ["state_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//jax",
        "@pip//numpy",
    ],
)

py_library(
    name = "module",
    srcs = ["module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":struct",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//jax",
    ],
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
        ":struct",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pytest",
        "@pip//optax",
    ],
)

py_library(
    name = "struct",
    srcs = ["struct.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":serialization",
        ":utils",
        "@pip//jax",
        "@pip//typing_extensions",
    ],
)

py_test(
    name = "struct_test",
    size = "small",
    srcs = ["struct_test.py"],
    deps = [
        ":struct",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//jax",
        "@pip//jaxlib",
        "@pip//pytest",
    ],
)
