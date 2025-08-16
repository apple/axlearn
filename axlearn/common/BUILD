load("//pytype_bazel:pytype.bzl", "pytype_library", "pytype_test")

pytype_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

pytype_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//attrs",
    ],
)

pytype_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pytest",
        "@pip//wrapt",
    ],
)

pytype_library(
    name = "layers",
    srcs = ["layers.py"],
    enable_pytype = False,  # Type annotation errors require code fixes
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":param_init",
    ],
)

pytype_library(
    name = "param_init",
    srcs = ["param_init.py"],
    enable_pytype = False,  # Type annotation errors require code fixes
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
    ],
)

pytype_library(
    name = "state_builder",
    srcs = ["state_builder.py"],
    enable_pytype = False,  # Missing imports and type annotation errors
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

pytype_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":serialization",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "module",
    srcs = ["module.py"],
    enable_pytype = False,  # Complex forward references and custom dataclass patterns
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":struct",
        ":summary",
        ":traceback_util",
        ":utils",
    ],
)

pytype_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//jax",
    ],
)

pytype_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    enable_pytype = False,  # Type error: No attribute 'replace' on _Box
    deps = [
        ":serialization",
        ":struct",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
        "@pip//optax",
        "@pip//pytest",
    ],
)

pytype_library(
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

pytype_test(
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

pytype_library(
    name = "summary",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":struct",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "traceback_util",
    srcs = ["traceback_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//absl_py",
        "@pip//jax",
    ],
)
