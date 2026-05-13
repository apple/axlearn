load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:utils",
        "@axlearn_pip//seqio",
    ],
)
