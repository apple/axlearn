load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [":common"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:utils",
        "//axlearn/vision:input_image",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":__init__",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/experiments/vision:__init__",
        "@axlearn_pip//absl_py",
    ],
)
