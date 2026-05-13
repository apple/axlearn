load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "testdata_dummy",
    testonly = True,
    srcs = ["testdata/dummy.py"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":testdata_dummy",
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)
