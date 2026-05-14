exports_files(
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "golden_files",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "testdata",
    srcs = glob(["**/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//axlearn/common:measurement"],
)
