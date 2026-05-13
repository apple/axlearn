load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "tpu_client",
    srcs = ["tpu_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//grpcio",
        "@axlearn_pip//prometheus_client",
        "@axlearn_pip//tpu_info",
    ],
)

py_library(
    name = "tpu_device_monitor",
    srcs = ["tpu_device_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "//axlearn/common/monitoring:device_monitor",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//tpu_info",
    ],
)

py_library(
    name = "tpu_client_test_lib",
    testonly = True,
    srcs = ["tpu_client_test.py"],
    data = glob(["testdata/*"]),
    deps = [
        ":tpu_client",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//grpcio",
        "@axlearn_pip//tpu_info",
    ],
)

py_test(
    name = "tpu_client_test",
    srcs = ["tpu_client_test.py"],
    data = glob(["testdata/*"]),
    deps = [
        ":tpu_client",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//grpcio",
        "@axlearn_pip//tpu_info",
    ],
)

py_test(
    name = "tpu_device_monitor_test",
    srcs = ["tpu_device_monitor_test.py"],
    deps = [
        ":tpu_client",
        ":tpu_client_test_lib",
        ":tpu_device_monitor",
        "//axlearn/common/monitoring:device_monitor",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tpu_info",
    ],
)
