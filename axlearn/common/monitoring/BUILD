load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "device_monitor",
    srcs = ["device_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "gpu_client",
    srcs = ["gpu_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "gpu_device_monitor",
    srcs = ["gpu_device_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_monitor",
        ":gpu_client",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "device_monitor_test",
    srcs = ["device_monitor_test.py"],
    deps = [
        ":device_monitor",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "gpu_client_test",
    srcs = ["gpu_client_test.py"],
    # manual: @unittest.skipIf(jax.default_backend() != "gpu") — needs GPU. See #2385.
    tags = ["manual"],
    deps = [
        ":gpu_client",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "gpu_device_monitor_test",
    srcs = ["gpu_device_monitor_test.py"],
    # manual: @unittest.skipIf(jax.default_backend() != "gpu") — needs GPU. See #2385.
    tags = ["manual"],
    deps = [
        ":device_monitor",
        ":gpu_device_monitor",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)
