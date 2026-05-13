load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "manager_pb2",
    srcs = [
        "manager_pb2.py",
        "manager_pb2_grpc.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//grpcio",
        "@axlearn_pip//protobuf",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":manager_pb2",
        "@axlearn_pip//protobuf",
    ],
)

py_library(
    name = "manager_client",
    srcs = ["manager_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":manager_pb2",
        ":utils",
        "@axlearn_pip//grpcio",
    ],
)

py_library(
    name = "manager_server",
    srcs = ["manager_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":manager_client",
        ":manager_pb2",
        ":utils",
        "@axlearn_pip//grpcio",
    ],
)

py_library(
    name = "manager",
    srcs = ["manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":manager_client",
        ":manager_pb2",
        ":manager_server",
        ":utils",
    ],
)

py_library(
    name = "monitor",
    srcs = ["monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":manager",
        ":utils",
    ],
)

py_binary(
    name = "agent",
    srcs = ["agent.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":monitor",
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)
