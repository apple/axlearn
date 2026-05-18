load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "bastion_vm",
    srcs = ["bastion_vm.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:quota",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "launch_utils",
    srcs = ["launch_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:job_types",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "logs",
    srcs = ["logs.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_cloud_logging",
    ],
)

py_test(
    name = "bastion_vm_test",
    srcs = ["bastion_vm_test.py"],
    deps = [
        ":bastion_vm",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "launch",
    srcs = ["launch.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bastion_vm",
        ":launch_utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:job_types",
        "//axlearn/cloud/common:quota",
        "//axlearn/cloud/common:scheduler",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/cloud/gcp/runners:__init__",
        "//axlearn/cloud/gcp/runners:base",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//psutil",
    ],
)

py_library(
    name = "cpu_runner",
    srcs = ["cpu_runner.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/cloud/gcp:vm",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "dataflow",
    srcs = ["dataflow.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:docker",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_api_python_client",
        "@axlearn_pip//google_auth",
    ],
)

py_library(
    name = "cpu_runner_test_lib",
    testonly = True,
    srcs = ["cpu_runner_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpu_runner",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/cloud/gcp:vm",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "logs_test",
    srcs = ["logs_test.py"],
    deps = [
        ":logs",
        "//axlearn/cloud",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "cpu_runner_test",
    srcs = ["cpu_runner_test.py"],
    deps = [
        ":cpu_runner",
        "//axlearn/cloud",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/cloud/gcp:vm",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "dataflow_test",
    srcs = ["dataflow_test.py"],
    deps = [
        ":cpu_runner",
        ":cpu_runner_test_lib",
        ":dataflow",
        "//axlearn/cloud",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "launch_test",
    srcs = ["launch_test.py"],
    deps = [
        ":bastion_vm",
        ":launch",
        ":launch_utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:job_types",
        "//axlearn/cloud/common:scheduler",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:node_pool_provisioner",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/cloud/gcp/runners:__init__",
        "//axlearn/cloud/gcp/runners:base",
        "//axlearn/cloud/gcp/runners:gke",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "launch_utils_test",
    srcs = ["launch_utils_test.py"],
    deps = [
        ":launch",
        ":launch_utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:job",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)
