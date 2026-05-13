load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":gke",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:job_flink",
        "//axlearn/cloud/gcp:job_pathways",
        "//axlearn/cloud/gcp:jobset_utils",
        "//axlearn/cloud/gcp:k8s_backend_policy",
        "//axlearn/cloud/gcp:k8s_health_check_policy",
        "//axlearn/cloud/gcp:k8s_http_route",
        "//axlearn/cloud/gcp:k8s_service",
        "//axlearn/cloud/gcp:node_pool_provisioner",
        "//axlearn/cloud/gcp:pathways_utils",
        "//axlearn/common:config",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bundler",
    ],
)

py_library(
    name = "gke",
    srcs = ["gke.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:event_queue",
        "//axlearn/cloud/common:metrics",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:config",
        "//axlearn/cloud/gcp:event_queue",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:job_flink",
        "//axlearn/cloud/gcp:jobset_utils",
        "//axlearn/cloud/gcp:node_pool",
        "//axlearn/cloud/gcp:node_pool_provisioner",
        "//axlearn/cloud/gcp:utils",
        "//axlearn/cloud/gcp:vertexai_tensorboard",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
        "@axlearn_pip//requests",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "gke_test",
    srcs = ["gke_test.py"],
    deps = [
        ":__init__",
        ":gke",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/cloud/gcp:bundler",
        "//axlearn/cloud/gcp:job",
        "//axlearn/cloud/gcp:job_flink",
        "//axlearn/cloud/gcp:jobset_utils",
        "//axlearn/cloud/gcp:k8s_backend_policy",
        "//axlearn/cloud/gcp:node_pool",
        "//axlearn/cloud/gcp:node_pool_provisioner",
        "//axlearn/cloud/gcp:pathways_utils",
        "//axlearn/cloud/gcp:test_utils",
        "//axlearn/cloud/gcp:topology_utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
        "@axlearn_pip//requests",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)
