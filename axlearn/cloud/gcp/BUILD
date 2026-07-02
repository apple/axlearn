load("@rules_python//python:defs.bzl", "py_library", "py_test")

# TODO(axlearn-dev): Add bazel rule for utils_test.
py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":scopes",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_auth",
    ],
)

py_library(
    name = "bundler",
    srcs = ["bundler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cloud_build",
        ":config",
        ":utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "cloud_build",
    srcs = ["cloud_build.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_cloud_build",
        "@axlearn_pip//google_cloud_compute",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//axlearn/cloud/common:config",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "event_queue",
    srcs = ["event_queue.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//axlearn/cloud/common:event_queue",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "job",
    srcs = ["job.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":jobset_utils",
        ":k8s_backend_policy",
        ":k8s_health_check_policy",
        ":k8s_http_route",
        ":k8s_service",
        ":lws_utils",
        ":utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:job",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "job_flink",
    srcs = ["job_flink.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job",
        ":jobset_utils",
        ":node_pool",
        ":system_characteristics",
        ":tpu",
        ":utils",
        "//axlearn/common:compiler_options",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
        "@axlearn_pip//requests",
    ],
)

py_library(
    name = "job_pathways",
    srcs = ["job_pathways.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job",
        ":pathways_utils",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "jobset_utils",
    srcs = ["jobset_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":k8s_readiness_probe",
        ":node_pool",
        ":system_characteristics",
        ":topology_utils",
        ":tpu",
        ":utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:pod_mutator",
        "//axlearn/cloud/common:user_command_patcher",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:compiler_options",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "k8s_backend_policy",
    srcs = ["k8s_backend_policy.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "k8s_health_check_policy",
    srcs = ["k8s_health_check_policy.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":pathways_utils",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "k8s_hpa",
    srcs = ["k8s_hpa.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "k8s_http_route",
    srcs = ["k8s_http_route.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "k8s_metrics_collection",
    srcs = ["k8s_metrics_collection.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "k8s_readiness_probe",
    srcs = ["k8s_readiness_probe.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "k8s_service",
    srcs = ["k8s_service.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":pathways_utils",
        ":utils",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_library(
    name = "lws_utils",
    srcs = ["lws_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":jobset_utils",
        ":system_characteristics",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:pod_mutator",
        "//axlearn/cloud/common:user_command_patcher",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "measurement",
    srcs = ["measurement.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:measurement_base",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//ml_goodput_measurement",
    ],
)

py_library(
    name = "node_pool",
    srcs = ["node_pool.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_api_python_client",
        "@axlearn_pip//google_auth",
    ],
)

py_library(
    name = "node_pool_provisioner",
    srcs = ["node_pool_provisioner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":job",
        ":job_flink",
        ":jobset_utils",
        ":node_pool",
        ":pathways_utils",
        ":system_characteristics",
        ":tpu",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "pathways_utils",
    srcs = ["pathways_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jobset_utils",
        ":lws_utils",
        ":system_characteristics",
        ":tpu",
        ":utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:compiler_options",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "scopes",
    srcs = ["scopes.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "storage",
    srcs = ["storage.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_auth",
        "@axlearn_pip//google_cloud_storage",
        "@axlearn_pip//tqdm",
    ],
)

py_library(
    name = "system_characteristics",
    srcs = ["system_characteristics.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "topology_utils",
    srcs = ["topology_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:bastion",
    ],
)

# TODO(axlearn-dev): Add bazel rule for tpu_test.
py_library(
    name = "tpu",
    srcs = ["tpu.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/cloud/common:job_types",
        "//axlearn/common:compiler_options",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "vertexai_tensorboard",
    srcs = ["vertexai_tensorboard.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "vm",
    srcs = ["vm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "//axlearn/cloud/common:docker",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_api_python_client",
        "@axlearn_pip//google_auth",
    ],
)

py_test(
    name = "bundler_test",
    srcs = ["bundler_test.py"],
    deps = [
        ":bundler",
        ":cloud_build",
        ":test_utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        "//axlearn/cloud/common:config",
        "//axlearn/cloud/common:config_test_lib",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "job_pathways_test",
    srcs = ["job_pathways_test.py"],
    deps = [
        ":bundler",
        ":job",
        ":jobset_utils",
        ":pathways_utils",
        ":test_utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "job_test",
    srcs = ["job_test.py"],
    deps = [
        ":bundler",
        ":job",
        ":jobset_utils",
        ":pathways_utils",
        ":test_utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "jobset_utils_test",
    srcs = ["jobset_utils_test.py"],
    deps = [
        ":bundler",
        ":jobset_utils",
        ":node_pool",
        ":system_characteristics",
        ":test_utils",
        ":tpu",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:job_types",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:compiler_options",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "lws_utils_test",
    srcs = ["lws_utils_test.py"],
    deps = [
        ":jobset_utils",
        ":lws_utils",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "measurement_test",
    srcs = ["measurement_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":measurement",
        "//axlearn/common:config",
        "//axlearn/common:measurement",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "node_pool_test",
    srcs = ["node_pool_test.py"],
    deps = [
        ":node_pool",
        ":test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "node_pool_provisioner_test",
    srcs = ["node_pool_provisioner_test.py"],
    deps = [
        ":bundler",
        ":job",
        ":jobset_utils",
        ":node_pool_provisioner",
        ":pathways_utils",
        ":test_utils",
        "//axlearn/cloud/common:bastion",
        "//axlearn/cloud/common:job_types",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "pathways_utils_test",
    srcs = ["pathways_utils_test.py"],
    deps = [
        ":bundler",
        ":jobset_utils",
        ":lws_utils",
        ":pathways_utils",
        ":test_utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:compiler_options",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "system_characteristics_test",
    srcs = ["system_characteristics_test.py"],
    deps = [
        ":system_characteristics",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "tpu_test",
    srcs = ["tpu_test.py"],
    deps = [
        ":tpu",
        "//axlearn/common:compiler_options",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//kubernetes",
    ],
)

py_test(
    name = "vm_test",
    srcs = ["vm_test.py"],
    deps = [
        ":vm",
        "@axlearn_pip//absl_py",
    ],
)

# Skipped py_test targets:
# - gke_test (in runners/): needs runners/gke.py py_library, which pulls in
#   cloud/common/metrics, event_queue, vertexai_tensorboard, and many more transitive deps.
#   Deferred until runners/ modules have full py_library coverage.
# - vertexai_tensorboard_test: needs google-cloud-aiplatform (not in requirements).

py_library(
    name = "tpu_health_check_main",
    srcs = ["tpu_health_check_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "tpu_health_check",
    srcs = ["tpu_health_check.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_health_check_main",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "cloud_build_test",
    size = "large",
    srcs = ["cloud_build_test.py"],
    deps = [
        ":cloud_build",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_cloud_build",
        "@axlearn_pip//google_cloud_compute",
    ],
)

py_test(
    name = "event_queue_test",
    srcs = ["event_queue_test.py"],
    deps = [
        ":config",
        ":event_queue",
        "//axlearn/cloud/common:config",
        "//axlearn/cloud/common:config_test_lib",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//pika",
    ],
)

py_test(
    name = "job_flink_test",
    size = "large",
    srcs = ["job_flink_test.py"],
    deps = [
        ":bundler",
        ":job_flink",
        ":jobset_utils",
        ":test_utils",
        "//axlearn/cloud/common:bundler",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "k8s_backend_policy_test",
    srcs = ["k8s_backend_policy_test.py"],
    deps = [
        ":k8s_backend_policy",
        ":utils",
        "//axlearn/common:test_utils",
    ],
)

py_test(
    name = "k8s_health_check_policy_test",
    srcs = ["k8s_health_check_policy_test.py"],
    deps = [
        ":k8s_health_check_policy",
        ":pathways_utils",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "k8s_http_route_test",
    srcs = ["k8s_http_route_test.py"],
    deps = [
        ":k8s_http_route",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "k8s_hpa_test",
    srcs = ["k8s_hpa_test.py"],
    deps = [
        ":k8s_hpa",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "k8s_metrics_collection_test",
    srcs = ["k8s_metrics_collection_test.py"],
    deps = [
        ":k8s_metrics_collection",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "k8s_readiness_probe_test",
    srcs = ["k8s_readiness_probe_test.py"],
    deps = [
        ":k8s_readiness_probe",
        "//axlearn/cloud/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "k8s_service_test",
    srcs = ["k8s_service_test.py"],
    deps = [
        ":k8s_service",
        ":pathways_utils",
        "//axlearn/cloud/common:utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "storage_test",
    srcs = ["storage_test.py"],
    # manual: integration test that calls real GCS APIs (gs://axlearn-public/testdata).
    # Needs gcloud auth credentials. See #2385.
    tags = [
        "gs_login",
        "manual",
    ],
    deps = [
        ":storage",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "tpu_health_check_test",
    srcs = ["tpu_health_check_test.py"],
    deps = [
        ":tpu_health_check",
        ":tpu_health_check_main",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "vertexai_tensorboard_test",
    srcs = ["vertexai_tensorboard_test.py"],
    deps = [
        ":config",
        ":test_utils",
        ":vertexai_tensorboard",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//google_cloud_aiplatform",
    ],
)
