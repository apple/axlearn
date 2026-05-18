load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "cleaner",
    srcs = ["cleaner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        ":scheduler",
        "//axlearn/common:config",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//toml",
    ],
)

py_library(
    name = "docker",
    srcs = ["docker.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "event_queue",
    srcs = ["event_queue.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "git_summary",
    srcs = ["git_summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "job",
    srcs = ["job.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//prometheus_client",
    ],
)

py_library(
    name = "uploader",
    srcs = ["uploader.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "validator",
    srcs = ["validator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        "//axlearn/common:config",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        "//axlearn/cloud",
        "//axlearn/cloud/gcp:tpu",
        "//axlearn/common:compiler_options",
        "//axlearn/common:config",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//psutil",
    ],
)

py_library(
    name = "bastion",
    srcs = ["bastion.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cleaner",
        ":event_queue",
        ":job_types",
        ":metrics",
        ":quota",
        ":scheduler",
        ":uploader",
        ":utils",
        ":validator",
        "//axlearn/common:config",
        "//axlearn/common:file_system",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "bundler",
    srcs = ["bundler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":docker",
        ":git_summary",
        ":utils",
        "//axlearn/common:config",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//prefixed",
    ],
)

py_library(
    name = "quota",
    srcs = ["quota.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//toml",
    ],
)

py_library(
    name = "scheduler",
    srcs = ["scheduler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        ":quota",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "job_types",
    srcs = ["job_types.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "pod_mutator",
    srcs = ["pod_mutator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":job_types",
        "//axlearn/common:config",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "user_command_patcher",
    srcs = ["user_command_patcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
    ],
)

py_test(
    name = "bastion_test",
    srcs = ["bastion_test.py"],
    deps = [
        ":bastion",
        ":job_types",
        ":scheduler_test_lib",
        "//axlearn/common:config",
        "//axlearn/common:file_system",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "bundler_test",
    srcs = ["bundler_test.py"],
    local = True,
    deps = [
        ":bundler",
        ":config",
        ":config_test_lib",
        ":git_summary",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//toml",
    ],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        ":config_test_lib",
        "//axlearn/cloud",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "config_test_lib",
    testonly = True,
    srcs = ["config_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//axlearn/cloud",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "event_queue_test",
    srcs = ["event_queue_test.py"],
    deps = [
        ":event_queue",
        "@axlearn_pip//pika",
    ],
)

py_test(
    name = "git_summary_test",
    srcs = ["git_summary_test.py"],
    local = True,
    deps = [
        ":git_summary",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "job_test",
    srcs = ["job_test.py"],
    deps = [
        ":job",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "quota_test",
    srcs = ["quota_test.py"],
    deps = [
        ":quota",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//toml",
    ],
)

py_test(
    name = "scheduler_test",
    srcs = ["scheduler_test.py"],
    deps = [
        ":job_types",
        ":quota",
        ":scheduler",
        ":scheduler_test_lib",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "scheduler_test_lib",
    testonly = True,
    srcs = ["scheduler_test.py"],
    deps = [
        ":job_types",
        ":quota",
        ":scheduler",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    data = ["testdata/counter.py"],
    deps = [
        ":job_types",
        ":utils",
        "//axlearn/cloud",
        "//axlearn/common:config",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//psutil",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "writer",
    srcs = ["writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:file_system",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "cleaner_test",
    srcs = ["cleaner_test.py"],
    deps = [
        ":bastion",
        ":cleaner",
        ":job_types",
        ":quota",
        ":scheduler",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "uploader_test",
    srcs = ["uploader_test.py"],
    deps = [
        ":uploader",
        "//axlearn/common:config",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "writer_test",
    srcs = ["writer_test.py"],
    deps = [
        ":writer",
        "@axlearn_pip//absl_py",
    ],
)
