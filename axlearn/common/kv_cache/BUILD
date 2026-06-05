load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

# Base KV cache module
py_library(
    name = "base_kv_cache",
    srcs = ["base_kv_cache.py"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Main KV cache implementation
py_library(
    name = "kv_cache",
    srcs = ["kv_cache.py"],
    deps = [
        ":base_kv_cache",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "sliding_window_kv_cache",
    srcs = ["sliding_window_kv_cache.py"],
    deps = [
        ":base_kv_cache",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "paged_kv_cache",
    srcs = ["paged_kv_cache.py"],
    deps = [
        ":base_kv_cache",
        ":paged_kv_cache_gpu_kernel",
        ":paged_kv_cache_tpu_kernel",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# TPU kernel for paged KV cache
py_library(
    name = "paged_kv_cache_tpu_kernel",
    srcs = ["paged_kv_cache_tpu_kernel.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# GPU kernel for paged KV cache
py_library(
    name = "paged_kv_cache_gpu_kernel",
    srcs = ["paged_kv_cache_gpu_kernel.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# KV cache package
py_library(
    name = "kv_cache_lib",
    srcs = ["__init__.py"],
    deps = [
        ":base_kv_cache",
        ":kv_cache",
        ":paged_kv_cache",
        ":paged_kv_cache_gpu_kernel",
        ":paged_kv_cache_tpu_kernel",
        ":sliding_window_kv_cache",
    ],
)

# Tests
py_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.py"],
    deps = [
        ":kv_cache",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "paged_kv_cache_test",
    srcs = ["paged_kv_cache_test.py"],
    deps = [
        ":kv_cache",
        ":paged_kv_cache",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "sliding_window_kv_cache_test",
    srcs = ["sliding_window_kv_cache_test.py"],
    deps = [
        ":sliding_window_kv_cache",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)
