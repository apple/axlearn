load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

# Core flash attention utilities (depends on implementations)
py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":common",
        ":gpu_attention",
        ":gpu_decoding",
        ":gpu_paged_attention",
        ":neuron_attention",
        ":tpu_attention",
        ":tpu_decoding",
        ":tpu_paged_attention",
        ":tpu_splash_attention",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Flash attention types
py_library(
    name = "attention_types",
    srcs = ["attention_types.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Common flash attention components
py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        ":attention_types",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "//axlearn/common/kv_cache:paged_kv_cache",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# Main flash attention layer
py_library(
    name = "layer",
    srcs = ["layer.py"],
    deps = [
        ":utils",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# Splash attention mask
py_library(
    name = "splash_attention_mask",
    srcs = ["splash_attention_mask.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# TPU-specific attention implementation
py_library(
    name = "tpu_attention",
    srcs = ["tpu_attention.py"],
    deps = [
        ":common",
        ":remat",
        ":splash_attention_mask",
        ":tpu_splash_attention",
        ":attention_types",
        "//axlearn/common:attention_bias",
        "//axlearn/common:utils",
        "//axlearn/common/kv_cache:base_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

# TPU Splash attention implementation
py_library(
    name = "tpu_splash_attention",
    srcs = ["tpu_splash_attention.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# TPU Paged attention implementation
py_library(
    name = "tpu_paged_attention",
    srcs = ["tpu_paged_attention.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# GPU-specific attention implementation
py_library(
    name = "gpu_attention",
    srcs = ["gpu_attention.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# GPU decoding implementation
py_library(
    name = "gpu_decoding",
    srcs = ["gpu_decoding.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# GPU paged attention implementation
py_library(
    name = "gpu_paged_attention",
    srcs = ["gpu_paged_attention.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# TPU paged attention kernel
py_library(
    name = "tpu_paged_attention_kernel",
    srcs = ["tpu_paged_attention_kernel.py"],
    deps = [
        "//axlearn/common:attention_bias",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "tpu_decoding",
    srcs = ["tpu_decoding.py"],
    deps = [
        ":common",
        ":tpu_paged_attention_kernel",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Neuron-specific attention implementation
py_library(
    name = "neuron_attention",
    srcs = ["neuron_attention.py"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Remat utilities
py_library(
    name = "remat",
    srcs = ["remat.py"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        ":common",
        ":layer",
        ":utils",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

# Flash attention package that exports all components
py_library(
    name = "flash_attention",
    srcs = ["__init__.py"],
    deps = [
        ":common",
        ":gpu_attention",
        ":gpu_decoding",
        ":layer",
        ":neuron_attention",
        ":remat",
        ":tpu_attention",
        ":tpu_paged_attention",
        ":tpu_splash_attention",
        ":utils",
    ],
)

py_library(
    name = "layer_test_lib",
    testonly = True,  # Added testonly since it depends on test_utils
    srcs = ["layer_test.py"],
    deps = [
        ":layer",
        ":test_utils",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/common/kv_cache",
        "//axlearn/common/kv_cache:sliding_window_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        ":test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":test_utils",
        ":utils",
        "//axlearn/common:test_utils",
        "//axlearn/common/kv_cache:sliding_window_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "splash_attention_mask_test",
    srcs = ["splash_attention_mask_test.py"],
    deps = [
        ":splash_attention_mask",
        "//axlearn/common:attention_bias",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "tpu_attention_test",
    srcs = ["tpu_attention_test.py"],
    deps = [
        ":splash_attention_mask",
        ":test_utils",
        ":tpu_attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "gpu_attention_test",
    srcs = ["gpu_attention_test.py"],
    # manual: requires GPU hardware, times out on CPU-only runners.
    tags = ["manual"],
    deps = [
        ":gpu_attention",
        ":test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "neuron_attention_test",
    srcs = ["neuron_attention_test.py"],
    deps = [
        ":neuron_attention",
        ":test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "decoding_test",
    timeout = "long",  # Set timeout to 'long' (typically around 600s)
    srcs = ["decoding_test.py"],
    deps = [
        ":gpu_decoding",
        ":test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "layer_test",
    size = "large",
    srcs = ["layer_test.py"],
    shard_count = 4,
    deps = [
        ":layer_test_lib",
    ],
)

py_test(
    name = "remat_test",
    srcs = ["remat_test.py"],
    deps = [
        ":layer",
        ":layer_test_lib",
        ":remat",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "tpu_paged_attention_test",
    size = "large",
    srcs = ["tpu_paged_attention_test.py"],
    tags = ["manual"],
    deps = [
        ":common",
        ":test_utils",
        ":tpu_paged_attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

# Benchmarks
py_binary(
    name = "gpu_attention_benchmark",
    testonly = True,
    srcs = ["gpu_attention_benchmark.py"],
    deps = [
        ":gpu_attention",
        ":test_utils",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_binary(
    name = "tpu_attention_benchmark",
    testonly = True,
    srcs = ["tpu_attention_benchmark.py"],
    deps = [
        ":test_utils",
        ":tpu_attention",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)
