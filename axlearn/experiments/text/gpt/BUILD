load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//:golden_config.bzl", "golden_config_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":c4_trainer",
        ":common",
        ":deterministic_trainer",
        ":envy",
        ":fuji",
        ":gspmd",
        ":pajama_sigmoid_trainer",
        ":pajama_trainer",
        ":qwen",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:attention",
        "//axlearn/common:base_model",
        "//axlearn/common:causal_lm",
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:embedding",
        "//axlearn/common:evaler",
        "//axlearn/common:input_base",
        "//axlearn/common:input_dispatch",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_lm",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:layers",
        "//axlearn/common:learner",
        "//axlearn/common:mixture_of_experts",
        "//axlearn/common:optimizer_base",
        "//axlearn/common:optimizers",
        "//axlearn/common:param_init",
        "//axlearn/common:schedule",
        "//axlearn/common:state_builder",
        "//axlearn/common:summary_writer",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/experiments/text:common",
        "@axlearn_pip//jax",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "c4_trainer",
    srcs = ["c4_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":envy",
        ":fuji",
        ":gspmd",
        ":qwen",
        ":vocabulary_fuji_v3",
        "//axlearn/common:config",
        "//axlearn/common:input_lm",
        "//axlearn/common:utils",
        "//axlearn/experiments/text:common",
    ],
)

py_library(
    name = "deterministic_trainer",
    srcs = ["deterministic_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":gala",
        ":honeycrisp",
        "//axlearn/common:config",
        "//axlearn/common:input_tf_data",
        "//axlearn/experiments/text:common",
    ],
)

py_library(
    name = "pajama_sigmoid_trainer",
    srcs = ["pajama_sigmoid_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":deterministic_trainer",
        ":gala",
        ":gala_sigmoid",
        ":pajama_trainer",
        "//axlearn/common:config",
    ],
)

py_library(
    name = "pajama_trainer",
    srcs = ["pajama_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":gala",
        ":honeycrisp",
        "//axlearn/common:config",
        "//axlearn/common:input_lm",
        "//axlearn/common:utils",
        "//axlearn/experiments/text:common",
    ],
)

py_library(
    name = "qwen",
    srcs = ["qwen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "//axlearn/common:mixture_of_experts",
        "//axlearn/common:trainer_config_modifier",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/common/flash_attention:remat",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "envy",
    srcs = ["envy.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":fuji",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "//axlearn/common:mixture_of_experts",
        "//axlearn/common:trainer",
        "//axlearn/common:trainer_config_modifier",
        "//axlearn/common:utils",
        "//axlearn/experiments:trainer_config_utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "fuji",
    srcs = ["fuji.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "//axlearn/common:trainer",
        "//axlearn/common:trainer_config_modifier",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/common/flash_attention:remat",
        "//axlearn/experiments:trainer_config_utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "gala",
    srcs = ["gala.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
    ],
)

py_library(
    name = "gala_sigmoid",
    srcs = ["gala_sigmoid.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":gala",
        "//axlearn/common:attention",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
    ],
)

py_library(
    name = "gspmd",
    srcs = ["gspmd.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "//axlearn/common:learner",
        "//axlearn/common:optimizers",
        "//axlearn/common:pipeline",
        "//axlearn/common:schedule",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "honeycrisp",
    srcs = ["honeycrisp.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "vocabulary_fuji_v3",
    srcs = ["vocabulary_fuji_v3.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:file_system",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

golden_config_test(
    name = "c4_trainer_golden_config_test",
    checks = [
        "check",
        "check_init",
        "check_regularizer",
    ],
    data = ["//axlearn/experiments/testdata:golden_files"],
    module = ":c4_trainer",
    deps = [":qwen"],
)

golden_config_test(
    name = "deterministic_trainer_golden_config_test",
    checks = ["check"],
    data = ["//axlearn/experiments/testdata:golden_files"],
    module = ":deterministic_trainer",
)

golden_config_test(
    name = "pajama_trainer_golden_config_test",
    checks = [
        "check",
        "check_init",
        "check_regularizer",
    ],
    data = ["//axlearn/experiments/testdata:golden_files"],
    module = ":pajama_trainer",
)

golden_config_test(
    name = "pajama_sigmoid_trainer_golden_config_test",
    checks = [
        "check",
        "check_init",
        "check_regularizer",
    ],
    data = ["//axlearn/experiments/testdata:golden_files"],
    module = ":pajama_sigmoid_trainer",
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "//axlearn/common:learner",
        "//axlearn/common:test_utils",
        "//axlearn/common:trainer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "gala_sigmoid_test",
    srcs = ["gala_sigmoid_test.py"],
    deps = [
        ":common",
        ":gala_sigmoid",
        "//axlearn/common:config",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:test_utils",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
    ],
)

# manual: instantiates up to 70B model parameters; OOMs/times out in RBE. See #2442.
py_test(
    name = "vocabulary_fuji_v3_test",
    srcs = ["vocabulary_fuji_v3_test.py"],
    # manual: needs tokenizer file not available in sandbox.
    tags = ["manual"],
    deps = [
        ":vocabulary_fuji_v3",
        "//axlearn/common:config",
        "//axlearn/common:input_lm",
        "//axlearn/common:input_test_utils",
        "//axlearn/common:input_text",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)
