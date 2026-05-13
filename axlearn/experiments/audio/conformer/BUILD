load("//:golden_config.bzl", "golden_config_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [":librispeech_trainer"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/audio:decoder_asr",
        "//axlearn/audio:encoder_asr",
        "//axlearn/audio:input_asr",
        "//axlearn/common:attention",
        "//axlearn/common:config",
        "//axlearn/common:conformer",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:layers",
        "//axlearn/common:learner",
        "//axlearn/common:optimizers",
        "//axlearn/common:param_init",
        "//axlearn/common:rnn",
        "//axlearn/common:schedule",
        "@axlearn_pip//seqio",
    ],
)

py_library(
    name = "librispeech_trainer",
    srcs = ["librispeech_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//axlearn/audio:encoder_asr",
        "//axlearn/audio:evaler_asr",
        "//axlearn/audio:frontend",
        "//axlearn/audio:frontend_utils",
        "//axlearn/audio:input_asr",
        "//axlearn/audio:model_asr",
        "//axlearn/audio:spectrum_augmenter",
        "//axlearn/audio:subsamplers",
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:decoding",
        "//axlearn/common:evaler",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:learner",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:bpe_32k_c4.model"],
    deps = [
        ":common",
        "//axlearn/audio:encoder_asr",
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "//axlearn/common:layers",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//pytest",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

golden_config_test(
    name = "librispeech_trainer_golden_config_test",
    module = ":librispeech_trainer",
    data = ["//axlearn/experiments/testdata:golden_files"],
    checks = ["check", "check_init", "check_regularizer"],
)

py_test(
    name = "librispeech_trainer_test",
    srcs = ["librispeech_trainer_test.py"],
    data = [
        "//axlearn/data/tokenizers/sentencepiece:librispeech_bpe_1024.model",
        "//axlearn/data/tokenizers/sentencepiece:librispeech_unigram_1024.model",
    ],
    deps = [
        ":librispeech_trainer",
        "//axlearn/experiments:test_utils",
    ],
)
