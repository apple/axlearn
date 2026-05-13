load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "frontend_utils",
    srcs = ["frontend_utils.py"],
    deps = [
        "//axlearn/common:ein_ops",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "frontend",
    srcs = ["frontend.py"],
    deps = [
        ":frontend_utils",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:ein_ops",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "subsamplers",
    srcs = ["subsamplers.py"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "encoder_asr",
    srcs = ["encoder_asr.py"],
    deps = [
        ":frontend",
        ":spectrum_augmenter",
        ":subsamplers",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:conformer",
        "//axlearn/common:convolution",
        "//axlearn/common:ein_ops",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "decoder_asr",
    srcs = ["decoder_asr.py"],
    deps = [
        "//axlearn/audio/aligner:ctc_aligner",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:decoding",
        "//axlearn/common:flax_struct",
        "//axlearn/common:layers",
        "//axlearn/common:logit_modifiers",
        "//axlearn/common:loss",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:rnn",
        "//axlearn/common:transducer",
        "//axlearn/common:utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "model_asr",
    srcs = ["model_asr.py"],
    deps = [
        ":decoder_asr",
        ":encoder_asr",
        "//axlearn/common:base_encoder_decoder",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    deps = [
        ":spectrum_augmenter",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "frontend_utils_test_lib",
    testonly = True,
    srcs = ["frontend_utils_test.py"],
    deps = [
        ":frontend_utils",
        ":test_utils",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "frontend_utils_test",
    srcs = ["frontend_utils_test.py"],
    env = {"JAX_ENABLE_X64": "1"},
    deps = [
        ":frontend_utils_test_lib",
    ],
)

py_test(
    name = "frontend_test",
    srcs = ["frontend_test.py"],
    env = {"JAX_ENABLE_X64": "1"},
    deps = [
        ":frontend",
        ":frontend_utils",
        ":frontend_utils_test_lib",
        ":test_utils",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "encoder_asr_test",
    srcs = ["encoder_asr_test.py"],
    env = {"JAX_ENABLE_X64": "1"},
    deps = [
        ":encoder_asr",
        ":frontend_utils",
        ":test_utils",
        "//axlearn/common:attention",
        "//axlearn/common:convolution",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/common/kv_cache:sliding_window_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "model_asr_test",
    srcs = ["model_asr_test.py"],
    deps = [
        ":decoder_asr",
        ":encoder_asr",
        ":model_asr",
        "//axlearn/common:attention",
        "//axlearn/common:causal_lm",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "subsamplers_test",
    srcs = ["subsamplers_test.py"],
    deps = [
        ":subsamplers",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "input_asr",
    srcs = ["input_asr.py"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:input_text",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:utils",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "input_asr_test",
    srcs = ["input_asr_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:librispeech_unigram_1024.model"],
    deps = [
        ":input_asr",
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "decoder_asr_test",
    srcs = ["decoder_asr_test.py"],
    data = ["//axlearn/experiments/testdata:axlearn.audio.decoder_asr_test.npz"],
    deps = [
        ":decoder_asr",
        "//axlearn/audio/aligner:ctc_aligner_test_lib",
        "//axlearn/common:attention",
        "//axlearn/common:causal_lm",
        "//axlearn/common:config",
        "//axlearn/common:decoding",
        "//axlearn/common:golden",
        "//axlearn/common:logit_modifiers",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:rnn",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "evaler_asr",
    srcs = ["evaler_asr.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":decoder_asr",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:metrics",
        "@axlearn_pip//jax",
        "@axlearn_pip//levenshtein",
        "@axlearn_pip//seqio",
    ],
)

py_test(
    name = "evaler_asr_test",
    srcs = ["evaler_asr_test.py"],
    data = [
        "//axlearn/data/tokenizers/sentencepiece:bpe_32k_c4.model",
        "//axlearn/data/tokenizers/sentencepiece:librispeech_unigram_1024.model",
    ],
    deps = [
        ":decoder_asr",
        ":evaler_asr",
        ":input_asr",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:decoding",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:metrics",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)
