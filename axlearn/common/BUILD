load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":param_init",
        ":pipeline",
        ":repeat",
        ":utils",
        "//axlearn/common/flash_attention:remat",
        "//axlearn/common/kv_cache",
        "//axlearn/common/kv_cache:base_kv_cache",
        "//axlearn/common/quantized_dot_general:layers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "golden_config",
    srcs = ["golden_config.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":module",
        ":test_utils",
        ":utils",
    ],
)

py_library(
    name = "golden",
    srcs = ["golden.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//numpy",
    ],
)


py_library(
    name = "attention_bias",
    srcs = ["attention_bias.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":flax_struct",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "attention_bias_test",
    srcs = ["attention_bias_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":metrics",
        ":module",
        ":param_init",
        ":traceback_util",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "base_layer_test",
    srcs = ["base_layer_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "array_serialization",
    srcs = ["array_serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorstore",
    ],
)

py_test(
    name = "array_serialization_test",
    srcs = ["array_serialization_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":array_serialization",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorstore",
    ],
)

py_library(
    name = "base_encoder_decoder",
    srcs = ["base_encoder_decoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":decoding",
        ":logit_modifiers",
        ":utils",
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":module",
    ],
)

py_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":array_serialization",
        ":config",
        ":elastic_input",
        ":file_system",
        ":metrics",
        ":module",
        ":summary_writer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//msgpack",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "checkpointer_orbax",
    srcs = ["checkpointer_orbax.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpointer",
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//etils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//orbax_checkpoint",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "checkpointer_test",
    srcs = ["checkpointer_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":array_serialization",
        ":checkpointer",
        ":checkpointer_orbax",
        ":file_system",
        ":metrics",
        ":serialization",
        ":summary_writer",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//orbax_checkpoint",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "checkpointer_orbax_test",
    srcs = ["checkpointer_orbax_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpointer",
        ":checkpointer_orbax",
        ":test_utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//orbax_checkpoint",
    ],
)

py_library(
    name = "compiler_options",
    srcs = ["compiler_options.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "compiler_options_test",
    srcs = ["compiler_options_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_options",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
        "@axlearn_pip//attrs",
    ],
)

py_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//attrs",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
        "@axlearn_pip//wrapt",
    ],
)

# NOTE (wyi): I haven’t added the convolution_test target because it verifies the JAX-based
# attention implementation by parity-checking against PyTorch's Conv_D.
py_library(
    name = "bert",
    srcs = ["bert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":embedding",
        ":encoder",
        ":layers",
        ":loss",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "convolution",
    srcs = ["convolution.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":ein_ops",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "conformer",
    srcs = ["conformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":convolution",
        ":ein_ops",
        ":layers",
        ":module",
        ":repeat",
        ":utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "logit_modifiers",
    srcs = ["logit_modifiers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":decoding",
        ":module",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "logit_modifiers_test",
    srcs = ["logit_modifiers_test.py"],
    deps = [
        ":decoding",
        ":logit_modifiers",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "decoding",
    srcs = ["decoding.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":flax_struct",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "decoding_test",
    srcs = ["decoding_test.py"],
    deps = [
        ":decoding",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//sentencepiece",
    ],
)

py_library(
    name = "embedding",
    srcs = ["embedding.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.embedding_test.npz",
    ],
    deps = [
        ":attention",
        ":golden",
        ":config",
        ":embedding",
        ":layers",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "deberta",
    srcs = ["deberta.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":bert",
        ":config",
        ":embedding",
        ":encoder",
        ":layers",
        ":module",
        ":param_init",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":decoding",
        ":ein_ops",
        ":embedding",
        ":layers",
        ":logit_modifiers",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":base_model",
        ":config",
        ":embedding",
        ":layers",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "encoder_decoder",
    srcs = ["encoder_decoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":base_encoder_decoder",
        ":config",
        ":decoder",
        ":decoding",
        ":encoder",
        ":logit_modifiers",
        ":loss",
        ":metrics",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "t5",
    srcs = ["t5.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":decoder",
        ":encoder",
        ":encoder_decoder",
        ":layers",
        ":param_init",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "t5x_param_converter",
    srcs = ["t5x_param_converter.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":t5",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "text_encoder",
    srcs = ["text_encoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":encoder",
        ":module",
        ":poolings",
        ":utils",
    ],
)

py_library(
    name = "distilbert",
    srcs = ["distilbert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":bert",
        ":config",
        ":layers",
        ":poolings",
        ":text_encoder",
    ],
)

py_library(
    name = "dit",
    srcs = ["dit.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":ein_ops",
        ":layers",
        ":module",
        ":utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "decoder_test",
    size = "enormous",
    srcs = ["decoder_test.py"],
    deps = [
        ":attention",
        ":attention_bias",
        ":base_layer",
        ":causal_lm",
        ":config",
        ":decoder",
        ":decoding",
        ":embedding",
        ":layers",
        ":logit_modifiers",
        ":module",
        ":test_utils",
        ":utils",
        "//axlearn/common/flash_attention:layer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "causal_lm",
    srcs = ["causal_lm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":base_model",
        ":config",
        ":decoder",
        ":decoding",
        ":embedding",
        ":layers",
        ":logit_modifiers",
        ":loss",
        ":loss_metrics",
        ":metrics",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "causal_lm_test",
    srcs = ["causal_lm_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.causal_lm_test.npz",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":golden",
        ":causal_lm",
        ":config",
        ":learner",
        ":loss",
        ":loss_metrics",
        ":metrics",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":param_init",
        ":test_utils",
        ":update_transformation",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "ein_ops",
    srcs = ["ein_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "elastic_input",
    srcs = ["elastic_input.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":input_dispatch",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "factorized_rms",
    srcs = ["factorized_rms.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":optimizer_base",
        ":schedule",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "file_system",
    srcs = ["file_system.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "evaler",
    srcs = ["evaler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_model",
        ":config",
        ":elastic_input",
        ":flax_struct",
        ":inference_output",
        ":input_base",
        ":metrics",
        ":module",
        ":summary_writer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "inference_output",
    srcs = ["inference_output.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "input_base",
    srcs = ["input_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_dispatch",
        ":module",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "input_base_test",
    srcs = ["input_base_test.py"],
    deps = [
        ":input_base",
        ":input_dispatch",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "input_dispatch",
    srcs = ["input_dispatch.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":convolution",
        ":loss",
        ":metrics_classification",
        ":module",
        ":param_init",
        "//axlearn/common/quantized_dot_general",
    ],
)

py_library(
    name = "metrics_classification",
    srcs = ["metrics_classification.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":normalize",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "loss_metrics",
    srcs = ["loss_metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":metrics",
        ":utils",
    ],
)

py_test(
    name = "loss_metrics_test",
    srcs = ["loss_metrics_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":embedding",
        ":loss_metrics",
        ":metrics",
        ":module",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "normalize",
    srcs = ["normalize.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":learner_base",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":update_transformation",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "learner_base",
    srcs = ["learner_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":module",
        ":optimizer_base",
        ":utils",
    ],
)

py_library(
    name = "param_init",
    srcs = ["param_init.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "poolings",
    srcs = ["poolings.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":attention_bias",
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "repeat",
    srcs = ["repeat.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "ragged_all_to_all_batching",
    srcs = ["ragged_all_to_all_batching.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "state_builder",
    srcs = ["state_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpointer",
        ":config",
        ":input_fake",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "segment_tree",
    srcs = ["segment_tree.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":serialization",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//attrs",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":summary",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "measurement_base",
    srcs = ["measurement_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "measurement",
    srcs = ["measurement.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":measurement_base",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "measurement_test",
    srcs = ["measurement_test.py"],
    deps = [
        ":measurement",
        ":measurement_base",
        "//axlearn/cloud",
        "//axlearn/cloud/gcp:measurement",
        "//axlearn/experiments/testdata",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "module",
    srcs = ["module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":flax_struct",
        ":summary",
        ":traceback_util",
        ":utils",
    ],
)

py_library(
    name = "optimizer_base",
    srcs = ["optimizer_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":utils",
        "@axlearn_pip//optax",
        "@axlearn_pip//typing_extensions",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":factorized_rms",
        ":flax_struct",
        ":module",
        ":optimizer_base",
        ":schedule",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
        "@axlearn_pip//typing_extensions",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    deps = [
        ":flax_struct",
        ":serialization",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "flax_struct",
    srcs = ["flax_struct.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":serialization",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//typing_extensions",
    ],
)

py_library(
    name = "flax_struct_test_lib",
    srcs = ["flax_struct_test.py"],
    deps = [
        ":flax_struct",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//jaxlib",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "flax_struct_test",
    size = "small",
    srcs = ["flax_struct_test.py"],
    deps = [
        ":flax_struct_test_lib",
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":flax_struct",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "summary_writer",
    srcs = ["summary_writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":module",
        ":summary",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//wandb",
    ],
)

py_library(
    name = "traceback_util",
    srcs = ["traceback_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":checkpointer",
        ":compiler_options",
        ":config",
        ":evaler",
        ":file_system",
        ":input_base",
        ":learner",
        ":measurement",
        ":module",
        ":optimizer_base",
        ":param_init",
        ":state_builder",
        ":summary_writer",
        ":update_transformation",
        ":utils",
        "//axlearn/common/monitoring:device_monitor",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "update_transformation",
    srcs = ["update_transformation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":flax_struct",
        ":learner_base",
        ":module",
        ":optimizer_base",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//optax",
    ],
)

py_library(
    name = "status_server",
    srcs = ["status_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "utils_spmd",
    srcs = ["utils_spmd.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//jax",
        "@axlearn_pip//portpicker",
    ],
)

py_library(
    name = "debug_utils",
    srcs = ["debug_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//jaxlib",
    ],
)

py_library(
    name = "gradient_accumulation",
    srcs = ["gradient_accumulation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//jaxlib",
    ],
)

py_library(
    name = "trainer_config_modifier",
    srcs = ["trainer_config_modifier.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":gradient_accumulation",
        ":module",
        ":trainer",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "trainer_test_lib",
    testonly = True,
    srcs = ["trainer_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":debug_utils",
        ":flax_struct_test_lib",
        ":layers",
        ":learner",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        ":trainer",
        ":trainer_config_modifier",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//jaxlib",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "launch",
    srcs = ["launch.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_options",
        ":status_server",
        ":utils",
        ":utils_spmd",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "launch_trainer",
    srcs = ["launch_trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":measurement",
        ":trainer",
        ":utils",
        "//axlearn/experiments:__init__",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_binary(
    name = "launch_trainer_main",
    srcs = ["launch_trainer_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":launch",
        ":launch_trainer",
        ":measurement",
        "//axlearn/cloud/gcp:measurement",
        # Experiment modules for dynamic loading with all dependencies
        "//axlearn/experiments:__init__",
        "//axlearn/experiments/logistic_regression:tutorial",
        # Add other experiment modules here as they get Bazelized
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "input_fake",
    srcs = ["input_fake.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "input_grain",
    srcs = ["input_grain.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":module",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//array_record",
        "@axlearn_pip//grain",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "input_grain_test",
    srcs = ["input_grain_test.py"],
    deps = [
        ":config",
        ":input_dispatch",
        ":input_fake",
        ":input_grain",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//grain",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "input_grain_tpu_test",
    srcs = ["input_grain_tpu_test.py"],
    tags = [
        "manual",
        "tpu",
    ],
    deps = [
        ":config",
        ":input_dispatch",
        ":input_grain",
        ":test_utils",
        ":utils",
        ":utils_spmd",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//grain",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "aot_compilation",
    srcs = ["aot_compilation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":inference",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "aot_compilation_test",
    srcs = ["aot_compilation_test.py"],
    deps = [
        ":aot_compilation",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
    ],
)

py_library(
    name = "input_lm",
    srcs = ["input_lm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":input_text",
        ":input_tf_data",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "input_lm_test",
    srcs = ["input_lm_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_fake",
        ":input_lm",
        ":input_test_utils",
        ":input_tf_data",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_datasets",
    ],
)

py_library(
    name = "input_text",
    srcs = ["input_text.py"],
    data = glob(["nltk_data/**"]),
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_tf_data",
        ":utils",
        "@axlearn_pip//nltk",
        "@axlearn_pip//numpy",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_text",
    ],
)

py_library(
    name = "input_tf_data",
    srcs = ["input_tf_data.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":input_base",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "input_test_utils",
    srcs = ["input_test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":input_tf_data",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "input_tf_data_test",
    srcs = ["input_tf_data_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpointer",
        ":config",
        ":input_dispatch",
        ":input_fake",
        ":input_tf_data",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_datasets",
    ],
)

py_test(
    name = "input_tf_data_gcs_test",
    srcs = ["input_tf_data_gcs_test.py"],
    # These tests require GCS authentication to access dataset metadata.
    # Run with: bazel test --test_tag_filters=gs_login //...
    # Skip with: bazel test --test_tag_filters=-gs_login //...
    tags = [
        "gs_login",
        "manual",
    ],  # manual = not included in //... by default
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_tf_data",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow_datasets",
    ],
)

py_library(
    name = "mixture_of_experts",
    srcs = ["mixture_of_experts.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":config",
        ":ein_ops",
        ":layers",
        ":module",
        ":param_init",
        ":ragged_all_to_all_batching",
        ":utils",
        "//axlearn/common/megablock:ops",
        "//axlearn/common/quantized_dot_general:layers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tokamax",
    ],
)

py_test(
    name = "mixture_of_experts_test",
    timeout = "long",  # Set timeout to 'long' (typically around 600s)
    srcs = ["mixture_of_experts_test.py"],
    shard_count = 8,
    deps = [
        ":attention",
        ":layers",
        ":mixture_of_experts",
        ":module",
        ":test_utils",
        ":utils",
        "//axlearn/common/quantized_dot_general:activation_clipping",
        "//axlearn/common/quantized_dot_general:layers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":learner",
        ":module",
        ":optimizers",
        ":schedule",
        ":trainer",
        ":utils",
        ":utils_spmd",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//jaxlib",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "vocabulary_bpe",
    srcs = ["vocabulary_bpe.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//regex",
        "@axlearn_pip//sentencepiece",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_text",
    ],
)

py_library(
    name = "inference_pipeline",
    srcs = ["inference_pipeline.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":inference",
        ":inference_output",
        ":input_tf_data",
        ":module",
        ":summary_writer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "inference",
    srcs = ["inference.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_model",
        ":checkpointer",
        ":config",
        ":module",
        ":state_builder",
        ":trainer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "inference_test",
    srcs = ["inference_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_model",
        ":checkpointer",
        ":config",
        ":inference",
        ":inference_output",
        ":inference_pipeline",
        ":input_tf_data",
        ":layers",
        ":module",
        ":optimizers",
        ":param_init",
        ":state_builder",
        ":test_utils",
        ":trainer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "vision_transformer",
    srcs = ["vision_transformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":convolution",
        ":layers",
        ":module",
        ":param_init",
        ":poolings",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "multiway_transformer",
    srcs = ["multiway_transformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":embedding",
        ":layers",
        ":module",
        ":param_init",
        ":poolings",
        ":utils",
        ":vision_transformer",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "vision_transformer_test",
    srcs = ["vision_transformer_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":module",
        ":test_utils",
        ":utils",
        ":vision_transformer",
        "//axlearn/vision:mask_generator",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "gradient_monitor",
    srcs = ["gradient_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":learner",
        ":optimizers",
        ":param_init",
        ":utils",
        "@axlearn_pip//attrs",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "gradient_monitor_test",
    srcs = ["gradient_monitor_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":gradient_monitor",
        ":learner",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        "//axlearn/common/kv_cache:base_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

# ===== py_library targets for source modules (needed by new py_test targets below) =====

py_library(
    name = "adapter_flax",
    srcs = ["adapter_flax.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":utils",
        "@axlearn_pip//flax",
    ],
)

py_library(
    name = "distillation",
    srcs = ["distillation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":layers",
        ":loss",
        ":metrics",
        ":module",
    ],
)

py_library(
    name = "eval_classification",
    srcs = ["eval_classification.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":evaler",
        ":metrics_classification",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "eval_retrieval",
    srcs = ["eval_retrieval.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":base_model",
        ":config",
        ":evaler",
        ":file_system",
        ":loss",
        ":metrics_retrieval",
        ":module",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "input_composite",
    srcs = ["input_composite.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":module",
    ],
)

py_library(
    name = "input_glue",
    srcs = ["input_glue.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_lm",
        ":input_text",
        ":input_tf_data",
    ],
)

py_library(
    name = "input_grain_lm",
    srcs = ["input_grain_lm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_grain",
        ":input_grain_text",
        "@axlearn_pip//grain",
    ],
)

py_library(
    name = "input_grain_text",
    srcs = ["input_grain_text.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_grain",
    ],
)

py_library(
    name = "input_mlm",
    srcs = ["input_mlm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_text",
        ":input_tf_data",
        ":utils",
    ],
)

py_library(
    name = "input_ranking",
    srcs = ["input_ranking.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":input_tf_data",
    ],
)

py_library(
    name = "input_t5",
    srcs = ["input_t5.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_lm",
        ":input_tf_data",
    ],
)

py_library(
    name = "liveness_monitor",
    srcs = ["liveness_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorboard",
    ],
)

py_library(
    name = "metrics_correlation",
    srcs = ["metrics_correlation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":metrics_classification",
        ":utils",
    ],
)

py_library(
    name = "metrics_glue",
    srcs = ["metrics_glue.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":evaler",
        ":metrics",
        ":metrics_classification",
        ":metrics_correlation",
        ":utils",
    ],
)

py_library(
    name = "metrics_retrieval",
    srcs = ["metrics_retrieval.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "metrics_text_dual_encoder",
    srcs = ["metrics_text_dual_encoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":config",
        ":evaler",
        ":loss",
        ":metrics_retrieval",
        ":text_dual_encoder",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "multi_stream_model",
    srcs = ["multi_stream_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":module",
        ":utils",
    ],
)

py_library(
    name = "quantizer",
    srcs = ["quantizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":metrics",
        ":module",
        ":normalize",
        ":param_init",
        ":schedule",
        ":utils",
    ],
)

py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":repeat",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "rnn_test",
    srcs = ["rnn_test.py"],
    deps = [
        ":layers",
        ":module",
        ":param_init",
        ":rnn",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "transducer",
    srcs = ["transducer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":utils",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "text_dual_encoder",
    srcs = ["text_dual_encoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":loss",
        ":module",
        ":multi_stream_model",
        ":param_init",
        ":schedule",
        ":text_encoder",
        ":utils",
    ],
)

py_library(
    name = "utils_tf",
    srcs = ["utils_tf.py"],
    visibility = ["//visibility:public"],
)

# ===== testonly py_library targets (for cross-test imports) =====

py_library(
    name = "evaler_test_lib",
    testonly = True,
    srcs = ["evaler_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":evaler",
        ":file_system",
        ":inference_output",
        ":input_base",
        ":layers",
        ":metrics",
        ":module",
        ":param_init",
        ":summary_writer",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "eval_retrieval_test_lib",
    testonly = True,
    srcs = ["eval_retrieval_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":eval_retrieval",
        ":evaler_test_lib",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "layers_test_lib",
    testonly = True,
    srcs = ["layers_test.py"],
    visibility = ["//visibility:public"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.layers_test.npz",
    ],
    deps = [
        ":base_layer",
        ":config",
        ":decoder",
        ":golden",
        ":layers",
        ":module",
        ":param_init",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# ===== py_test targets (51 new tests) =====

py_test(
    name = "adapter_flax_test",
    srcs = ["adapter_flax_test.py"],
    deps = [
        ":adapter_flax",
        ":base_layer",
        ":config",
        ":module",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "distillation_test",
    srcs = ["distillation_test.py"],
    deps = [
        ":distillation",
        ":loss",
        ":module",
        ":test_utils",
        ":utils",
        ":vision_transformer",
        "//axlearn/vision:image_classification",
    ],
)

py_test(
    name = "ein_ops_test",
    srcs = ["ein_ops_test.py"],
    deps = [
        ":ein_ops",
        ":test_utils",
        "@axlearn_pip//einops",
    ],
)

# Re-enabled on RBE after axlearn-wheels published fixed tensorflow-io 0.37.3 wheels
# (see issue #2457). Pinned to direct URLs in requirements_lock.txt.
# Linux-only: tensorflow-io 0.37.3 wheel in axlearn-wheels is manylinux_2_31_x86_64.
py_test(
    name = "env_test",
    srcs = ["env_test.py"],
    target_compatible_with = ["@platforms//os:linux"],
    deps = [
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_io",
    ],
)

py_test(
    name = "eval_classification_test",
    srcs = ["eval_classification_test.py"],
    deps = [
        ":eval_classification",
        ":evaler_test_lib",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "eval_retrieval_test",
    srcs = ["eval_retrieval_test.py"],
    deps = [
        ":attention_bias",
        ":eval_retrieval",
        ":evaler_test_lib",
        ":test_utils",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "evaler_test",
    srcs = ["evaler_test.py"],
    deps = [
        ":evaler_test_lib",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "factorized_rms_test",
    srcs = ["factorized_rms_test.py"],
    deps = [
        ":base_layer",
        ":factorized_rms",
        ":optimizer_base",
        ":optimizers",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "file_system_test",
    srcs = ["file_system_test.py"],
    deps = [
        ":file_system",
        ":test_utils",
        "@axlearn_pip//google_api_core",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_composite_test",
    srcs = ["input_composite_test.py"],
    deps = [
        ":config",
        ":input_composite",
        ":input_tf_data",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "input_dispatch_test",
    srcs = ["input_dispatch_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    deps = [
        ":input_dispatch",
        ":test_utils",
        ":utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_glue_test",
    srcs = ["input_glue_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:t5-base"],
    deps = [
        ":config",
        ":input_fake",
        ":input_glue",
        ":input_test_utils",
        ":input_tf_data",
        ":utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_grain_lm_test",
    srcs = ["input_grain_lm_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:t5-base"],
    deps = [
        ":input_fake",
        ":input_grain",
        ":input_grain_lm",
        ":input_grain_text",
        ":input_test_utils",
        ":test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_grain_text_test",
    srcs = ["input_grain_text_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:t5-base"],
    deps = [
        ":config",
        ":input_fake",
        ":input_grain",
        ":input_grain_text",
        ":input_test_utils",
        ":test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_mlm_test",
    srcs = ["input_mlm_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:t5-base"],
    shard_count = 4,
    deps = [
        ":config",
        ":input_mlm",
        ":input_test_utils",
        ":input_text",
        ":input_tf_data",
        ":utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "input_ranking_test",
    srcs = ["input_ranking_test.py"],
    deps = [
        ":input_ranking",
        ":test_utils",
    ],
)

py_test(
    name = "input_t5_test",
    srcs = ["input_t5_test.py"],
    data = ["//axlearn/data/tokenizers/sentencepiece:t5-base"],
    deps = [
        ":config",
        ":input_fake",
        ":input_t5",
        ":input_test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "launch_test",
    srcs = ["launch_test.py"],
    deps = [
        ":launch",
        ":test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "launch_trainer_test",
    srcs = ["launch_trainer_test.py"],
    deps = [
        ":launch_trainer",
        ":test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    deps = [
        ":base_layer",
        ":base_model",
        ":config",
        ":gradient_accumulation",
        ":layers",
        ":learner",
        ":metrics",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        ":update_transformation",
        ":update_transformation_test_lib",
        ":utils",
        "//axlearn/common/quantized_dot_general:layers",
    ],
)

py_library(
    name = "update_transformation_test_lib",
    testonly = True,
    srcs = ["update_transformation_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":learner",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        ":update_transformation",
        ":utils",
    ],
)

py_test(
    name = "liveness_monitor_test",
    srcs = ["liveness_monitor_test.py"],
    deps = [
        ":liveness_monitor",
        ":test_utils",
    ],
)

# manual: sklearn private API _binary_clf_curve removed in scikit-learn 1.8.
py_test(
    name = "metrics_classification_test",
    srcs = ["metrics_classification_test.py"],
    deps = [
        ":metrics_classification",
        ":test_utils",
        "@axlearn_pip//evaluate",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//scikit_learn",
    ],
)

py_test(
    name = "metrics_correlation_test",
    srcs = ["metrics_correlation_test.py"],
    deps = [
        ":metrics_correlation",
        ":test_utils",
        "@axlearn_pip//evaluate",
        "@axlearn_pip//jax",
        "@axlearn_pip//scikit_learn",
        "@axlearn_pip//scipy",
    ],
)

py_test(
    name = "metrics_glue_test",
    srcs = ["metrics_glue_test.py"],
    deps = [
        ":metrics_glue",
        ":test_utils",
        ":utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "metrics_retrieval_test",
    srcs = ["metrics_retrieval_test.py"],
    deps = [
        ":loss",
        ":metrics_retrieval",
        ":test_utils",
        "@axlearn_pip//pytest",
        "@axlearn_pip//scikit_learn",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        ":module",
        ":summary",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "metrics_text_dual_encoder_test",
    srcs = ["metrics_text_dual_encoder_test.py"],
    deps = [
        ":eval_retrieval_test_lib",
        ":metrics_text_dual_encoder",
        ":test_utils",
        ":text_dual_encoder",
        ":utils",
    ],
)

py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":config",
        ":layers",
        ":metrics",
        ":module",
        ":summary",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "multi_stream_model_test",
    srcs = ["multi_stream_model_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":multi_stream_model",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "normalize_test",
    srcs = ["normalize_test.py"],
    deps = [
        ":normalize",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = [
        ":base_layer",
        ":checkpointer",
        ":config",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "param_init_test",
    srcs = ["param_init_test.py"],
    deps = [
        ":param_init",
        ":test_utils",
    ],
)

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":module",
        ":param_init",
        ":pipeline",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "poolings_test",
    srcs = ["poolings_test.py"],
    deps = [
        ":layers",
        ":module",
        ":poolings",
        ":test_utils",
        ":utils",
    ],
)

py_test(
    name = "schedule_test",
    srcs = ["schedule_test.py"],
    deps = [":schedule"],
)

py_test(
    name = "state_builder_test",
    size = "large",
    srcs = ["state_builder_test.py"],
    deps = [
        ":base_layer",
        ":base_model",
        ":bert",
        ":checkpointer_orbax",
        ":config",
        ":convolution",
        ":input_base",
        ":input_fake",
        ":layers",
        ":module",
        ":repeat",
        ":state_builder",
        ":test_utils",
        ":trainer",
        ":utils",
    ],
)

py_test(
    name = "status_server_test",
    srcs = ["status_server_test.py"],
    deps = [
        ":status_server",
        ":test_utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "summary_test",
    srcs = ["summary_test.py"],
    deps = [
        ":config",
        ":evaler",
        ":learner",
        ":module",
        ":optimizers",
        ":summary",
        ":summary_writer",
        ":test_utils",
        ":trainer",
        ":trainer_test_lib",
        ":utils",
        "@axlearn_pip//pillow",
    ],
)

py_test(
    name = "summary_writer_test",
    srcs = ["summary_writer_test.py"],
    deps = [
        ":config",
        ":evaler_test_lib",
        ":metrics",
        ":summary",
        ":summary_writer",
        ":test_utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorboard",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":layers",
        ":test_utils",
    ],
)

py_test(
    name = "text_dual_encoder_test",
    srcs = ["text_dual_encoder_test.py"],
    deps = [
        ":bert",
        ":config",
        ":layers",
        ":learner",
        ":module",
        ":schedule",
        ":test_utils",
        ":text_dual_encoder",
        ":text_encoder",
        ":utils",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "trainer_config_modifier_test",
    srcs = ["trainer_config_modifier_test.py"],
    deps = [
        ":attention",
        ":base_layer",
        ":causal_lm",
        ":config",
        ":layers",
        ":optimizers",
        ":test_utils",
        ":trainer",
        ":trainer_config_modifier",
        ":trainer_test_lib",
        "//axlearn/common/quantized_dot_general:layers",
    ],
)

py_test(
    name = "trainer_test",
    size = "enormous",
    srcs = ["trainer_test.py"],
    shard_count = 4,
    deps = [":trainer_test_lib"],
)

py_test(
    name = "update_transformation_test",
    srcs = ["update_transformation_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":learner",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":schedule",
        ":test_utils",
        ":update_transformation",
        ":utils",
    ],
)

py_test(
    name = "utils_tf_test",
    srcs = ["utils_tf_test.py"],
    deps = [
        ":utils_tf",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "debug_utils_test",
    srcs = ["debug_utils_test.py"],
    deps = [
        ":debug_utils",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "traceback_util_test",
    srcs = ["traceback_util_test.py"],
    deps = [
        ":test_utils",
        ":traceback_util",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "segment_tree_test",
    srcs = ["segment_tree_test.py"],
    deps = [
        ":segment_tree",
        ":test_utils",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "gradient_accumulation_test",
    srcs = ["gradient_accumulation_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
    deps = [
        ":gradient_accumulation",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

# manual: assert_allclose with atol/rtol=1e-6 fails in the Bazel sandbox environment.
# Re-enable after loosening tolerance or investigating platform-specific float differences.
py_test(
    name = "gda_test",
    srcs = ["gda_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    deps = [
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "host_array_test",
    srcs = ["host_array_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    deps = [
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "multiway_transformer_test",
    srcs = ["multiway_transformer_test.py"],
    deps = [
        ":attention",
        ":attention_bias",
        ":module",
        ":multiway_transformer",
        ":test_utils",
        ":utils",
        "//axlearn/vision:mask_generator",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "quantizer_test",
    size = "large",
    srcs = ["quantizer_test.py"],
    data = ["//axlearn/experiments/testdata:axlearn.common.quantizer_test/test_forward_against_fairseq.npy"],
    deps = [
        ":module",
        ":normalize",
        ":param_init",
        ":quantizer",
        ":schedule",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "repeat_test",
    srcs = ["repeat_test.py"],
    deps = [
        ":base_layer",
        ":config",
        ":decoder",
        ":layers",
        ":layers_test_lib",
        ":module",
        ":param_init",
        ":repeat",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//scikit_learn",
    ],
)

py_library(
    name = "lora",
    srcs = ["lora.py"],
    deps = [
        ":attention",
        ":base_layer",
        ":config",
        ":layers",
        ":module",
        ":param_init",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "bert_test_lib",
    testonly = True,
    srcs = ["bert_test.py"],
    visibility = ["//visibility:public"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.bert_test.npz",
    ],
    deps = [
        ":attention",
        ":attention_bias",
        ":bert",
        ":golden",
        ":layers",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "deberta_test_lib",
    testonly = True,
    srcs = ["deberta_test.py"],
    visibility = ["//visibility:public"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.deberta_test.npz",
    ],
    deps = [
        ":attention",
        ":bert",
        ":deberta",
        ":golden",
        ":layers",
        ":module",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "t5_test_lib",
    testonly = True,
    srcs = ["t5_test.py"],
    visibility = ["//visibility:public"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.t5_test.npz",
    ],
    deps = [
        ":attention",
        ":golden",
        ":layers",
        ":module",
        ":t5",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "elastic_input_test",
    srcs = ["elastic_input_test.py"],
    deps = [
        ":config",
        ":elastic_input",
        ":input_tf_data",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_datasets",
    ],
)

py_test(
    name = "transducer_test",
    size = "large",
    srcs = ["transducer_test.py"],
    deps = [
        ":config",
        ":module",
        ":test_utils",
        ":transducer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "attention_test",
    size = "large",
    timeout = "long",
    srcs = ["attention_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.attention_test.npz",
    ],
    shard_count = 50,
    deps = [
        ":attention",
        ":attention_bias",
        ":base_layer",
        ":config",
        ":decoder",
        ":golden",
        ":layers",
        ":module",
        ":optimizer_base",
        ":optimizers",
        ":param_init",
        ":pipeline",
        ":test_utils",
        ":utils",
        "//axlearn/common/kv_cache:paged_kv_cache",
        "//axlearn/common/kv_cache:sliding_window_kv_cache",
        "//axlearn/common/quantized_dot_general:layers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "bert_test",
    size = "large",
    srcs = ["bert_test.py"],
    deps = [
        ":bert_test_lib",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "conformer_test",
    size = "large",
    srcs = ["conformer_test.py"],
    data = ["//axlearn/experiments/testdata:axlearn.common.conformer_test/test_against_fairseq.npy"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    deps = [
        ":attention",
        ":attention_bias",
        ":conformer",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "convolution_test",
    size = "large",
    srcs = ["convolution_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.convolution_test.npz",
    ],
    deps = [
        ":convolution",
        ":ein_ops",
        ":golden",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "deberta_test",
    size = "large",
    timeout = "long",
    srcs = ["deberta_test.py"],
    shard_count = 16,
    deps = [
        ":deberta_test_lib",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "distilbert_test",
    srcs = ["distilbert_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.distilbert_test.npz",
    ],
    deps = [
        ":distilbert",
        ":golden",
        ":module",
        ":test_utils",
        ":text_encoder",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "dit_test",
    size = "large",
    srcs = ["dit_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.dit_test.npz",
    ],
    deps = [
        ":attention",
        ":attention_bias",
        ":dit",
        ":golden",
        ":layers",
        ":module",
        ":test_utils",
        ":vision_transformer",
        "//axlearn/common/kv_cache:sliding_window_kv_cache",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)
py_test(
    name = "encoder_decoder_test",
    size = "large",
    timeout = "long",
    srcs = ["encoder_decoder_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.encoder_decoder_test.npz",
    ],
    deps = [
        ":attention",
        ":base_layer",
        ":bert",
        ":causal_lm",
        ":decoder",
        ":decoding",
        ":encoder",
        ":encoder_decoder",
        ":golden",
        ":layers",
        ":module",
        ":t5",
        ":t5x_param_converter",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.encoder_test.npz",
    ],
    deps = [
        ":attention",
        ":bert",
        ":bert_test_lib",
        ":encoder",
        ":golden",
        ":layers",
        ":module",
        ":param_init",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "input_text_test",
    size = "large",
    srcs = ["input_text_test.py"],
    data = [
        "//axlearn/data/tokenizers/sentencepiece:t5-base",
    ],
    deps = [
        ":config",
        ":input_test_utils",
        ":input_text",
        ":input_tf_data",
        ":test_utils",
        ":vocabulary_bpe",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
        "@axlearn_pip//seqio",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "layers_test",
    size = "large",
    srcs = ["layers_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.layers_test.npz",
    ],
    deps = [
        ":decoder",
        ":golden",
        ":layers_test_lib",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//scikit_learn",
    ],
)

py_test(
    name = "lora_test",
    srcs = ["lora_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.lora_test.npz",
    ],
    deps = [
        ":attention",
        ":attention_bias",
        ":ein_ops",
        ":golden",
        ":layers",
        ":lora",
        ":module",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "loss_test",
    size = "large",
    srcs = ["loss_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.loss_test.npz",
    ],
    deps = [
        ":ein_ops",
        ":golden",
        ":loss",
        ":metrics",
        ":test_utils",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//optax",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "t5_test",
    srcs = ["t5_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_False_100.npy",
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_False_20.npy",
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_False_256.npy",
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_True_100.npy",
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_True_20.npy",
        "//axlearn/experiments/testdata:axlearn.common.t5_test/test_buckets_against_t5x_True_256.npy",
    ],
    deps = [
        ":t5_test_lib",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "utils_test",
    size = "large",
    srcs = ["utils_test.py"],
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    deps = [
        ":aot_compilation",
        ":attention",
        ":base_layer",
        ":config",
        ":flax_struct",
        ":layers",
        ":learner",
        ":metrics",
        ":module",
        ":optimizers",
        ":repeat",
        ":serialization",
        ":test_utils",
        ":trainer",
        ":utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "vocabulary_bpe_test",
    srcs = ["vocabulary_bpe_test.py"],
    data = [
        "//axlearn/data/tokenizers/bpe:gpt2.model",
        "//axlearn/data/tokenizers/bpe:opt.model",
        "//axlearn/data/tokenizers/bpe:roberta-base.model",
        "//axlearn/experiments/testdata:axlearn.common.vocabulary_bpe_test.npz",
    ],
    deps = [
        ":golden",
        ":input_test_utils",
        ":test_utils",
        ":vocabulary_bpe",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "neural_retrieval",
    srcs = ["neural_retrieval.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":layers",
        ":text_dual_encoder",
        ":utils_text_dual_encoder",
    ],
)

py_library(
    name = "splade",
    srcs = ["splade.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_bias",
        ":bert",
        ":config",
        ":layers",
        ":module",
        ":poolings",
        ":utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "utils_text_dual_encoder",
    srcs = ["utils_text_dual_encoder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":base_layer",
        ":bert",
        ":config",
        ":layers",
        ":param_init",
        ":poolings",
        ":state_builder",
        ":text_dual_encoder",
        ":text_encoder",
    ],
)

py_test(
    name = "neural_retrieval_test",
    srcs = ["neural_retrieval_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.neural_retrieval_test.npz",
    ],
    deps = [
        ":module",
        ":golden",
        ":neural_retrieval",
        ":test_utils",
        ":text_dual_encoder",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "splade_test",
    srcs = ["splade_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.splade_test.npz",
    ],
    deps = [
        ":golden",
        ":layers",
        ":module",
        ":splade",
        ":test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)


py_test(
    name = "utils_text_dual_encoder_test",
    srcs = ["utils_text_dual_encoder_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.common.utils_text_dual_encoder_test.npz",
    ],
    deps = [
        ":golden",
        ":module",
        ":test_utils",
        ":text_dual_encoder",
        ":utils_text_dual_encoder",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)
