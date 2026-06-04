load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "anchor",
    srcs = ["anchor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":matchers",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":window_attention",
        "//axlearn/common:attention",
        "//axlearn/common:attention_bias",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "augment",
    srcs = ["augment.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "beit_image_tokenizer",
    srcs = ["beit_image_tokenizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clip",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:poolings",
        "//axlearn/common:quantizer",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "box_coder",
    srcs = ["box_coder.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "clip",
    srcs = ["clip.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":mobilenets",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:bert",
        "//axlearn/common:config",
        "//axlearn/common:embedding",
        "//axlearn/common:layers",
        "//axlearn/common:loss",
        "//axlearn/common:module",
        "//axlearn/common:multi_stream_model",
        "//axlearn/common:param_init",
        "//axlearn/common:poolings",
        "//axlearn/common:text_encoder",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "coca",
    srcs = ["coca.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clip",
        "//axlearn/common:attention",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:embedding",
        "//axlearn/common:encoder",
        "//axlearn/common:layers",
        "//axlearn/common:loss",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:multi_stream_model",
        "//axlearn/common:poolings",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "coco_evaluator",
    srcs = ["coco_evaluator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":coco_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pycocotools",
    ],
)

py_library(
    name = "coco_utils",
    srcs = ["coco_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils_detection",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pillow",
        "@axlearn_pip//pycocotools",
        "@axlearn_pip//six",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "cyclip",
    srcs = ["cyclip.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clip",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:loss",
        "//axlearn/common:module",
        "//axlearn/common:multi_stream_model",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "detection_generator",
    srcs = ["detection_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":box_coder",
        ":nms",
        ":utils_detection",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "efficientdet",
    srcs = ["efficientdet.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fpn",
        ":mobilenets",
        ":retinanet",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "eval_detection",
    srcs = ["eval_detection.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":coco_evaluator",
        ":utils_visualization",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "feature_tokenizer",
    srcs = ["feature_tokenizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
    ],
)

py_library(
    name = "fpn",
    srcs = ["fpn.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "image_classification",
    srcs = ["image_classification.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "input_detection",
    srcs = ["input_detection.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":input_image",
        ":mask_generator",
        ":utils_detection",
        "//axlearn/common:config",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:utils",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "input_image",
    srcs = ["input_image.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":augment",
        ":mask_generator",
        "//axlearn/common:config",
        "//axlearn/common:input_tf_data",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "mask_generator",
    srcs = ["mask_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "masked_image_model",
    srcs = ["masked_image_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "matchers",
    srcs = ["matchers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":similarity_ops",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "metrics_vqa",
    srcs = ["metrics_vqa.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "mobilenets",
    srcs = ["mobilenets.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":mobilenets_blocks",
        "//axlearn/common:base_layer",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
    ],
)

py_library(
    name = "mobilenets_blocks",
    srcs = ["mobilenets_blocks.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
    ],
)

py_library(
    name = "nms",
    srcs = ["nms.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "resnet",
    srcs = ["resnet.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "retinanet",
    srcs = ["retinanet.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anchor",
        ":box_coder",
        ":detection_generator",
        ":fpn",
        ":resnet",
        ":utils_detection",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:layers",
        "//axlearn/common:loss",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "roi_aligner",
    srcs = ["roi_aligner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":spatial_transform_ops",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "similarity_ops",
    srcs = ["similarity_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "spatial_transform_ops",
    srcs = ["spatial_transform_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "utils_detection",
    srcs = ["utils_detection.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_library(
    name = "utils_visualization",
    srcs = ["utils_visualization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":input_image",
        ":utils_detection",
        "//axlearn/common:utils",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "vitdet_transformer",
    srcs = ["vitdet_transformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        "//axlearn/common:config",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:poolings",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "window_attention",
    srcs = ["window_attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

# Tests

py_test(
    name = "augment_test",
    srcs = ["augment_test.py"],
    deps = [
        ":augment",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "beit_image_tokenizer_test",
    srcs = ["beit_image_tokenizer_test.py"],
    deps = [
        ":beit_image_tokenizer",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "coca_test",
    srcs = ["coca_test.py"],
    deps = [
        ":clip",
        ":coca",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "cyclip_test",
    srcs = ["cyclip_test.py"],
    deps = [
        ":clip",
        ":cyclip",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "efficientdet_test",
    srcs = ["efficientdet_test.py"],
    deps = [
        ":efficientdet",
        ":retinanet",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "feature_tokenizer_test",
    srcs = ["feature_tokenizer_test.py"],
    deps = [
        ":clip",
        ":feature_tokenizer",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "input_detection_test",
    srcs = ["input_detection_test.py"],
    deps = [
        ":input_detection",
        ":utils_detection",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "input_image_test",
    srcs = ["input_image_test.py"],
    deps = [
        ":input_image",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//tensorflow",
        "@axlearn_pip//tensorflow_datasets",
    ],
)

py_test(
    name = "mask_generator_test",
    srcs = ["mask_generator_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":mask_generator",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "masked_image_model_test",
    srcs = ["masked_image_model_test.py"],
    deps = [
        ":beit_image_tokenizer",
        ":clip",
        ":feature_tokenizer",
        ":mask_generator",
        ":masked_image_model",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "metrics_vqa_test",
    srcs = ["metrics_vqa_test.py"],
    deps = [
        ":metrics_vqa",
        "@axlearn_pip//absl_py",
    ],
)

py_test(
    name = "resnet_test",
    size = "large",
    srcs = ["resnet_test.py"],
    deps = [
        ":resnet",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "roi_aligner_test",
    srcs = ["roi_aligner_test.py"],
    deps = [
        ":roi_aligner",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "utils_detection_test",
    srcs = ["utils_detection_test.py"],
    deps = [
        ":utils_detection",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "vitdet_transformer_test",
    srcs = ["vitdet_transformer_test.py"],
    deps = [
        ":vitdet_transformer",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

# New py_library targets for detection pipeline

py_library(
    name = "detection_heads",
    srcs = ["detection_heads.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:convolution",
        "//axlearn/common:flax_struct",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "rcnn_losses",
    srcs = ["rcnn_losses.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:loss",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "rcnn_sampler",
    srcs = ["rcnn_sampler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":matchers",
        ":samplers",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "roi_generator",
    srcs = ["roi_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":box_coder",
        ":detection_generator",
        ":utils_detection",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "rpn_sampler",
    srcs = ["rpn_sampler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":matchers",
        ":samplers",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "samplers",
    srcs = ["samplers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:utils",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_library(
    name = "rcnn",
    srcs = ["rcnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anchor",
        ":box_coder",
        ":detection_generator",
        ":detection_heads",
        ":fpn",
        ":rcnn_losses",
        ":rcnn_sampler",
        ":resnet",
        ":roi_aligner",
        ":roi_generator",
        ":rpn_sampler",
        ":utils_detection",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "@axlearn_pip//jax",
    ],
)

# New py_test targets

py_test(
    name = "anchor_test",
    srcs = ["anchor_test.py"],
    deps = [
        ":anchor",
        ":matchers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "box_coder_test",
    srcs = ["box_coder_test.py"],
    deps = [
        ":box_coder",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "detection_heads_test",
    srcs = ["detection_heads_test.py"],
    deps = [
        ":detection_heads",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "fpn_test",
    srcs = ["fpn_test.py"],
    deps = [
        ":fpn",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "matchers_test",
    srcs = ["matchers_test.py"],
    deps = [
        ":matchers",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "rcnn_losses_test",
    srcs = ["rcnn_losses_test.py"],
    deps = [
        ":rcnn_losses",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "rcnn_sampler_test",
    srcs = ["rcnn_sampler_test.py"],
    deps = [
        ":rcnn_sampler",
        ":similarity_ops",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "rcnn_test",
    size = "enormous",
    srcs = ["rcnn_test.py"],
    shard_count = 5,
    deps = [
        ":input_detection",
        ":rcnn",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "retinanet_test",
    size = "large",
    srcs = ["retinanet_test.py"],
    shard_count = 4,
    deps = [
        ":input_detection",
        ":retinanet",
        "//axlearn/common:config",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "roi_generator_test",
    srcs = ["roi_generator_test.py"],
    deps = [
        ":anchor",
        ":roi_generator",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "rpn_sampler_test",
    srcs = ["rpn_sampler_test.py"],
    deps = [
        ":rpn_sampler",
        ":similarity_ops",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "samplers_test",
    srcs = ["samplers_test.py"],
    deps = [
        ":samplers",
        "//axlearn/common:test_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "similarity_ops_test",
    srcs = ["similarity_ops_test.py"],
    deps = [
        ":similarity_ops",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "spatial_transform_ops_test",
    srcs = ["spatial_transform_ops_test.py"],
    env = {"JAX_ENABLE_X64": "1"},
    deps = [
        ":spatial_transform_ops",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_library(
    name = "virtex",
    srcs = ["virtex.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":resnet",
        "//axlearn/common:base_layer",
        "//axlearn/common:config",
        "//axlearn/common:decoder",
        "//axlearn/common:decoding",
        "//axlearn/common:layers",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:param_init",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "detection_generator_test",
    srcs = ["detection_generator_test.py"],
    deps = [
        ":anchor",
        ":detection_generator",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "mobilenets_blocks_test",
    srcs = ["mobilenets_blocks_test.py"],
    deps = [
        ":mobilenets_blocks",
        "//axlearn/common:layers",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "nms_test",
    srcs = ["nms_test.py"],
    deps = [
        ":nms",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "virtex_test",
    size = "enormous",
    srcs = ["virtex_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":virtex",
        "//axlearn/common:decoding",
        "//axlearn/common:module",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.vision.attention_test.npz",
    ],
    deps = [
        ":attention",
        "//axlearn/common:attention",
        "//axlearn/common:golden",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "clip_test",
    srcs = ["clip_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.vision.clip_test.npz",
    ],
    deps = [
        ":clip",
        "//axlearn/common:golden",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:text_encoder",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "mobilenets_test",
    size = "large",
    srcs = ["mobilenets_test.py"],
    shard_count = 4,
    deps = [
        ":image_classification",
        ":mobilenets",
        "//axlearn/common:module",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
    ],
)

py_test(
    name = "window_attention_test",
    srcs = ["window_attention_test.py"],
    data = [
        "//axlearn/experiments/testdata:axlearn.vision.window_attention_test.npz",
    ],
    deps = [
        ":window_attention",
        "//axlearn/common:golden",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "image_classification_test",
    srcs = ["image_classification_test.py"],
    deps = [
        ":image_classification",
        ":mobilenets",
        ":resnet",
        "//axlearn/common:config",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "//axlearn/common:vision_transformer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
        "@axlearn_pip//numpy",
        "@axlearn_pip//pytest",
    ],
)

py_test(
    name = "coco_utils_test",
    srcs = ["coco_utils_test.py"],
    deps = [
        ":coco_utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//numpy",
        "@axlearn_pip//tensorflow",
    ],
)

py_test(
    name = "eval_detection_test",
    srcs = ["eval_detection_test.py"],
    deps = [
        ":eval_detection",
        ":input_detection",
        "//axlearn/common:base_model",
        "//axlearn/common:config",
        "//axlearn/common:metrics",
        "//axlearn/common:module",
        "//axlearn/common:test_utils",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)
