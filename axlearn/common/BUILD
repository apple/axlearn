load("//pytype_bazel:pytype.bzl", "pytype_library", "pytype_test")

pytype_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

pytype_library(
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
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "array_serialization",
    srcs = ["array_serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "base_model",
    srcs = ["base_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":module",
    ],
)

pytype_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":array_serialization",
        ":config",
        ":file_system",
        ":metrics",
        ":module",
        ":summary_writer",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//msgpack",
        "@pip//tensorflow",
    ],
)

pytype_library(
    name = "compiler_options",
    srcs = ["compiler_options.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//attrs",
    ],
)

pytype_library(
    name = "factorized_rms",
    srcs = ["factorized_rms.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":optimizer_base",
        ":schedule",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
        "@pip//optax",
    ],
)

pytype_library(
    name = "file_system",
    srcs = ["file_system.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//absl_py",
        "@pip//tensorflow",
    ],
)

pytype_library(
    name = "device_monitor",
    srcs = ["monitoring/device_monitor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "evaler",
    srcs = ["evaler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_model",
        ":config",
        ":inference_output",
        ":input_base",
        ":metrics",
        ":module",
        ":struct",
        ":summary_writer",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "inference_output",
    srcs = ["inference_output.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":module",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
        "@pip//tensorflow",
    ],
)

pytype_library(
    name = "input_base",
    srcs = ["input_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_dispatch",
        ":module",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "input_dispatch",
    srcs = ["input_dispatch.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pytest",
        "@pip//wrapt",
    ],
)

pytype_library(
    name = "layers",
    srcs = ["layers.py"],
    enable_pytype = False,  # Type annotation errors require code fixes
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":param_init",
    ],
)

pytype_library(
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
        "@pip//jax",
        "@pip//optax",
    ],
)

pytype_library(
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

pytype_library(
    name = "param_init",
    srcs = ["param_init.py"],
    enable_pytype = False,  # Type annotation errors require code fixes
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
    ],
)

pytype_library(
    name = "state_builder",
    srcs = ["state_builder.py"],
    enable_pytype = False,  # Missing imports and type annotation errors
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":module",
        ":utils",
    ],
)

pytype_library(
    name = "schedule",
    srcs = ["schedule.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":utils",
        "@pip//jax",
        "@pip//optax",
    ],
)

pytype_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":serialization",
        "@pip//absl_py",
        "@pip//attrs",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":summary",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "measurement",
    srcs = ["measurement.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "module",
    srcs = ["module.py"],
    enable_pytype = False,  # Complex forward references and custom dataclass patterns
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":struct",
        ":summary",
        ":traceback_util",
        ":utils",
    ],
)

pytype_library(
    name = "optimizer_base",
    srcs = ["optimizer_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":utils",
        "@pip//optax",
        "@pip//typing_extensions",
    ],
)

pytype_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":factorized_rms",
        ":module",
        ":optimizer_base",
        ":schedule",
        ":struct",
        ":utils",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//jax",
        "@pip//optax",
        "@pip//typing_extensions",
    ],
)

pytype_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//jax",
    ],
)

pytype_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    enable_pytype = False,  # Type error: No attribute 'replace' on _Box
    deps = [
        ":serialization",
        ":struct",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
        "@pip//optax",
        "@pip//pytest",
    ],
)

pytype_library(
    name = "struct",
    srcs = ["struct.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":serialization",
        ":utils",
        "@pip//jax",
        "@pip//typing_extensions",
    ],
)

pytype_test(
    name = "struct_test",
    size = "small",
    srcs = ["struct_test.py"],
    deps = [
        ":struct",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//jax",
        "@pip//jaxlib",
        "@pip//pytest",
    ],
)

pytype_library(
    name = "summary",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":struct",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "summary_writer",
    srcs = ["summary_writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":file_system",
        ":module",
        ":summary",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
        "@pip//tensorflow",
    ],
)

pytype_library(
    name = "traceback_util",
    srcs = ["traceback_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "trainer",
    srcs = ["trainer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":base_model",
        ":checkpointer",
        ":compiler_options",
        ":config",
        ":device_monitor",
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
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "update_transformation",
    srcs = ["update_transformation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_layer",
        ":config",
        ":learner_base",
        ":module",
        ":optimizer_base",
        ":struct",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//optax",
    ],
)

pytype_library(
    name = "status_server",
    srcs = ["status_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "utils_spmd",
    srcs = ["utils_spmd.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//jax",
        "@pip//portpicker",
    ],
)

pytype_library(
    name = "launch",
    srcs = ["launch.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_options",
        ":status_server",
        ":utils",
        ":utils_spmd",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "launch_trainer",
    srcs = ["launch_trainer.py"],
    enable_pytype = False,  # Avoid circular dependency with experiments
    visibility = ["//visibility:public"],
    deps = [
        ":file_system",
        ":measurement",
        ":trainer",
        ":utils",
        "@pip//absl_py",
        "@pip//jax",
    ],
)

pytype_library(
    name = "launch_trainer_main_lib",
    srcs = ["launch_trainer_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":launch",
        ":launch_trainer",
        ":measurement",
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "input_fake",
    srcs = ["input_fake.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":utils",
        "@pip//jax",
        "@pip//numpy",
        "@pip//tensorflow",
    ],
)

pytype_library(
    name = "input_grain",
    srcs = ["input_grain.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":input_base",
        ":module",
        ":utils",
        "@pip//absl_py",
        "@pip//array_record",
        "@pip//grain",
        "@pip//jax",
        "@pip//numpy",
    ],
)
