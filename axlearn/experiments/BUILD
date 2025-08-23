load("//pytype_bazel:pytype.bzl", "pytype_library", "pytype_test")

pytype_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "@pip//absl_py",
    ],
)

pytype_library(
    name = "trainer_config_utils",
    srcs = ["trainer_config_utils.py"],
    enable_pytype = False,  # flash_attention.layer not yet in BUILD system
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "@pip//cloudpickle",
    ],
)

pytype_test(
    name = "trainer_config_utils_test",
    srcs = ["trainer_config_utils_test.py"],
    enable_pytype = False,  # flash_attention.layer not yet in BUILD system
    deps = [
        ":trainer_config_utils",
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "@pip//absl_py",
        "@pip//jax",
        "@pip//numpy",
    ],
)

pytype_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    enable_pytype = False,  # huggingface module not yet in BUILD system
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:summary_writer",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pytest",
    ],
)

pytype_library(
    name = "logistic_regression_tutorial",
    srcs = ["logistic_regression_tutorial.py"],
    enable_pytype = False,  # Complex dependencies - focus on functionality
    visibility = ["//visibility:public"],
    deps = [
        ":trainer_config_utils",
        "//axlearn/common:base_model",
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:input_fake",
        "//axlearn/common:input_grain",
        "//axlearn/common:layers",
        "//axlearn/common:learner",
        "//axlearn/common:module",
        "//axlearn/common:optimizers",
        "//axlearn/common:schedule",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "@pip//jax",
        "@pip//numpy",
    ],
)

# Filegroup containing all experiment modules for runtime access
filegroup(
    name = "all_experiments",
    srcs = [
        ":__init__",
        ":logistic_regression_tutorial",
        # Add other experiment modules here as they get Bazelized
    ],
    visibility = ["//visibility:public"],
)
