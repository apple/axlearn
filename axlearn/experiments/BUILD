load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "trainer_config_utils",
    srcs = ["trainer_config_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:utils",
        "//axlearn/common/flash_attention:layer",
        "@axlearn_pip//cloudpickle",
    ],
)

py_test(
    name = "trainer_config_utils_test",
    srcs = ["trainer_config_utils_test.py"],
    deps = [
        ":trainer_config_utils",
        "//axlearn/common:config",
        "//axlearn/common:input_fake",
        "//axlearn/common:trainer_test_lib",
        "//axlearn/common/flash_attention:layer",
        "//axlearn/common/flash_attention:layer_test_lib",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//axlearn/common:checkpointer",
        "//axlearn/common:config",
        "//axlearn/common:evaler",
        "//axlearn/common:module",
        "//axlearn/common:summary_writer",
        "//axlearn/common:test_utils",
        "//axlearn/common:trainer",
        "//axlearn/common:utils",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//chex",
        "@axlearn_pip//jax",
        "@axlearn_pip//pytest",
    ],
)

# Filegroup containing all experiment modules for runtime access
filegroup(
    name = "all_experiments",
    srcs = [
        ":__init__",
        "//axlearn/experiments/logistic_regression:tutorial",
        # Add other experiment modules here as they get Bazelized
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "golden_ckpt_test",
    srcs = ["golden_ckpt_test.py"],
    # manual: needs checkpoint data files not available in the Bazel sandbox. See #2385.
    tags = ["manual"],
    deps = [
        "//axlearn/common:config",
        "//axlearn/common:inference",
        "//axlearn/common:summary_writer",
        "//axlearn/common:test_utils",
        "//axlearn/common:trainer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//jax",
    ],
)

py_test(
    name = "aot_test",
    srcs = ["aot_test.py"],
    # manual: @pytest.mark.skip("CI does not support AOT compilation.") See #2385.
    tags = ["manual"],
    deps = [
        ":test_utils",
        "//axlearn/common:aot_compilation",
        "//axlearn/common:trainer",
        "//axlearn/experiments/text/gpt:c4_trainer",
        "@axlearn_pip//absl_py",
        "@axlearn_pip//pytest",
    ],
)
