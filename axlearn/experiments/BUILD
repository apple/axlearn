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
