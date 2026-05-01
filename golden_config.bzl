"""Bazel rules for AXLearn."""

load("@rules_python//python:defs.bzl", "py_test")

def golden_config_test(name, module, data = [], match = None, checks = None, deps = [], **kwargs):
    """Creates a golden config test for an experiment module.

    Args:
        name: Name of the test target.
        module: Label of the py_library containing named_trainer_configs().
        data: Data files (golden config files).
        match: Optional regex to filter config names.
        checks: Optional list of check names (e.g., ["check", "check_init"]).
            Defaults to ["check", "check_init"].
        deps: Additional deps beyond module and golden_config.
        **kwargs: Additional arguments passed to py_test.
    """
    script_name = name + "_main.py"

    _generate_golden_test_script(
        name = name + "_gen",
        module = module,
        output = script_name,
        match = match or "",
        checks = checks or [],
    )

    py_test(
        name = name,
        srcs = [script_name],
        data = data,
        deps = [module, "//axlearn/common:golden_config"] + deps,
        main = script_name,
        **kwargs
    )

def _generate_golden_test_script_impl(ctx):
    module = ctx.attr.module
    module_path = module.label.package.replace("/", ".")
    if module.label.name != module.label.package.split("/")[-1]:
        module_path = module_path + "." + module.label.name

    # Testdata subdir name: full dotted module path
    testdata_subdir = module_path

    # Relative path from generated script to experiments/testdata/
    # Script is at {package}/{name}_main.py, testdata is at axlearn/experiments/testdata/
    package_parts = module.label.package.split("/")
    # Count dirs from package to axlearn/experiments/
    # e.g., axlearn/experiments/text/gpt -> need ../../testdata/
    experiments_idx = -1
    for i, part in enumerate(package_parts):
        if part == "experiments":
            experiments_idx = i
            break
    if experiments_idx >= 0:
        depth = len(package_parts) - experiments_idx - 1
        rel_testdata = "/".join([".."] * depth) + "/testdata/" + testdata_subdir
    else:
        rel_testdata = "testdata/" + testdata_subdir

    match_arg = ""
    if ctx.attr.match:
        match_arg = ', match="{}"'.format(ctx.attr.match)

    checks_arg = ""
    if ctx.attr.checks:
        checks_list = ", ".join(["golden_config." + c for c in ctx.attr.checks])
        checks_arg = ", checks=({},)".format(checks_list)

    module_name = module_path.split(".")[-1]
    parent_module = ".".join(module_path.split(".")[:-1])

    content = """\
import os
from axlearn.common import golden_config
from {parent_module} import {module_name}

if __name__ == "__main__":
    golden_config.test_main(
        {module_name},
        os.path.join(os.path.dirname(__file__), "{rel_testdata}"){match_arg}{checks_arg},
    )
""".format(
        parent_module = parent_module,
        module_name = module_name,
        rel_testdata = rel_testdata,
        match_arg = match_arg,
        checks_arg = checks_arg,
    )

    ctx.actions.write(output = ctx.outputs.output, content = content)

_generate_golden_test_script = rule(
    implementation = _generate_golden_test_script_impl,
    attrs = {
        "module": attr.label(
            mandatory = True,
            providers = [PyInfo],
        ),
        "output": attr.output(mandatory = True),
        "match": attr.string(default = ""),
        "checks": attr.string_list(default = []),
    },
)
