load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def _pytype_check_script_impl(ctx):
    """Generate a Python script to run pytype."""

    srcs = ctx.files.srcs

    # Create the test script
    test_script = ctx.actions.declare_file(ctx.label.name + ".py")
    ctx.actions.write(
        output = test_script,
        content = '''"""Generated pytype validation."""
import subprocess
import sys
import os

def main():
    """Run pytype on the specified files."""
    # Get the runfiles directory for finding our files
    runfiles_dir = os.environ.get('RUNFILES_DIR', os.path.dirname(__file__))
    workspace_name = '_main'  # Default workspace name

    # Source files to check (find them in runfiles)
    source_files = []
    for src_name in {srcs}:
        # Try to find source file in the runfiles directory
        # First try with the package directory from the label
        label_package = "{package}"
        if label_package:
            src_path = os.path.join(runfiles_dir, workspace_name, label_package, src_name)
            if os.path.exists(src_path):
                source_files.append(src_path)
                continue

        # Fallback: search in workspace root
        src_path = os.path.join(runfiles_dir, workspace_name, src_name)
        if os.path.exists(src_path):
            source_files.append(src_path)
        else:
            # Final fallback: look relative to script
            rel_path = os.path.join(os.path.dirname(__file__), src_name)
            if os.path.exists(rel_path):
                source_files.append(rel_path)
            else:
                print(f"Warning: Could not find source file: {{src_name}}")

    # Set up PYTHONPATH to include dependencies
    python_paths = [
        # Add the workspace root so local imports work
        os.path.join(runfiles_dir, workspace_name),
    ]

    # Add existing PYTHONPATH to get all pip dependencies
    existing_path = os.environ.get('PYTHONPATH', '')
    if existing_path:
        python_paths.append(existing_path)

    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join(python_paths)

    cmd = [sys.executable, "-m", "pytype", "-k", "-v", "1"] + source_files
    print("Running pytype:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Print output for debugging
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print("Type checking " + ("passed" if result.returncode == 0 else "failed"))
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
'''.format(
            srcs = [s.basename for s in srcs],
            package = ctx.label.package,
        ),
    )

    return [DefaultInfo(files = depset([test_script]))]

def pytype_library(name, srcs = None, pytype_srcs = None, enable_pytype = True, **kwargs):
    """Create a py_library with integrated pytype checking.

    Args:
        name: Target name
        srcs: Source files for the library
        pytype_srcs: Source files to type check (defaults to srcs)
        enable_pytype: Whether to enable pytype checking
        **kwargs: Additional arguments passed to py_library
    """

    if enable_pytype and (srcs or pytype_srcs):
        check_srcs = pytype_srcs if pytype_srcs else srcs
        if check_srcs:
            # Generate the pytype check script
            script_name = name + "_pytype_script"
            _pytype_check_script(
                name = script_name,
                srcs = check_srcs,
            )

            # Create pytype check test (runs during build)
            # Include the original deps so pytype can find imported modules
            pytype_deps = [requirement("pytype")]
            if "deps" in kwargs:
                pytype_deps.extend(kwargs["deps"])

            py_test(
                name = name + "_pytype_check",
                srcs = [":" + script_name],
                main = script_name + ".py",
                data = check_srcs,
                deps = pytype_deps,
                size = "medium",  # 300s timeout for type checking
                tags = ["pytype"],
            )

    # Create the original py_library
    py_library(
        name = name,
        srcs = srcs,
        **kwargs
    )

def pytype_binary(name, srcs = None, pytype_srcs = None, enable_pytype = True, **kwargs):
    """Create a py_binary with integrated pytype checking.

    Args:
        name: Target name
        srcs: Source files for the binary
        pytype_srcs: Source files to type check (defaults to srcs)
        enable_pytype: Whether to enable pytype checking
        **kwargs: Additional arguments passed to py_binary
    """

    if enable_pytype and (srcs or pytype_srcs):
        check_srcs = pytype_srcs if pytype_srcs else srcs
        if check_srcs:
            # Generate the pytype check script
            script_name = name + "_pytype_script"
            _pytype_check_script(
                name = script_name,
                srcs = check_srcs,
            )

            # Create pytype check test (runs during build)
            # Include the original deps so pytype can find imported modules
            pytype_deps = [requirement("pytype")]
            if "deps" in kwargs:
                pytype_deps.extend(kwargs["deps"])

            py_test(
                name = name + "_pytype_check",
                srcs = [":" + script_name],
                main = script_name + ".py",
                data = check_srcs,
                deps = pytype_deps,
                size = "medium",  # 300s timeout for type checking
                tags = ["pytype"],
            )

    # Create the original py_binary
    py_binary(
        name = name,
        srcs = srcs,
        **kwargs
    )

def pytype_test(name, srcs = None, pytype_srcs = None, enable_pytype = True, **kwargs):
    """Create a py_test with integrated pytype checking.

    Args:
        name: Target name
        srcs: Source files for the test
        pytype_srcs: Source files to type check (defaults to srcs)
        enable_pytype: Whether to enable pytype checking
        **kwargs: Additional arguments passed to py_test
    """

    if enable_pytype and (srcs or pytype_srcs):
        check_srcs = pytype_srcs if pytype_srcs else srcs
        if check_srcs:
            # Generate the pytype check script
            script_name = name + "_pytype_script"
            _pytype_check_script(
                name = script_name,
                srcs = check_srcs,
            )

            # Create pytype check test (runs during build)
            # Include the original deps so pytype can find imported modules
            pytype_deps = [requirement("pytype")]
            if "deps" in kwargs:
                pytype_deps.extend(kwargs["deps"])

            py_test(
                name = name + "_pytype_check",
                srcs = [":" + script_name],
                main = script_name + ".py",
                data = check_srcs,
                deps = pytype_deps,
                size = "medium",  # 300s timeout for type checking
                tags = ["pytype"],
            )

    # Create the original py_test
    py_test(
        name = name,
        srcs = srcs,
        **kwargs
    )

_pytype_check_script = rule(
    implementation = _pytype_check_script_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".py"],
            mandatory = True,
        ),
    },
)
