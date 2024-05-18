# python axlearn/profiler/setup.py build_ext --inplace --force

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "profiler",
        ["axlearn/profiler/profiler.cpp"],
        extra_compile_args=['-g'],  # For debugging, remove in production
        # Link tcmalloc_and_profiler to support heap profiler.
        # libraries= ['profiler'],
        libraries= ['tcmalloc_and_profiler'],
        # library_dirs=['/usr/lib/x86_64-linux-gnu'],  # Library directory
        # runtime_library_dirs=['/usr/lib/x86_64-linux-gnu'],  # Runtime library directory
        # include_dirs=['/usr/include'],  # Include directory, adjust as necessary
        # extra_objects=['/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.a'],  # Static link
    ),
]

setup(
    name="profiler",
    version="0.1",
    packages=['pprof'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
