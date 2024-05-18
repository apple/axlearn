from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "profiler",
        ["profiler.cpp"],
        # Link tcmalloc_and_profiler to support heap profiler.
        libraries=['tcmalloc_and_profiler'],
    ),
]

setup(
    name="profiler",
    version="0.1",
    packages=['pprof'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
