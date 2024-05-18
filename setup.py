from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "profiler",
        ["profiler.cpp"],
        extra_compile_args=['-g'],  # For debugging, remove in production
        # extra_objects=['/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.a'],  # Static link
        library_dirs=['/usr/lib/x86_64-linux-gnu'],  # Library directory
        runtime_library_dirs=['/usr/lib/x86_64-linux-gnu'],  # Runtime library directory
        include_dirs=['/usr/include'],  # Include directory, adjust as necessary
        libraries=['tcmalloc_and_profiler'],  # Ensure the profiler library is linked
    ),
]

setup(
    name="profiler",
    version="0.1",
    packages=['pprof'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
