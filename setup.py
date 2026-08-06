from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "llm_allocator_cpp",
        ["csrc/src/bindings.cpp"],
        include_dirs=["csrc/include"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="llm_allocator_cpp",
    version="0.1.0",
    description="C++ DRAM KV-Cache Page Allocator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)