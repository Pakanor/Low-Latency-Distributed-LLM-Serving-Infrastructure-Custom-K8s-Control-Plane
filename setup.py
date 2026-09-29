import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _rocm_home():
    if os.getenv("LLM_ALLOCATOR_NO_HIP", "").lower() in ("1", "true", "yes"):
        return None
    home = os.getenv("ROCM_HOME") or os.getenv("ROCM_PATH")
    if home:
        return home
    try:
        import torch
    except ImportError:
        return None
    if getattr(torch.version, "hip", None) is None:
        return None
    return os.getenv("HIP_HOME") or "/opt/rocm"


def _hip_extension_kwargs(rocm_home):
    if not rocm_home:
        return None
    hipcc = os.path.join(rocm_home, "bin", "hipcc")
    if not os.path.isfile(hipcc):
        return None

    include_dirs = [os.path.join(rocm_home, "include")]
    library_dirs = [os.path.join(rocm_home, "lib")]
    extra_compile_args = ["-O3", "-std=c++17"]

    offload_arch = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if offload_arch:
        extra_compile_args.append(f"--offload-arch={offload_arch}")

    return {
        "compiler": hipcc,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": ["amdhip64"],
        "define_macros": [("LLM_INFRA_HIP_ENABLED", "1")],
        "extra_compile_args": extra_compile_args,
    }


def _build_extension():
    hip_kwargs = _hip_extension_kwargs(_rocm_home())
    if hip_kwargs is None:
        return Pybind11Extension(
            "llm_allocator_cpp",
            ["csrc/src/bindings.cpp"],
            include_dirs=["csrc/include"],
            define_macros=[("LLM_INFRA_HIP_ENABLED", "0")],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    print(f"[llm_allocator_cpp] HIP zero-copy pool enabled (hipcc: {hip_kwargs['compiler']})")
    return Pybind11Extension(
        "llm_allocator_cpp",
        ["csrc/src/bindings.cpp"],
        include_dirs=hip_kwargs["include_dirs"] + ["csrc/include"],
        library_dirs=hip_kwargs["library_dirs"],
        libraries=hip_kwargs["libraries"],
        define_macros=hip_kwargs["define_macros"],
        extra_compile_args=hip_kwargs["extra_compile_args"],
    )


setup(
    name="llm_allocator_cpp",
    version="0.1.0",
    description="C++ DRAM KV-Cache Page Allocator",
    ext_modules=[_build_extension()],
    cmdclass={"build_ext": build_ext},
)
