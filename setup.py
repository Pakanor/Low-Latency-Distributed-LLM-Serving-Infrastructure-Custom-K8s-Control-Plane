import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _detect_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _rocm_build_enabled() -> bool:
    if os.getenv("LLM_ALLOCATOR_NO_HIP", "").lower() in ("1", "true", "yes"):
        return False
    torch = _detect_torch()
    if torch is not None and getattr(torch.version, "hip", None) is not None:
        return True
    return False


def _cuda_available() -> str | None:
    torch = _detect_torch()
    if torch is not None and getattr(torch.version, "cuda", None) is not None:
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            if CUDA_HOME:
                return CUDA_HOME
        except Exception:
            pass
        return os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    return None


def _rocm_home() -> str | None:
    if os.getenv("LLM_ALLOCATOR_NO_HIP", "").lower() in ("1", "true", "yes"):
        return None
    home = os.getenv("ROCM_HOME") or os.getenv("ROCM_PATH")
    if home:
        return home
    torch = _detect_torch()
    if torch is None or getattr(torch.version, "hip", None) is None:
        return None
    return os.getenv("HIP_HOME") or "/opt/rocm"


def _hip_compile_args(rocm_home: str) -> list[str]:
    offload_arch = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if offload_arch:
        return ["-O3", "-std=c++17", f"--offload-arch={offload_arch}"]
    return ["-O3", "-std=c++17", "--offload-arch=gfx1100"]


def _build_extension():
    rocm_home = _rocm_home()
    cuda_home = _cuda_available()

    if rocm_home:
        hipcc = os.path.join(rocm_home, "bin", "hipcc")
        if not os.path.isfile(hipcc):
            rocm_home = None

    if rocm_home:
        print(f"[llm_allocator_cpp] ROCm/HIP enabled (hipcc: {hipcc})")
        print("[llm_allocator_cpp] Native paged-attention GPU kernel: csrc/src/paged_kernels.cu")
        return Pybind11Extension(
            "llm_allocator_cpp",
            ["csrc/src/bindings.cpp", "csrc/src/paged_kernels.cu"],
            include_dirs=[os.path.join(rocm_home, "include"), "csrc/include"],
            library_dirs=[os.path.join(rocm_home, "lib")],
            libraries=["amdhip64"],
            define_macros=[("LLM_INFRA_HIP_ENABLED", "1")],
            extra_compile_args=_hip_compile_args(rocm_home),
        )

    if cuda_home:
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(nvcc):
            from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension
            print(f"[llm_allocator_cpp] CUDA enabled (nvcc: {nvcc})")
            print("[llm_allocator_cpp] Native paged-attention GPU kernel: csrc/src/paged_kernels.cu")
            gpu_archs = os.getenv("GPU_ARCHS") or "70;80;90"
            return TorchCUDAExtension(
                "llm_allocator_cpp",
                ["csrc/src/bindings.cpp", "csrc/src/paged_kernels.cu"],
                include_dirs=["csrc/include"],
                define_macros=[("LLM_INFRA_HIP_ENABLED", "1")],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
                },
            )

    print("[llm_allocator_cpp] CPU-only build (LLM_INFRA_HIP_ENABLED=0)")
    print("[llm_allocator_cpp] Native paged-attention CPU kernel: csrc/src/paged_kernels.cpp")
    return Pybind11Extension(
        "llm_allocator_cpp",
        ["csrc/src/bindings.cpp", "csrc/src/paged_kernels.cpp"],
        include_dirs=["csrc/include"],
        define_macros=[("LLM_INFRA_HIP_ENABLED", "0")],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
    )


cuda_home = _cuda_available()
cmdclass = {"build_ext": build_ext}
if cuda_home and os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
    from torch.utils.cpp_extension import BuildExtension
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="llm_allocator_cpp",
    version="0.1.0",
    description="C++ DRAM KV-Cache Page Allocator",
    ext_modules=[_build_extension()],
    cmdclass=cmdclass,
)
