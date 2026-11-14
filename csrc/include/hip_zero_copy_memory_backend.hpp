#ifndef LLM_INFRA_HIP_ZERO_COPY_MEMORY_BACKEND_HPP
#define LLM_INFRA_HIP_ZERO_COPY_MEMORY_BACKEND_HPP

#include "memory_backend.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// HIP / AMD Unified Memory backend.
//
// Uses hipHostMalloc(hipHostMallocMapped) so the host allocation is
// accessible from the GPU without an explicit copy. If the mapped
// allocation fails (or HIP is unavailable at compile time) we fall back
// to plain page-aligned host memory — the allocator still works, it just
// loses the zero-copy fast path.

#if LLM_INFRA_HIP_ENABLED
#include <hip/hip_runtime_api.h>
#endif

namespace llm_infra {

class HipZeroCopyMemoryBackend : public MemoryBackend {
public:
    HipZeroCopyMemoryBackend() {
#if LLM_INFRA_HIP_ENABLED
        void* probe = nullptr;
        hipError_t err = hipHostMalloc(&probe, 64, hipHostMallocMapped);
        if (err == hipSuccess && probe != nullptr) {
            void* dev_view = nullptr;
            hipError_t view_err = hipHostGetDevicePointer(&dev_view, probe, 0);
            if (view_err == hipSuccess && dev_view != nullptr) {
                hipFreeHost(probe);
                zero_copy_available_ = true;
            } else {
                hipFreeHost(probe);
                zero_copy_available_ = false;
            }
        } else {
            zero_copy_available_ = false;
        }
#else
        zero_copy_available_ = false;
#endif
    }

    void* allocate_host_block(size_t block_size_bytes) override {
        void* ptr = nullptr;
#if LLM_INFRA_HIP_ENABLED
        if (zero_copy_available_) {
            hipError_t err = hipHostMalloc(&ptr, block_size_bytes, hipHostMallocMapped);
            if (err == hipSuccess && ptr != nullptr) {
                return ptr;
            }
            // Fall through to plain host allocation on failure.
            std::cerr << "[llm_infra] hipHostMalloc mapped failed: "
                      << hipGetErrorString(err) << " - using plain host memory\n";
        }
#endif
        int ret = posix_memalign(&ptr, kAlignment, block_size_bytes);
        if (ret != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void free_host_block(void* ptr) override {
        if (ptr == nullptr) {
            return;
        }
#if LLM_INFRA_HIP_ENABLED
        if (zero_copy_available_) {
            hipError_t err = hipFreeHost(ptr);
            if (err == hipSuccess) {
                return;
            }
        }
#endif
        free(ptr);
    }

    void* allocate_device_block(size_t block_size_bytes) override {
        return allocate_host_block(block_size_bytes);
    }

    void free_device_block(void* ptr) override {
        free_host_block(ptr);
    }

    void copy_block(void* /*dst*/, const void* /*src*/, size_t /*block_size_bytes*/,
                    void* /*stream*/) override {
        // Unified Memory: host and device share the same physical pages,
        // so no explicit copy is required. The data written by the
        // scatter step is already visible to the GPU.
    }

    void synchronize_stream(void* /*stream*/) override {
#if LLM_INFRA_HIP_ENABLED
        hipDeviceSynchronize();
#endif
    }

    bool is_zero_copy() const override {
        return zero_copy_available_;
    }

    const char* name() const override {
        return zero_copy_available_ ? "HipZeroCopyMemoryBackend"
                                    : "HipZeroCopyMemoryBackend (host fallback)";
    }

    
    uintptr_t get_device_view(void* host_ptr) const {
        if (!host_ptr || !zero_copy_available_) {
            return 0;
        }
#if LLM_INFRA_HIP_ENABLED
        void* dev_view = nullptr;
        hipError_t err = hipHostGetDevicePointer(&dev_view, host_ptr, 0);
        if (err == hipSuccess && dev_view != nullptr) {
            return reinterpret_cast<uintptr_t>(dev_view);
        }
#endif
        return 0;
    }

private:
    static constexpr size_t kAlignment = 4096;
    bool zero_copy_available_{false};
};

}  // namespace llm_infra

#endif  // LLM_INFRA_HIP_ZERO_COPY_MEMORY_BACKEND_HPP