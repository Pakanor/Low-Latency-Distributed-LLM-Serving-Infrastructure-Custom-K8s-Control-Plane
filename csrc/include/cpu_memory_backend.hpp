#ifndef LLM_INFRA_CPU_MEMORY_BACKEND_HPP
#define LLM_INFRA_CPU_MEMORY_BACKEND_HPP

#include "memory_backend.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llm_infra {

// CPU memory backend — the default for the current CPU-only build.
//
// Both host and device blocks live in the same process heap. On this
// platform "device" is a logical concept (the paged attention kernel
// reads from the same memory the KV cache manager writes to), so
// allocate_device_block() returns host memory and copy_block() is a
// plain std::memcpy.
class CpuMemoryBackend : public MemoryBackend {
public:
  
    void* allocate_host_block(size_t block_size_bytes) override {
        void* ptr = nullptr;
        // Page-align host blocks to keep the KV cache DMA/TLB-friendly
        // and to match the old posix_memalign behaviour exactly.
        int ret = posix_memalign(&ptr, kAlignment, block_size_bytes);
        if (ret != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void free_host_block(void* ptr) override {
            free(ptr);
    }

    void* allocate_device_block(size_t block_size_bytes) override {
        // On CPU the "device" pool is host memory — return a fresh
        // page-aligned allocation so the two pools remain independently
        // base_host_ptr_ was one big block but cpu_block_ptrs_ were
        // computed offsets; here each block is individually owned).
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, kAlignment, block_size_bytes);
        if (ret != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void free_device_block(void* ptr) override {
            free(ptr);
    }

    void copy_block(void* dst, const void* src, size_t block_size_bytes,
                    void* /*stream*/) override {
        std::memcpy(dst, src, block_size_bytes);
    }

    void synchronize_stream(void* /*stream*/) override {
        // CPU backend: memcpy is synchronous
    }

    bool is_zero_copy() const override {
        // On CPU host == device address space, so the kernel can read
        // the KV cache directly without an explicit transfer.
        return true;
    }

    const char* name() const override {
        return "CpuMemoryBackend";
    }

private:
    static constexpr size_t kAlignment = 4096;
};

}  // namespace llm_infra

#endif  // LLM_INFRA_CPU_MEMORY_BACKEND_HPP