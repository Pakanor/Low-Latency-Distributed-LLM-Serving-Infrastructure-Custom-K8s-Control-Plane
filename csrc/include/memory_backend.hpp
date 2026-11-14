#ifndef LLM_INFRA_MEMORY_BACKEND_HPP
#define LLM_INFRA_MEMORY_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace llm_infra {

// Hardware Abstraction Layer for the KV-cache page allocator.
//
// The PageAllocator owns pure scheduling logic: block tables, free lists,
// ref-counting, and swap bookkeeping. It never touches raw CPU/GPU memory
// directly — every allocation, deallocation, and transfer goes through a
// MemoryBackend. This keeps the scheduler identical across:
//   - CPU (CpuMemoryBackend)
//   - AMD Unified Memory (HipZeroCopyMemoryBackend)
//   - NVIDIA dGPU (CudaMemoryBackend — to be added in the future)
//
// FUTURE: when a CudaMemoryBackend is plugged in, copy_*() should eventually
// call cudaMemcpyAsync with a CUDA stream. For now the synchronous path is
// correct and testable; the async upgrade is a backend-only change.
class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;

    MemoryBackend() = default;

    MemoryBackend(const MemoryBackend&) = delete;
    MemoryBackend& operator=(const MemoryBackend&) = delete;

   
    virtual void* allocate_host_block(size_t block_size_bytes) = 0;

    virtual void free_host_block(void* ptr) = 0;

   
    virtual void* allocate_device_block(size_t block_size_bytes) = 0;

    virtual void free_device_block(void* ptr) = 0;

    // Copy block_size_bytes from src to dst.
    //
    // Direction is caller-chosen: the backend picks the appropriate
    // memcpy kind (host-to-host, device-to-host, host-to-device,
    // device-to-device, or no-op on zero-copy).
    //
    // `stream` is reserved for asynchronous backends (e.g. CUDA streams).
    // Pass nullptr for synchronous behaviour. When a non-null stream is
    // provided the copy may return before the data is visible; the caller
    // must call synchronize_stream(stream) before reading the destination.
    virtual void copy_block(void* dst, const void* src, size_t block_size_bytes,
                            void* stream) = 0;

    // Block until all previously queued work on `stream` has completed.
    // `stream` may be nullptr to mean "the default stream / all streams".
    virtual void synchronize_stream(void* stream) = 0;

    virtual bool is_zero_copy() const = 0;

    virtual const char* name() const = 0;
};

}  

#endif  // LLM_INFRA_MEMORY_BACKEND_HPP