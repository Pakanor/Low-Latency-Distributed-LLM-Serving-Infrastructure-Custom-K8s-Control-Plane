#ifndef PAGE_ALLOCATOR_HPP
#define PAGE_ALLOCATOR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <string>

#ifndef LLM_INFRA_HIP_ENABLED
#define LLM_INFRA_HIP_ENABLED 0
#endif

#include "memory_backend.hpp"
#include "cpu_memory_backend.hpp"
#include "hip_zero_copy_memory_backend.hpp"

namespace llm_infra {

inline std::unique_ptr<MemoryBackend> make_default_memory_backend() {
#if LLM_INFRA_HIP_ENABLED
    return std::make_unique<HipZeroCopyMemoryBackend>();
#else
    return std::make_unique<CpuMemoryBackend>();
#endif
}

struct Block {
    int id;
    int ref_count{0};
};

class PageAllocator {
public:
    PageAllocator(size_t total_blocks, size_t tokens_per_block, size_t bytes_per_token,
                  size_t total_cpu_blocks = 32,
                  std::unique_ptr<MemoryBackend> backend = nullptr)
        : total_blocks_(total_blocks),
          tokens_per_block_(tokens_per_block),
          bytes_per_token_(bytes_per_token),
          total_cpu_blocks_(total_cpu_blocks),
          backend_(backend ? std::move(backend) : make_default_memory_backend()) {
        if (tokens_per_block == 0) {
            throw std::invalid_argument("tokens_per_block must be greater than 0");
        }
        if (bytes_per_token == 0) {
            throw std::invalid_argument("bytes_per_token must be greater than 0");
        }

        size_t block_size_bytes = tokens_per_block_ * bytes_per_token_;
        if (total_cpu_blocks_ > 0) {
            size_t total_ram_bytes = total_cpu_blocks_ * block_size_bytes;
            base_host_ptr_ = static_cast<uint8_t*>(backend_->allocate_host_block(total_ram_bytes));
            if (!base_host_ptr_) {
                throw std::bad_alloc();
            }
        }

        for (size_t i = 0; i < total_blocks_; ++i) {
            free_blocks_.push(static_cast<int>(i));
            blocks_.push_back(Block{static_cast<int>(i), 0});
        }

        cpu_block_ptrs_.resize(total_cpu_blocks_);
        for (size_t i = 0; i < total_cpu_blocks_; ++i) {
            cpu_free_blocks_.push_back(static_cast<int>(i));
            cpu_block_ptrs_[i] = base_host_ptr_ + (i * block_size_bytes);
        }
    }

    ~PageAllocator() {
        if (base_host_ptr_) {
            backend_->free_host_block(base_host_ptr_);
            base_host_ptr_ = nullptr;
        }
    }
    PageAllocator(const PageAllocator&) = delete;
    PageAllocator& operator=(const PageAllocator&) = delete;

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            throw std::runtime_error("Out of Memory: No free VRAM pages available in KV-Cache pool!");
        }
        int block_id = free_blocks_.front();
        free_blocks_.pop();
        blocks_[block_id].ref_count = 1;
        return block_id;
    }

    std::vector<int> allocate_blocks(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.size() < count) {
            throw std::runtime_error("Out of Memory: Not enough free blocks available in KV-Cache pool!");
        }
        std::vector<int> allocated;
        allocated.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            int block_id = free_blocks_.front();
            free_blocks_.pop();
            blocks_[block_id].ref_count = 1;
            allocated.push_back(block_id);
        }
        return allocated;
    }

    void share_block(int block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(block_id);
        if (blocks_[block_id].ref_count <= 0) {
            throw std::runtime_error("Cannot share unallocated block " + std::to_string(block_id));
        }
        blocks_[block_id].ref_count++;
    }

    void free_block(int block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(block_id);

        if (blocks_[block_id].ref_count <= 0) {
            auto swapped = gpu_to_cpu_map_.find(block_id);
            if (swapped != gpu_to_cpu_map_.end()) {
                cpu_free_blocks_.push_back(swapped->second);
                gpu_to_cpu_map_.erase(swapped);
                return;
            }
            throw std::runtime_error("Double-free or free on unallocated block detected: ID " + std::to_string(block_id));
        }

        blocks_[block_id].ref_count--;
        if (blocks_[block_id].ref_count == 0) {
            free_blocks_.push(block_id);
        }
    }

    int swap_out(int gpu_block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);

        if (blocks_[gpu_block_id].ref_count != 1) {
            throw std::runtime_error("Cannot swap out block with ref_count != 1");
        }

        if (cpu_free_blocks_.empty()) {
            throw std::runtime_error("OOM: Host RAM Block Pool exhausted during swap out");
        }

        if (gpu_to_cpu_map_.find(gpu_block_id) != gpu_to_cpu_map_.end()) {
            throw std::runtime_error("Block is already swapped out: " + std::to_string(gpu_block_id));
        }

        int cpu_block_id = cpu_free_blocks_.back();
        cpu_free_blocks_.pop_back();

        gpu_to_cpu_map_[gpu_block_id] = cpu_block_id;

        blocks_[gpu_block_id].ref_count = 0;
        free_blocks_.push(gpu_block_id);

        return cpu_block_id;
    }

    int swap_in(int gpu_block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);

        auto it = gpu_to_cpu_map_.find(gpu_block_id);
        if (it == gpu_to_cpu_map_.end()) {
            throw std::runtime_error("Block ID " + std::to_string(gpu_block_id) + " not found in Host RAM mapping");
        }

        if (free_blocks_.empty()) {
            throw std::runtime_error("OOM: No free VRAM blocks available to swap in block " + std::to_string(gpu_block_id));
        }

        int new_vram_block = free_blocks_.front();
        free_blocks_.pop();
        blocks_[new_vram_block].ref_count = 1;

        int cpu_block_id = it->second;
        cpu_free_blocks_.push_back(cpu_block_id);
        gpu_to_cpu_map_.erase(it);

        return new_vram_block;
    }

    int get_cpu_block_id(int gpu_block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);
        auto it = gpu_to_cpu_map_.find(gpu_block_id);
        if (it == gpu_to_cpu_map_.end()) {
            throw std::runtime_error("Block " + std::to_string(gpu_block_id) + " is not swapped out");
        }
        return it->second;
    }

    void release_swapped_block(int gpu_block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);
        auto it = gpu_to_cpu_map_.find(gpu_block_id);
        if (it == gpu_to_cpu_map_.end()) {
            throw std::runtime_error("Block " + std::to_string(gpu_block_id) + " is not swapped out");
        }
        cpu_free_blocks_.push_back(it->second);
        gpu_to_cpu_map_.erase(it);
    }

    uintptr_t get_host_ptr(int gpu_block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);
        auto it = gpu_to_cpu_map_.find(gpu_block_id);
        if (it == gpu_to_cpu_map_.end()) {
            return 0;
        }
        return reinterpret_cast<uintptr_t>(cpu_block_ptrs_[it->second]);
    }

    size_t get_block_size() const { return tokens_per_block_; }

    size_t get_bytes_per_token() const { return bytes_per_token_; }

    size_t get_host_pool_bytes() const { return total_cpu_blocks_ * tokens_per_block_ * bytes_per_token_; }

    size_t get_num_free_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }

    size_t get_num_free_cpu_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cpu_free_blocks_.size();
    }

    size_t get_total_blocks() const { return total_blocks_; }

    void dump_state(std::ostream& os) const {
        std::lock_guard<std::mutex> lock(mutex_);
        os << "total_blocks=" << total_blocks_
           << " free_blocks=" << free_blocks_.size()
           << " cpu_free=" << cpu_free_blocks_.size()
           << " cpu_total=" << total_cpu_blocks_
           << " swapped=" << gpu_to_cpu_map_.size()
           << " ref_counts>0={";
        bool first = true;
        for (size_t i = 0; i < blocks_.size(); ++i) {
            if (blocks_[i].ref_count > 0) {
                if (!first) os << ",";
                os << i << ":" << blocks_[i].ref_count;
                first = false;
            }
        }
        os << "}";
    }

    bool is_zero_copy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return backend_->is_zero_copy();
    }

    uintptr_t get_device_ptr(int gpu_block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);
        if (!backend_->is_zero_copy() || !base_host_ptr_) {
            return 0;
        }
        if (auto* hip_backend = dynamic_cast<const HipZeroCopyMemoryBackend*>(backend_.get())) {
            return hip_backend->get_device_view(base_host_ptr_);
        }
        return reinterpret_cast<uintptr_t>(base_host_ptr_);
    }

    double get_utilization_percentage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (total_blocks_ == 0) return 0.0;
        return 100.0 * (total_blocks_ - free_blocks_.size()) / total_blocks_;
    }

private:
    void validate_block_id(int block_id) const {
        if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
            throw std::out_of_range("Block ID out of range: " + std::to_string(block_id));
        }
    }

    size_t total_blocks_;
    size_t tokens_per_block_;
    size_t bytes_per_token_;
    size_t total_cpu_blocks_;
    uint8_t* base_host_ptr_{nullptr};

    std::unique_ptr<MemoryBackend> backend_;
    std::vector<void*> cpu_block_ptrs_;

    std::vector<Block> blocks_;
    std::queue<int> free_blocks_;
    std::vector<int> cpu_free_blocks_;
    std::unordered_map<int, int> gpu_to_cpu_map_;
    mutable std::mutex mutex_;
};

class SequenceBlockTable {
public:
    explicit SequenceBlockTable(PageAllocator& allocator) 
        : allocator_(&allocator), block_size_(allocator.get_block_size()) {}

    SequenceBlockTable(const SequenceBlockTable&) = delete;
    SequenceBlockTable& operator=(const SequenceBlockTable&) = delete;

    SequenceBlockTable(SequenceBlockTable&& other) noexcept 
        : allocator_(other.allocator_),
          block_size_(other.block_size_),
          tokens_count_(other.tokens_count_),
          physical_blocks_(std::move(other.physical_blocks_)) {
        other.allocator_ = nullptr;
        other.tokens_count_ = 0;
    }

    SequenceBlockTable& operator=(SequenceBlockTable&& other) noexcept {
        if (this != &other) {
            release_internal();
            allocator_ = other.allocator_;
            block_size_ = other.block_size_;
            tokens_count_ = other.tokens_count_;
            physical_blocks_ = std::move(other.physical_blocks_);
            other.allocator_ = nullptr;
            other.tokens_count_ = 0;
        }
        return *this;
    }

    ~SequenceBlockTable() {
        release_internal();
    }

    void append_block(int block_id) {
        validate_block_id(block_id);
        physical_blocks_.push_back(block_id);
    }

    void replace_block(int old_block_id, int new_block_id) {
        validate_block_id(new_block_id);
        auto it = std::find(physical_blocks_.begin(), physical_blocks_.end(), old_block_id);
        if (it == physical_blocks_.end()) {
            throw std::runtime_error("Block ID " + std::to_string(old_block_id) + " not found in block table");
        }
        *it = new_block_id;
    }

    void append_token(PageAllocator& allocator) {
        if (tokens_count_ % block_size_ == 0) {
            int new_block = allocator.allocate_block();
            physical_blocks_.push_back(new_block);
        }
        tokens_count_++;
    }

    void release() {
        release_internal();
    }

    const std::vector<int>& get_physical_blocks() const { return physical_blocks_; }
    size_t get_tokens_count() const { return tokens_count_; }

private:
    void validate_block_id(int block_id) const {
        if (block_id < 0 || static_cast<size_t>(block_id) >= allocator_->get_total_blocks()) {
            throw std::out_of_range("Block ID out of range: " + std::to_string(block_id));
        }
    }

    void release_internal() {
        if (allocator_ != nullptr) {
            for (int block_id : physical_blocks_) {
                allocator_->free_block(block_id);
            }
            physical_blocks_.clear();
            tokens_count_ = 0;
        }
    }

    PageAllocator* allocator_{nullptr};
    size_t block_size_;
    size_t tokens_count_{0};
    std::vector<int> physical_blocks_;
};

} 

#endif