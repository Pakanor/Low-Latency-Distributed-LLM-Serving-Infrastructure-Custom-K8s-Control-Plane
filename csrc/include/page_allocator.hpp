#ifndef PAGE_ALLOCATOR_HPP
#define PAGE_ALLOCATOR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>

namespace llm_infra {

struct Block {
    int id;
    int ref_count{0};
};

class PageAllocator {
public:
    PageAllocator(size_t total_blocks, size_t block_size_bytes, size_t total_cpu_blocks = 32)
        : total_blocks_(total_blocks), block_size_(block_size_bytes), total_cpu_blocks_(total_cpu_blocks) {
        if (block_size_bytes == 0) {
            throw std::invalid_argument("block_size must be greater than 0");
        }

        
        size_t total_ram_bytes = total_cpu_blocks_ * block_size_;
        int ret = posix_memalign(&base_host_ptr_, 4096, total_ram_bytes);
        if (ret != 0 || !base_host_ptr_) {
            throw std::bad_alloc();
        }

        uint8_t* byte_ptr = static_cast<uint8_t*>(base_host_ptr_);

        for (size_t i = 0; i < total_blocks_; ++i) {
            free_blocks_.push(static_cast<int>(i));
            blocks_.push_back(Block{static_cast<int>(i), 0});
        }

        // Przypisanie fizycznych wskaźników RAM do bloków swapowych
        cpu_block_ptrs_.resize(total_cpu_blocks_);
        for (size_t i = 0; i < total_cpu_blocks_; ++i) {
            cpu_free_blocks_.push_back(static_cast<int>(i));
            cpu_block_ptrs_[i] = byte_ptr + (i * block_size_);
        }
    }

    ~PageAllocator() {
        if (base_host_ptr_) {
            free(base_host_ptr_);
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
            throw std::runtime_error("Double-free or free on unallocated block detected: ID " + std::to_string(block_id));
        }

        blocks_[block_id].ref_count--;
        if (blocks_[block_id].ref_count == 0) {
            free_blocks_.push(block_id);
        }
    }

    uintptr_t swap_out(int gpu_block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_block_id(gpu_block_id);

        if (blocks_[gpu_block_id].ref_count != 1) {
            throw std::runtime_error("Cannot swap out block with ref_count != 1");
        }

        if (cpu_free_blocks_.empty()) {
            throw std::runtime_error("OOM: Host RAM Block Pool exhausted during swap out");
        }

        int cpu_block_id = cpu_free_blocks_.back();
        cpu_free_blocks_.pop_back();

        gpu_to_cpu_map_[gpu_block_id] = cpu_block_id;

        blocks_[gpu_block_id].ref_count = 0;
        free_blocks_.push(gpu_block_id);

        void* host_ptr = cpu_block_ptrs_[cpu_block_id];
        return reinterpret_cast<uintptr_t>(host_ptr);
    }

    uintptr_t swap_in(int gpu_block_id) {
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
        void* host_ptr = cpu_block_ptrs_[cpu_block_id];

        cpu_free_blocks_.push_back(cpu_block_id);
        gpu_to_cpu_map_.erase(it);

        return reinterpret_cast<uintptr_t>(host_ptr);
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

    size_t get_block_size() const { return block_size_; }

    size_t get_num_free_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }

    size_t get_num_free_cpu_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cpu_free_blocks_.size();
    }

    size_t get_total_blocks() const { return total_blocks_; }

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
    size_t block_size_;
    size_t total_cpu_blocks_;
    
    void* base_host_ptr_{nullptr};
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
        physical_blocks_.push_back(block_id);
    }

    void append_token(PageAllocator& allocator) {
        if (tokens_count_ % block_size_ == 0) {
            int new_block = allocator.allocate_block();
            physical_blocks_.push_back(new_block);
        }
        tokens_count_++;
    }

    void release(PageAllocator& allocator) {
        for (int block_id : physical_blocks_) {
            allocator.free_block(block_id);
        }
        physical_blocks_.clear();
        tokens_count_ = 0;
    }

    const std::vector<int>& get_physical_blocks() const { return physical_blocks_; }
    size_t get_tokens_count() const { return tokens_count_; }

private:
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