#ifndef PAGE_ALLOCATOR_HPP
#define PAGE_ALLOCATOR_HPP

#include <vector>
#include <queue>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <mutex>
#include <string>

namespace llm_infra {

struct Block {
    int id;
    int ref_count{0};
};

class PageAllocator {
public:
    PageAllocator(size_t total_blocks, size_t block_size)
        : total_blocks_(total_blocks), block_size_(block_size) {
        if (block_size == 0) {
            throw std::invalid_argument("block_size must be greater than 0");
        }
        for (size_t i = 0; i < total_blocks_; ++i) {
            free_blocks_.push(static_cast<int>(i));
            blocks_.push_back(Block{static_cast<int>(i), 0});
        }
    }

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            throw std::runtime_error("Out of Memory: No free DRAM pages available in KV-Cache pool!");
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

    size_t get_block_size() const { return block_size_; }

    size_t get_num_free_blocks() {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }

    size_t get_total_blocks() const { return total_blocks_; }

    double get_utilization_percentage() {
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
    std::vector<Block> blocks_;
    std::queue<int> free_blocks_;
    std::mutex mutex_;
};

class SequenceBlockTable {
public:
    explicit SequenceBlockTable(PageAllocator& allocator) 
        : block_size_(allocator.get_block_size()) {}

    SequenceBlockTable(const SequenceBlockTable&) = delete;
    SequenceBlockTable& operator=(const SequenceBlockTable&) = delete;

    SequenceBlockTable(SequenceBlockTable&&) noexcept = default;
    SequenceBlockTable& operator=(SequenceBlockTable&&) noexcept = default;

    ~SequenceBlockTable() = default;

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
    size_t block_size_;
    size_t tokens_count_{0};
    std::vector<int> physical_blocks_;
};

} 

#endif 