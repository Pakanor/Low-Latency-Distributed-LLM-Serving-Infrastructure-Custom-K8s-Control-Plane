#ifndef PAGE_ALLOCATOR_HPP
#define PAGE_ALLOCATOR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstddef>

namespace llm_infra {

struct Block {
    int id;
    int ref_count{0};
};

class PageAllocator {
public:
    PageAllocator(size_t total_blocks, size_t block_size)
        : total_blocks_(total_blocks), block_size_(block_size) {
        for (size_t i = 0; i < total_blocks_; ++i) {
            free_blocks_.push(i);
            blocks_.push_back(Block{static_cast<int>(i), 0});
        }
    }

    // Alokuje jedną stronę pamięci
    int allocate_block() {
        if (free_blocks_.empty()) {
            throw std::runtime_error("Out of Memory: No free DRAM pages available in KV-Cache pool!");
        }
        int block_id = free_blocks_.front();
        free_blocks_.pop();
        blocks_[block_id].ref_count = 1;
        return block_id;
    }

    // Zwalnia stronę nazad do puli wolnych
    void free_block(int block_id) {
        if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
            throw std::out_of_range("Invalid block_id");
        }
        blocks_[block_id].ref_count--;
        if (blocks_[block_id].ref_count == 0) {
            free_blocks_.push(block_id);
        }
    }

    // Metryki dla Observability (przydadzą się w Slice 3 do Operatora w Go!)
    size_t get_num_free_blocks() const { return free_blocks_.size(); }
    size_t get_total_blocks() const { return total_blocks_; }
    double get_utilization_percentage() const {
        return 100.0 * (total_blocks_ - free_blocks_.size()) / total_blocks_;
    }

private:
    size_t total_blocks_;
    size_t block_size_; // Liczba tokenów w jednym bloku (np. 16)
    std::vector<Block> blocks_;
    std::queue<int> free_blocks_;
};

// Mapowanie konwersji ze stron logicznych sekwencji na strony fizyczne
class SequenceBlockTable {
public:
    SequenceBlockTable(size_t block_size) : block_size_(block_size) {}

    void append_token(PageAllocator& allocator) {
        if (tokens_count_ % block_size_ == 0) {
            // Wyczerpaliśmy bieżący blok, alokujemy nowy
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

} // namespace llm_infra

#endif // PAGE_ALLOCATOR_HPP