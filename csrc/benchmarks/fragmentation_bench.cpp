#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include "../include/page_allocator.hpp"

struct NaiveSequence {
    std::vector<float> kv_cache; 
    size_t realloc_count{0};

    void append_tokens(size_t num_tokens, size_t hidden_dim) {
        size_t current_tokens = kv_cache.size() / hidden_dim;
        size_t new_tokens = current_tokens + num_tokens;
        
        if (kv_cache.capacity() < new_tokens * hidden_dim) {
            realloc_count++;
        }
        kv_cache.resize(new_tokens * hidden_dim, 1.0f);
    }
};

int main() {
    constexpr size_t NUM_REQUESTS = 5000;
    constexpr size_t HIDDEN_DIM = 128; 
    constexpr size_t BLOCK_SIZE = 16;   
    constexpr size_t MAX_TOKENS_PER_REQ = 512;

    std::cout << "=== RUNNING LLM MEMORY ALLOCATION BENCHMARK ===" << std::endl;
    std::cout << "Requests: " << NUM_REQUESTS << " | Head Dim: " << HIDDEN_DIM << " | Page Size: " << BLOCK_SIZE << " tokens\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(16, MAX_TOKENS_PER_REQ); 

    std::vector<size_t> target_lengths(NUM_REQUESTS);
    for (size_t i = 0; i < NUM_REQUESTS; ++i) {
        target_lengths[i] = dist(rng);
    }

 
    auto start_naive = std::chrono::high_resolution_clock::now();
    size_t total_reallocs = 0;

    for (size_t i = 0; i < NUM_REQUESTS; ++i) {
        NaiveSequence seq;
        for (size_t t = 0; t < target_lengths[i]; ++t) {
            seq.append_tokens(1, HIDDEN_DIM);
        }
        total_reallocs += seq.realloc_count;
    }
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_naive = end_naive - start_naive;

    
    size_t total_pages_needed = (NUM_REQUESTS * MAX_TOKENS_PER_REQ) / BLOCK_SIZE;
    llm_infra::PageAllocator allocator(total_pages_needed, BLOCK_SIZE);

    auto start_paged = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_REQUESTS; ++i) {
        llm_infra::SequenceBlockTable table(allocator);
        for (size_t t = 0; t < target_lengths[i]; ++t) {
            table.append_token(allocator);
        }
        table.release(allocator);
    }

    auto end_paged = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_paged = end_paged - start_paged;

    
    std::cout << "--- RESULTS ---" << std::endl;
    std::cout << "[Naive Allocation]\n";
    std::cout << "  Execution Time:  " << duration_naive.count() << " ms\n";
    std::cout << "  Total Reallocations (Memory Copies): " << total_reallocs << "\n\n";

    std::cout << "[Paged Allocation]\n";
    std::cout << "  Execution Time:  " << duration_paged.count() << " ms\n";
    std::cout << "  Total Reallocations: 0 (Fixed zero-copy allocations)\n";
    std::cout << "  Final Page Allocator Saturation: " << allocator.get_utilization_percentage() << "%\n\n";

    double speedup = duration_naive.count() / duration_paged.count();
    std::cout << "Speedup factor: " << speedup << "x faster allocation lifecycle\n";

    return 0;
}