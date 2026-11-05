// Thread-safety smoke test for PageAllocator + SequenceBlockTable.
//
// Runs N worker threads hammering a shared allocator with concurrent
// allocate/free operations. Each thread tracks its own held blocks and
// drains them on exit, so the final state is deterministic regardless
// of scheduling.
//
// Invariants checked after join:
//   - free_blocks_ + sum(ref_count) == total_blocks_
//   - cpu_free_blocks_ + mapped cpu blocks == total_cpu_blocks_
//   - no data races (build with -fsanitize=thread)
//
// OOM exceptions from allocate_block are EXPECTED when the pool is under
// pressure; they are not counted as failures. Only leaked blocks at the end
// indicate a real bug.
//
// Usage:
//   g++ -std=c++17 -O1 -g -fsanitize=thread -pthread -I../include \
//       thread_safety_test.cpp -o thread_safety_test
//   ./thread_safety_test

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <cassert>
#include "../include/page_allocator.hpp"

using namespace llm_infra;

static constexpr int kTotalBlocks = 64;
static constexpr int kCpuBlocks = 16;
static constexpr int kBlockSize = 8;
static constexpr int kThreads = 4;

struct ThreadState {
    std::atomic<bool> stop{false};
    std::atomic<long> allocs{0};
    std::atomic<long> frees{0};
    std::atomic<long> ooms{0};
};

int main() {
    PageAllocator alloc(kTotalBlocks, kBlockSize, 4, kCpuBlocks);
    std::vector<ThreadState> states(kThreads);
    std::vector<std::thread> workers;

    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            std::mt19937 rng(1000 + t);
            auto& st = states[t];
            std::vector<int> held;
            while (!st.stop.load(std::memory_order_relaxed)) {
                int r = rng() % 100;
                if (r < 50) {
                    // allocate a block
                    try {
                        int bid = alloc.allocate_block();
                        held.push_back(bid);
                        st.allocs.fetch_add(1);
                    } catch (const std::exception&) {
                        st.ooms.fetch_add(1);
                    }
                } else if (!held.empty()) {
                    // free a random held block
                    int idx = rng() % held.size();
                    int bid = held[idx];
                    held[idx] = held.back();
                    held.pop_back();
                    try {
                        alloc.free_block(bid);
                        st.frees.fetch_add(1);
                    } catch (const std::exception&) {
                        // Re-add to held so we can drain it later.
                        held.push_back(bid);
                        st.ooms.fetch_add(1);
                    }
                }
            }
            // Drain: free everything held.
            for (int bid : held) {
                try { alloc.free_block(bid); } catch (const std::exception&) {}
            }
        });
    }

    // Let threads run for a bounded time.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (auto& st : states) st.stop.store(true);
    for (auto& w : workers) w.join();

    long total_allocs = 0, total_frees = 0, total_ooms = 0;
    for (const auto& st : states) {
        total_allocs += st.allocs.load();
        total_frees += st.frees.load();
        total_ooms += st.ooms.load();
    }
    std::cout << "thread_safety: allocs=" << total_allocs
              << " frees=" << total_frees
              << " ooms=" << total_ooms << "\n";

    // Final invariants.
    size_t free_blocks = alloc.get_num_free_blocks();
    size_t total = alloc.get_total_blocks();
    size_t free_cpu = alloc.get_num_free_cpu_blocks();
    size_t total_cpu = kCpuBlocks;

    std::cerr << "free_blocks=" << free_blocks << " total_blocks=" << total
              << " free_cpu=" << free_cpu << " total_cpu=" << total_cpu << "\n";
    alloc.dump_state(std::cerr);

    assert(free_blocks == total && "all GPU blocks must be free after drain");
    assert(free_cpu == total_cpu && "all CPU blocks must be free after drain");

    std::cout << "thread_safety: PASS\n";
    return 0;
}