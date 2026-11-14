#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/page_allocator.hpp"

using namespace llm_infra;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "        \
                      << #cond << "\n";                                        \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

#define CHECK_THROWS_AS(expr, exception_type)                                  \
    do {                                                                       \
        bool caught = false;                                                   \
        try {                                                                  \
            (void)(expr);                                                      \
        } catch (const exception_type&) {                                      \
            caught = true;                                                     \
        } catch (...) {                                                        \
        }                                                                      \
        if (!caught) {                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__                 \
                      << ": expected " #exception_type " from " #expr "\n";    \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

struct BackendStats {
    int host_allocations{0};
    int host_frees{0};
    int device_allocations{0};
    int device_frees{0};
    int copies{0};
    int synchronizations{0};
    size_t last_host_bytes{0};
};

class CountingBackend : public MemoryBackend {
public:
    explicit CountingBackend(BackendStats* stats, bool zero_copy = true)
        : stats_(stats), zero_copy_(zero_copy) {}

    void* allocate_host_block(size_t block_size_bytes) override {
        ++stats_->host_allocations;
        stats_->last_host_bytes = block_size_bytes;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, kAlignment, block_size_bytes) != 0) {
            return nullptr;
        }
        return ptr;
    }

    void free_host_block(void* ptr) override {
        if (ptr != nullptr) {
            ++stats_->host_frees;
        }
        std::free(ptr);
    }

    void* allocate_device_block(size_t block_size_bytes) override {
        ++stats_->device_allocations;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, kAlignment, block_size_bytes) != 0) {
            return nullptr;
        }
        return ptr;
    }

    void free_device_block(void* ptr) override {
        if (ptr != nullptr) {
            ++stats_->device_frees;
        }
        std::free(ptr);
    }

    void copy_block(void* dst, const void* src, size_t block_size_bytes,
                    void* /*stream*/) override {
        ++stats_->copies;
        std::memcpy(dst, src, block_size_bytes);
    }

    void synchronize_stream(void* /*stream*/) override {
        ++stats_->synchronizations;
    }

    bool is_zero_copy() const override { return zero_copy_; }

    const char* name() const override { return "CountingBackend"; }

private:
    static constexpr size_t kAlignment = 4096;
    BackendStats* stats_;
    bool zero_copy_;
};

class FailingBackend : public MemoryBackend {
public:
    void* allocate_host_block(size_t) override { return nullptr; }
    void free_host_block(void*) override {}
    void* allocate_device_block(size_t) override { return nullptr; }
    void free_device_block(void*) override {}
    void copy_block(void*, const void*, size_t, void*) override {}
    void synchronize_stream(void*) override {}
    bool is_zero_copy() const override { return false; }
    const char* name() const override { return "FailingBackend"; };
};

void test_cpu_backend() {
    CpuMemoryBackend backend;
    CHECK(backend.is_zero_copy());
    CHECK(std::string(backend.name()) == "CpuMemoryBackend");

    void* host = backend.allocate_host_block(128);
    void* device = backend.allocate_device_block(128);
    CHECK(host != nullptr);
    CHECK(device != nullptr);
    CHECK(host != device);

    std::memset(host, 0xAB, 128);
    std::memset(device, 0x00, 128);
    backend.copy_block(device, host, 128, nullptr);
    CHECK(std::memcmp(host, device, 128) == 0);
    backend.synchronize_stream(nullptr);

    backend.free_host_block(host);
    backend.free_device_block(device);
}

void test_hip_backend_fallback_is_safe() {
    HipZeroCopyMemoryBackend backend;
    void* host = backend.allocate_host_block(128);
    CHECK(host != nullptr);

#if !LLM_INFRA_HIP_ENABLED
    CHECK(!backend.is_zero_copy());
    CHECK(backend.get_device_view(host) == 0);
#endif

    backend.copy_block(host, host, 128, nullptr);
    backend.synchronize_stream(nullptr);
    backend.free_device_block(host);
}

void test_page_allocator_geometry_and_default_backend() {
    PageAllocator alloc(4, 4, 4, 2);
    CHECK(alloc.get_total_blocks() == 4);
    CHECK(alloc.get_block_size() == 4);
    CHECK(alloc.get_bytes_per_token() == 4);
    CHECK(alloc.get_host_pool_bytes() == 2 * 4 * 4);
    CHECK(alloc.get_num_free_blocks() == 4);
    CHECK(alloc.get_num_free_cpu_blocks() == 2);
    CHECK(alloc.get_utilization_percentage() == 0.0);
    CHECK(alloc.is_zero_copy());
    // get_device_ptr is deprecated but still functional for host pool view
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK(alloc.get_device_ptr(0) != 0);
    CHECK_THROWS_AS(alloc.get_device_ptr(4), std::out_of_range);
#pragma GCC diagnostic pop
}

void test_page_allocator_uses_injected_backend() {
    BackendStats stats;
    {
        PageAllocator alloc(3, 4, 4, 2,
                            std::make_unique<CountingBackend>(&stats));
        CHECK(stats.host_allocations == 1);
        CHECK(stats.last_host_bytes == 2 * 4 * 4);
        CHECK(stats.host_frees == 0);
        CHECK(alloc.is_zero_copy());
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        CHECK(alloc.get_device_ptr(0) != 0);
#pragma GCC diagnostic pop
    }
    CHECK(stats.host_frees == 1);
}

void test_page_allocator_rejects_failed_host_allocation() {
    CHECK_THROWS_AS(
        (PageAllocator(1, 4, 4, 1, std::make_unique<FailingBackend>())),
        std::bad_alloc);
}

void test_page_allocator_non_zero_copy_backend_disables_device_ptr() {
    BackendStats stats;
    PageAllocator alloc(2, 4, 4, 1,
                        std::make_unique<CountingBackend>(&stats, false));
    CHECK(!alloc.is_zero_copy());
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK(alloc.get_device_ptr(0) == 0);
#pragma GCC diagnostic pop
}

void test_allocation_refcounts_and_errors() {
    PageAllocator alloc(3, 4, 4, 1);
    int first = alloc.allocate_block();
    int second = alloc.allocate_block();
    CHECK(first == 0);
    CHECK(second == 1);
    CHECK(alloc.get_num_free_blocks() == 1);

    alloc.share_block(first);
    CHECK(alloc.get_num_free_blocks() == 1);
    alloc.free_block(first);
    CHECK(alloc.get_num_free_blocks() == 1);
    alloc.free_block(first);
    CHECK(alloc.get_num_free_blocks() == 2);
    CHECK_THROWS_AS(alloc.free_block(first), std::runtime_error);
    CHECK_THROWS_AS(alloc.share_block(first), std::runtime_error);

    auto batch = alloc.allocate_blocks(2);
    CHECK(batch.size() == 2);
    CHECK(alloc.get_num_free_blocks() == 0);
    CHECK_THROWS_AS(alloc.allocate_block(), std::runtime_error);
    CHECK_THROWS_AS(alloc.allocate_blocks(1), std::runtime_error);

    for (int block_id : batch) {
        alloc.free_block(block_id);
    }
    alloc.free_block(second);
    CHECK(alloc.get_num_free_blocks() == 3);
}

void test_swap_roundtrip_and_host_pointers() {
    PageAllocator alloc(2, 4, 4, 1);
    int gpu_block = alloc.allocate_block();
    // get_host_ptr requires cpu_block_id, not gpu_block_id
    // Before swap_out, there's no CPU mapping for this GPU block

    int cpu_block = alloc.swap_out(gpu_block);
    CHECK(cpu_block == 0);
    CHECK(alloc.get_cpu_block_id(gpu_block) == 0);
    CHECK(alloc.get_host_ptr(cpu_block) != 0);
    CHECK(alloc.get_num_free_blocks() == 2);
    CHECK(alloc.get_num_free_cpu_blocks() == 0);
    CHECK_THROWS_AS(alloc.swap_out(gpu_block), std::runtime_error);

    int swapped_in = alloc.swap_in(cpu_block);
    CHECK(swapped_in == 1);
    CHECK(alloc.get_num_free_blocks() == 1);
    CHECK(alloc.get_num_free_cpu_blocks() == 1);
    CHECK_THROWS_AS(alloc.get_cpu_block_id(gpu_block), std::runtime_error);

    alloc.free_block(swapped_in);
    CHECK(alloc.get_num_free_blocks() == 2);
}

void test_release_swapped_block_returns_cpu_block() {
    PageAllocator alloc(1, 4, 4, 1);
    int gpu_block = alloc.allocate_block();
    int cpu_block = alloc.swap_out(gpu_block);
    CHECK(alloc.get_num_free_cpu_blocks() == 0);

    alloc.release_swapped_block(cpu_block);
    CHECK(alloc.get_num_free_cpu_blocks() == 1);
    CHECK_THROWS_AS(alloc.get_cpu_block_id(gpu_block), std::runtime_error);
}

void test_sequence_block_table_append_and_release_without_argument() {
    PageAllocator alloc(8, 4, 4, 2);
    {
        SequenceBlockTable table(alloc);
        for (int i = 0; i < 5; ++i) {
            table.append_token();
        }
        CHECK(table.get_physical_blocks() == std::vector<int>({0, 1}));
        CHECK(table.get_tokens_count() == 5);
        CHECK(alloc.get_num_free_blocks() == 6);

        table.release();
        CHECK(table.get_physical_blocks().empty());
        CHECK(table.get_tokens_count() == 0);
        CHECK(alloc.get_num_free_blocks() == 8);

        table.release();
        CHECK(alloc.get_num_free_blocks() == 8);
    }
    CHECK(alloc.get_num_free_blocks() == 8);
}

void test_sequence_block_table_append_token_allocates_blocks() {
    PageAllocator alloc(4, 4, 4, 0);
    SequenceBlockTable table(alloc);
    for (int i = 0; i < 4; ++i) {
        table.append_token();
    }
    CHECK(table.get_tokens_count() == 4);
    CHECK(table.get_physical_blocks() == std::vector<int>({0}));

    table.append_token();
    CHECK(table.get_tokens_count() == 5);
    CHECK(table.get_physical_blocks() == std::vector<int>({0, 1}));
    CHECK(alloc.get_num_free_blocks() == 2);

    table.release();
    CHECK(alloc.get_num_free_blocks() == 4);
}

void test_sequence_block_table_validation() {
    PageAllocator alloc(8, 4, 4, 0);
    SequenceBlockTable table(alloc);
    CHECK_THROWS_AS(table.append_block(-1), std::out_of_range);
    CHECK_THROWS_AS(table.append_block(8), std::out_of_range);

    int old_block = alloc.allocate_block();
    int new_block = alloc.allocate_block();
    table.append_block(old_block);
    CHECK_THROWS_AS(table.replace_block(old_block, 8), std::out_of_range);
    CHECK_THROWS_AS(table.replace_block(99, new_block), std::runtime_error);
    table.replace_block(old_block, new_block);
    CHECK(table.get_physical_blocks() == std::vector<int>({new_block}));

    table.release();
    alloc.free_block(old_block);
    CHECK(alloc.get_num_free_blocks() == 8);
}

void test_sequence_block_table_move_ctor_and_move_assignment() {
    PageAllocator alloc(4, 4, 4, 0);

    SequenceBlockTable first(alloc);
    for (int i = 0; i < 5; ++i) {
        first.append_token();
    }
    SequenceBlockTable moved(std::move(first));
    CHECK(moved.get_physical_blocks() == std::vector<int>({0, 1}));
    CHECK(alloc.get_num_free_blocks() == 2);
    moved.release();
    CHECK(alloc.get_num_free_blocks() == 4);

    SequenceBlockTable source(alloc);
    for (int i = 0; i < 5; ++i) {
        source.append_token();
    }

    SequenceBlockTable target(alloc);
    target.append_token();

    target = std::move(source);
    CHECK(target.get_physical_blocks() == std::vector<int>({2, 3}));
    CHECK(alloc.get_num_free_blocks() == 2);

    target.release();
    CHECK(alloc.get_num_free_blocks() == 4);
}

void test_allocator_edge_cases() {
    CHECK_THROWS_AS((PageAllocator(1, 0, 4, 1)), std::invalid_argument);
    CHECK_THROWS_AS((PageAllocator(1, 4, 0, 1)), std::invalid_argument);

    PageAllocator empty(0, 4, 4, 0);
    CHECK(empty.get_num_free_blocks() == 0);
    CHECK(empty.get_utilization_percentage() == 0.0);
    CHECK_THROWS_AS(empty.allocate_block(), std::runtime_error);
    CHECK_THROWS_AS(empty.get_host_ptr(0), std::out_of_range);

    PageAllocator no_host(2, 4, 4, 0);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK(no_host.get_device_ptr(0) == 0);
#pragma GCC diagnostic pop

    PageAllocator one(1, 4, 4, 0);
    int block = one.allocate_block();
    CHECK_THROWS_AS(one.allocate_block(), std::runtime_error);
    one.share_block(block);
    CHECK_THROWS_AS(one.swap_out(block), std::runtime_error);
    one.free_block(block);
    CHECK_THROWS_AS(one.swap_out(block), std::runtime_error);
    CHECK_THROWS_AS(one.swap_in(0), std::runtime_error);

    PageAllocator swap_no_host(1, 4, 4, 0);
    int swapped = swap_no_host.allocate_block();
    CHECK_THROWS_AS(swap_no_host.swap_out(swapped), std::runtime_error);
}

}  // namespace

int main() {
    test_cpu_backend();
    test_hip_backend_fallback_is_safe();
    test_page_allocator_geometry_and_default_backend();
    test_page_allocator_uses_injected_backend();
    test_page_allocator_rejects_failed_host_allocation();
    test_page_allocator_non_zero_copy_backend_disables_device_ptr();
    test_allocation_refcounts_and_errors();
    test_swap_roundtrip_and_host_pointers();
    test_release_swapped_block_returns_cpu_block();
    test_sequence_block_table_append_and_release_without_argument();
    test_sequence_block_table_append_token_allocates_blocks();
    test_sequence_block_table_validation();
    test_sequence_block_table_move_ctor_and_move_assignment();
    test_allocator_edge_cases();

    if (failures != 0) {
        std::cerr << "allocator_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "allocator_test: PASS\n";
    return 0;
}