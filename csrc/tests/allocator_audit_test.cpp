#include <algorithm>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#if defined(__unix__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../include/page_allocator.hpp"

using namespace llm_infra;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
        ++g_failures; \
    } \
} while (false)

#define CHECK_EQ(a, b) do { \
    ++g_checks; \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #a << " == " << #b \
                  << " (actual: " << _a << " vs " << _b << ")\n"; \
        ++g_failures; \
    } \
} while (false)

#define CHECK_NOTHROW(expr) do { \
    ++g_checks; \
    try { (void)(expr); } catch (const std::exception& e) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #expr \
                  << " threw " << typeid(e).name() << ": " << e.what() << "\n"; \
        ++g_failures; \
    } catch (...) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #expr << " threw unknown\n"; \
        ++g_failures; \
    } \
} while (false)

#define CHECK_THROWS_AS(expr, exception_type) do { \
    ++g_checks; \
    bool _caught = false; std::string _what = "<none>"; std::string _type = "<none>"; \
    try { (void)(expr); } catch (const exception_type& e) { _caught = true; } \
    catch (const std::exception& e) { _type = typeid(e).name(); _what = e.what(); } \
    catch (...) { _type = "<non-std>"; } \
    if (!_caught) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": expected " #exception_type \
                  << " from " #expr << " but got " << _type << " (" << _what << ")\n"; \
        ++g_failures; \
    } \
} while (false)

struct TestCase { const char* name; void (*fn)(); };

void run_test(const TestCase& tc) {
    int before = g_failures;
    try {
        tc.fn();
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << tc.name << ": uncaught " << typeid(e).name() << ": " << e.what() << "\n";
        ++g_failures;
    } catch (...) {
        std::cerr << "FAIL " << tc.name << ": uncaught unknown exception\n";
        ++g_failures;
    }
    if (g_failures == before) {
        std::cout << "PASS " << tc.name << "\n";
    } else {
        std::cout << "FAIL " << tc.name << " (" << (g_failures - before) << ")\n";
    }
}

// ---- H1: ID collision after swap_out ----
// Scenario from spec: allocate block A, swap_out(A), allocate until ID A issued again.
// Check get_host_ptr/get_cpu_block_id for live block, swap_out of new owner,
// release() of old SBT decrementing ref_count of wrong block, CPU pool accounting.
void test_h1_swap_out_id_reuse() {
    // 2 GPU blocks, 1 CPU block: force reuse of swapped-out GPU id.
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block();
    CHECK_EQ(a, 0);
    int cpu = alloc.swap_out(a); // frees GPU id 0, keeps mapping gpu_to_cpu_map_[0]=cpu
    CHECK_EQ(cpu, 0);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2)); // id 0 back in free queue
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(0));

    // Allocate until we get id 0 again (it was pushed to queue front? queue pop/front)
    int b = alloc.allocate_block(); // likely 1
    int c = alloc.allocate_block(); // likely 0 reused
    // find which one is reused id 0
    int reused = -1, other = -1;
    if (b == 0) { reused = b; other = c; } else if (c == 0) { reused = c; other = b; }
    // If FIFO queue gave 1 then 0, reused should be 0. Record what happened.
    std::cerr << "  H1 ids after reuse: b=" << b << " c=" << c << " reused_is_0=" << (reused==0) << "\n";
    CHECK_EQ(reused, 0); // contract assumption: swapped-out id becomes allocatable
    (void)other;

    // Live block 0 should NOT be considered swapped.
    // get_host_ptr takes cpu_block_id, so we can't query live block's host ptr directly
    // get_cpu_block_id should throw for live block
    int cpu_id = -999;
    bool cpu_id_threw = false;
    try { cpu_id = alloc.get_cpu_block_id(0); } catch (...) { cpu_id_threw = true; }
    std::cerr << "  H1 live block get_cpu_block_id threw=" << cpu_id_threw << " val=" << cpu_id << "\n";
    // Expected contract: live owner is not swapped -> get_cpu_block_id throws.
    CHECK(cpu_id_threw);

    // swap_out of new owner of id 0 should work (or fail sanely), not "already swapped".
    bool swap_out_new_threw_already = false;
    try {
        // need free cpu block; currently cpu pool exhausted by stale mapping
        alloc.swap_out(0);
    } catch (const std::exception& e) {
        std::string w = e.what();
        swap_out_new_threw_already = (w.find("already swapped") != std::string::npos);
        std::cerr << "  H1 swap_out(new owner of 0) threw: " << w << "\n";
    }
    CHECK(!swap_out_new_threw_already);

    // Clean up: try to release in a way that exposes ref_count corruption.
    // If we couldn't swap_out, free live blocks.
    // This is best-effort; main goal is detecting corruption.
}

void test_h1_controlled_stale_release() {
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block(); // 0, ref 1
    CHECK_EQ(a, 0);
    // simulate old table owning block 0 by keeping ref_count semantics:
    // swap_out requires ref_count==1 and sets it to 0, pushes to free queue
    alloc.swap_out(a);
    // now allocate others to reuse id 0
    int b = alloc.allocate_block(); // 1
    int c = alloc.allocate_block(); // 0 reused, ref 1
    CHECK_EQ(b, 1);
    CHECK_EQ(c, 0);
    // "old owner" calls free_block on id 0 (as SBT.release would do)
    // This should free the NEW owner's block 0 if corruption exists.
    size_t free_before = alloc.get_num_free_blocks();
    bool threw = false;
    try { alloc.free_block(0); } catch (...) { threw = true; }
    size_t free_after = alloc.get_num_free_blocks();
    std::cerr << "  H1 stale free_block(0): threw=" << threw
              << " free_before=" << free_before << " free_after=" << free_after
              << " cpu_free=" << alloc.get_num_free_cpu_blocks() << "\n";
    // If not corrupted, block 0 is legitimately owned by new owner (ref1) so free is valid
    // and decrements its ref to 0 -> moves to free list. That means stale mapping still exists
    // separately - check cpu accounting.
    // Key invariant: cpu_free + swapped == total_cpu
    size_t cpu_free = alloc.get_num_free_cpu_blocks();
    std::cerr << "  H1 cpu_free after stale free=" << cpu_free << " (expected 1 since stale map cleaned up)\n";
    // With fix, stale map entry for gpu 0 is cleaned up on reuse, cpu block returned to free pool
    CHECK_EQ(cpu_free, static_cast<size_t>(1));
}

void test_h1_cpu_pool_accounting() {
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block(); // 0
    alloc.swap_out(a);
    int b = alloc.allocate_block(); // 1
    int c = alloc.allocate_block(); // 0
    (void)b; (void)c;
    // At this point: gpu 0 is live (new owner), gpu 1 live, stale swap map cleaned up
    // cpu_free should be 1 (cpu block returned to free pool)
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(1));
    // Try to make system consistent: release new owners
    try { alloc.free_block(0); } catch (...) {}
    try { alloc.free_block(1); } catch (...) {}
    // Now gpu blocks may be free, but cpu accounting?
    std::cerr << "  H1 final: gpu_free=" << alloc.get_num_free_blocks()
              << " cpu_free=" << alloc.get_num_free_cpu_blocks() << "\n";
    // Desired end state: gpu_free==2, cpu_free==1
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2));
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(1));
}

void test_h2_free_block_releases_swapped_cpu() {
    // swap_out -> table.release() path: release calls free_block on ids in table.
    // After swap_out, block ref_count is 0 and id in gpu_to_cpu_map.
    // free_block on ref_count==0 + in map should release CPU block (per H2).
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block(); // 0
    SequenceBlockTable table(alloc);
    // table doesn't own a via append_block refcounting; simulate: table holds id
    // but easier: just call swap_out then free_block directly as release would.
    (void)table;
    alloc.swap_out(a);
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(0));
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2)); // gpu id freed by swap_out
    // free_block on swapped-out id (ref_count 0, in map)
    alloc.free_block(a);
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(1));
    // calling again should be error (double free of swapped mapping)
    CHECK_THROWS_AS(alloc.free_block(a), std::runtime_error);
}

void test_h3_get_device_ptr_offset() {
    PageAllocator alloc(4, 4, 4, 2);
    // get_device_ptr ignores block_id and returns base_host_ptr (or device view)
    // This is by design - it's deprecated. Per-block access uses get_host_ptr(cpu_block_id).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    uintptr_t p0 = alloc.get_device_ptr(0);
    uintptr_t p1 = alloc.get_device_ptr(1);
#pragma GCC diagnostic pop
    std::cerr << "  H3 get_device_ptr(0)=" << p0 << " get_device_ptr(1)=" << p1 << " diff=" << (p1-p0) << "\n";
    // Deprecated API returns same pointer for all blocks (base host ptr or device view)
    if (p0 != 0) {
        CHECK_EQ(p1, p0); // expected: same pointer returned for all blocks
    }
}

void test_h4_allocate_blocks_exception_safety() {
    PageAllocator alloc(3, 4, 4, 0);
    alloc.allocate_block(); // 1 free left? total 3, allocated 1 -> 2 free
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2));
    CHECK_THROWS_AS(alloc.allocate_blocks(5), std::runtime_error);
    // after exception, free count must be unchanged
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2));
}

void test_h4_append_token_oom_no_leak() {
    PageAllocator alloc(1, 2, 4, 0); // 1 block, 2 tokens per block
    SequenceBlockTable table(alloc);
    table.append_token(); // uses block 0, tokens=1
    table.append_token(); // fills block 0, tokens=2
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    // next append needs new block but none free
    size_t tokens_before = table.get_tokens_count();
    CHECK_THROWS_AS(table.append_token(), std::runtime_error);
    CHECK_EQ(table.get_tokens_count(), tokens_before);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    table.release();
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(1));
}

void test_h4_constructor_invalid_args() {
    CHECK_THROWS_AS(PageAllocator(1, 0, 4, 1), std::invalid_argument);
    CHECK_THROWS_AS(PageAllocator(1, 4, 0, 1), std::invalid_argument);
}

void test_h4_overflow_size() {
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
    std::cerr << "  H4 overflow skipped under ASan (huge alloc is fatal by design)\n";
    return;
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    std::cerr << "  H4 overflow skipped under ASan (huge alloc is fatal by design)\n";
    return;
#endif
    size_t huge = std::numeric_limits<size_t>::max() / 2;
#if defined(__unix__)
    pid_t pid = fork();
    if (pid == 0) {
        try {
            PageAllocator alloc(1, huge, 4, 1);
            _exit(10);
        } catch (...) {
            _exit(20);
        }
    } else if (pid > 0) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            std::cerr << "  H4 overflow child exit=" << code << "\n";
            CHECK_EQ(code, 20);
        } else {
            std::cerr << "  H4 overflow child signaled/crashed status=" << status << "\n";
            CHECK(false);
        }
    } else {
        std::cerr << "  H4 overflow fork failed\n";
        CHECK(false);
    }
#else
    (void)huge;
#endif
}

void test_h5_moved_from_state() {
    PageAllocator alloc(4, 4, 4, 0);
    SequenceBlockTable t(alloc);
    t.append_token();
    SequenceBlockTable moved(std::move(t));
    // Query methods should work on moved-from object
    CHECK(t.get_physical_blocks().empty());
    CHECK_EQ(t.get_tokens_count(), static_cast<size_t>(0));
    CHECK_NOTHROW(t.release());
    CHECK_NOTHROW(t.release()); // idempotent
    moved.release();
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(4));
}

void test_h5_append_on_moved_from() {
    PageAllocator alloc(4, 4, 4, 0);
#if defined(__unix__)
    // Run in child: mutating operations on moved-from should throw logic_error
    pid_t pid = fork();
    if (pid == 0) {
        SequenceBlockTable t(alloc);
        SequenceBlockTable moved(std::move(t));
        try { t.append_block(0); _exit(30); } catch (const std::logic_error&) { _exit(31); } catch (...) { _exit(32); }
    } else if (pid > 0) {
        int status = 0; waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            std::cerr << "  H5 append_block on moved-from child exit=" << code << " (30=no throw/crash,31=logic_error,32=other)\n";
            CHECK_EQ(code, 31); // desired: throw logic_error cleanly
        } else {
            std::cerr << "  H5 append_block on moved-from crashed/signaled status=" << status << "\n";
            CHECK(false); // crash = not safe
        }
    }
    pid_t pid2 = fork();
    if (pid2 == 0) {
        SequenceBlockTable t(alloc);
        SequenceBlockTable moved(std::move(t));
        try { t.append_token(); _exit(30); } catch (const std::logic_error&) { _exit(31); } catch (...) { _exit(32); }
    } else if (pid2 > 0) {
        int status = 0; waitpid(pid2, &status, 0);
        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            std::cerr << "  H5 append_token on moved-from child exit=" << code << "\n";
            CHECK_EQ(code, 31);
        } else {
            std::cerr << "  H5 append_token on moved-from crashed/signaled status=" << status << "\n";
            CHECK(false);
        }
    }
#else
    (void)alloc;
#endif
}

void test_h5_move_assign_releases_old() {
    PageAllocator alloc(4, 4, 4, 0);
    SequenceBlockTable target(alloc);
    target.append_token();
    target.append_token();
    target.append_token();
    target.append_token();
    target.append_token(); // owns 0,1
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2));
    SequenceBlockTable source(alloc);
    source.append_token(); // owns 2
    target = std::move(source);
    // old blocks 0,1 of target must be released; target now owns 2
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(3)); // 0,1,3 free; 2 owned
    target.release();
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(4));
}

void test_h5_self_move() {
    PageAllocator alloc(4, 4, 4, 0);
    SequenceBlockTable t(alloc);
    t.append_token();

    const size_t tokens_before = t.get_tokens_count();
    const std::vector<int> blocks_before = t.get_physical_blocks();
    const size_t free_before = alloc.get_num_free_blocks();

    // Self-move through a reference so -Wself-move stays enabled (-Werror).
    auto& r = t;
    t = std::move(r);

    // A self-move must be a no-op: same tokens, same physical blocks, same pool.
    CHECK_EQ(t.get_tokens_count(), tokens_before);
    CHECK(t.get_physical_blocks() == blocks_before);
    CHECK_EQ(alloc.get_num_free_blocks(), free_before);

    // release() must free exactly the blocks the table owned and return the pool
    // to full. Any exception here fails the test (no catch-all).
    t.release();
    CHECK(t.get_physical_blocks().empty());
    CHECK_EQ(t.get_tokens_count(), static_cast<size_t>(0));
    CHECK_EQ(alloc.get_num_free_blocks(), alloc.get_total_blocks());
}

void test_h6_replace_block_no_refcount_change() {
    PageAllocator alloc(4, 4, 4, 0);
    int a = alloc.allocate_block(); // ref 1
    int b = alloc.allocate_block(); // ref 1
    SequenceBlockTable table(alloc);
    table.append_block(a);
    // replace a with b in table: contract says no ref_count change, no free of old
    table.replace_block(a, b);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(2));
    // a is still allocated (ref 1) but not in table; must be manually freed or leaked intentionally
    // table release will free b
    table.release();
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(3)); // b freed, a still ref1
    alloc.free_block(a);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(4));
}

void test_h7_edge_case_semantics() {
    // Original test: share_block + free_block leaves block live (ref 1),
    // then swap_out throws due to no host pool - NOT because block freed.
    // Correct test: swap_out on shared block should fail with ref_count reason.
    PageAllocator alloc(1, 4, 4, 1);
    int block = alloc.allocate_block();
    alloc.share_block(block); // ref 2
    // swap_out requires ref_count == 1
    try {
        alloc.swap_out(block);
        std::cerr << "  H7 swap_out(shared block) unexpectedly succeeded\n";
        CHECK(false);
    } catch (const std::exception& e) {
        std::string w = e.what();
        std::cerr << "  H7 swap_out(shared block) error: " << w << "\n";
        // should mention ref_count, not host pool
        CHECK(w.find("ref_count") != std::string::npos);
    }
    alloc.free_block(block);
    alloc.free_block(block);
}

void test_h8_backend_copy_and_free() {
    // CpuMemoryBackend: copy is memcpy, overlapping is UB per C standard but memcpy used
    CpuMemoryBackend backend;
    void* p = backend.allocate_host_block(128);
    CHECK(p != nullptr);
    std::memset(p, 0xAB, 128);
    // overlapping memcpy - UB in general; but check if implementation uses memcpy or memmove
    // We just report, not assert crash. Use separate non-overlapping check.
    void* q = backend.allocate_device_block(128);
    CHECK(q != nullptr);
    backend.copy_block(q, p, 128, nullptr);
    CHECK(std::memcmp(p, q, 128) == 0);
    backend.free_device_block(q);
    backend.free_host_block(p);

    HipZeroCopyMemoryBackend hip;
    void* h = hip.allocate_host_block(128);
    CHECK(h != nullptr);
    std::memset(h, 0x5A, 128);
    // hip backend copy_block is no-op in zero-copy/unified path per implementation
    // but test as written: copy_block(host,host,128) - self copy, still "overlapping"
    CHECK_NOTHROW(hip.copy_block(h, h, 128, nullptr));
    CHECK_NOTHROW(hip.synchronize_stream(nullptr));
    // free via free_device_block - implementation routes to free_host_block
    CHECK_NOTHROW(hip.free_device_block(h));
}

// ---- Krok 3: boundaries, swaps, invariants ----
[[maybe_unused]] void check_invariants(const PageAllocator& alloc, size_t total_gpu, size_t total_cpu, const char* ctx) {
    size_t free_gpu = alloc.get_num_free_blocks();
    size_t free_cpu = alloc.get_num_free_cpu_blocks();
    // We can't see internal maps/refcounts without dump; basic sanity:
    if (free_gpu > total_gpu) {
        std::cerr << "FAIL invariant " << ctx << ": free_gpu " << free_gpu << " > total " << total_gpu << "\n";
        ++g_failures; ++g_checks;
    } else { ++g_checks; }
    if (free_cpu > total_cpu) {
        std::cerr << "FAIL invariant " << ctx << ": free_cpu " << free_cpu << " > total " << total_cpu << "\n";
        ++g_failures; ++g_checks;
    } else { ++g_checks; }
}

void test_boundaries_block_id() {
    PageAllocator alloc(2, 4, 4, 1);
    int b = alloc.allocate_block();
    CHECK_THROWS_AS(alloc.free_block(-1), std::out_of_range);
    CHECK_THROWS_AS(alloc.free_block(2), std::out_of_range);
    CHECK_THROWS_AS(alloc.share_block(-1), std::out_of_range);
    CHECK_THROWS_AS(alloc.share_block(INT_MAX), std::out_of_range);
    CHECK_THROWS_AS(alloc.swap_out(-1), std::out_of_range);
    CHECK_THROWS_AS(alloc.swap_out(INT_MIN), std::out_of_range);
    CHECK_THROWS_AS(alloc.swap_in(2), std::out_of_range);
    CHECK_THROWS_AS(alloc.get_cpu_block_id(-1), std::out_of_range);
    CHECK_THROWS_AS(alloc.release_swapped_block(2), std::out_of_range);
    CHECK_THROWS_AS(alloc.get_host_ptr(-1), std::out_of_range);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK_THROWS_AS(alloc.get_device_ptr(-1), std::out_of_range);
#pragma GCC diagnostic pop
    alloc.free_block(b);
}

void test_swap_in_no_free_gpu_state_unchanged() {
    // Need a swapped block and no free GPU blocks to swap into.
    PageAllocator alloc(2, 4, 4, 2);
    int a = alloc.allocate_block(); // 0
    int b = alloc.allocate_block(); // 1
    alloc.swap_out(a); // gpu 0 freed, cpu 0 used, map has a
    // now allocate gpu 0 again
    int c = alloc.allocate_block(); // gets 0
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    // swap_in(a's original id mapping) - but mapping key is a(==0). swap_in looks up gpu_block_id in map.
    // state: map has {0: cpu0}, gpu 0 is live owned by c, no free gpu blocks.
    size_t cpu_free_before = alloc.get_num_free_cpu_blocks();
    CHECK_THROWS_AS(alloc.swap_in(0), std::runtime_error);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), cpu_free_before);
    // cleanup: free c and b, then handle stale map
    alloc.free_block(c);
    alloc.free_block(b);
    // stale map still holds cpu block for key 0
    std::cerr << "  swap_in_no_gpu final cpu_free=" << alloc.get_num_free_cpu_blocks() << "\n";
}

void test_swap_out_no_cpu_state_unchanged() {
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block();
    int b = alloc.allocate_block();
    alloc.swap_out(a); // uses only cpu block
    // swap_in a to get it back? but need to understand state
    int a2 = alloc.swap_in(a); // frees cpu, gives new gpu id
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(1));
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    // now try swap_out b with no cpu free? cpu is free (1). allocate cpu by swapping a2? a2 is new id
    alloc.swap_out(a2);
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), static_cast<size_t>(0));
    // swap_out b should fail, state unchanged
    CHECK_THROWS_AS(alloc.swap_out(b), std::runtime_error);
    // b still allocated (ref 1)
    CHECK_NOTHROW(alloc.share_block(b)); // proves still live
    alloc.free_block(b); // undo share
    alloc.free_block(b);
    int back = alloc.swap_in(a2);
    alloc.free_block(back);
}

void test_full_cycle_exhaust_reuse() {
    PageAllocator alloc(4, 4, 4, 0);
    std::vector<int> ids;
    for (int i = 0; i < 4; ++i) ids.push_back(alloc.allocate_block());
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    CHECK_THROWS_AS(alloc.allocate_block(), std::runtime_error);
    for (int id : ids) alloc.free_block(id);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(4));
    ids.clear();
    for (int i = 0; i < 4; ++i) ids.push_back(alloc.allocate_block());
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(0));
    for (int id : ids) alloc.free_block(id);
    CHECK_EQ(alloc.get_num_free_blocks(), static_cast<size_t>(4));
}

void test_zero_cpu_blocks_swap_ops() {
    PageAllocator alloc(2, 4, 4, 0);
    int a = alloc.allocate_block();
    CHECK_THROWS_AS(alloc.swap_out(a), std::runtime_error);
    CHECK_THROWS_AS(alloc.swap_in(a), std::runtime_error);
    CHECK_THROWS_AS(alloc.release_swapped_block(a), std::runtime_error);
    CHECK_THROWS_AS(alloc.get_cpu_block_id(a), std::runtime_error);
    // get_host_ptr requires a CPU block ID; with no CPU pool there's no valid ID to query
    // CHECK_EQ(alloc.get_host_ptr(a), static_cast<uintptr_t>(0)); // Invalid: a is GPU block ID
    alloc.free_block(a);
}

void test_dump_state_no_throw() {
    PageAllocator alloc(2, 4, 4, 1);
    int a = alloc.allocate_block();
    std::ostringstream oss;
    CHECK_NOTHROW(alloc.dump_state(oss));
    CHECK(!oss.str().empty());
    alloc.swap_out(a);
    std::ostringstream oss2;
    CHECK_NOTHROW(alloc.dump_state(oss2));
    alloc.release_swapped_block(a);
}

void test_utilization_partial() {
    PageAllocator alloc(4, 4, 4, 0);
    CHECK_EQ(alloc.get_utilization_percentage(), 0.0);
    alloc.allocate_block();
    double u = alloc.get_utilization_percentage();
    CHECK(u > 24.9 && u < 25.1);
}

// ---- Step 1: kill M7 (get_host_ptr returns base instead of per-block) ----
// Contract: get_host_ptr(cpu_id) yields the address of that CPU block, distinct
// per block and inside the host pool, at block-size stride.
void test_m7_get_host_ptr_per_block() {
    const size_t total_cpu = 4;
    PageAllocator alloc(8, 4, 4, total_cpu);  // block bytes = 4 tokens * 4 bytes = 16
    const size_t block_bytes = alloc.get_host_pool_bytes() / total_cpu;
    CHECK_EQ(block_bytes, static_cast<size_t>(16));

    const int kSwapped = 3;
    std::vector<int> gpu_ids;
    for (int i = 0; i < kSwapped; ++i) gpu_ids.push_back(alloc.allocate_block());

    std::vector<int> cpu_ids;
    for (int g : gpu_ids) cpu_ids.push_back(alloc.swap_out(g));
    CHECK_EQ(cpu_ids.size(), static_cast<size_t>(kSwapped));

    // (a) CPU block handles must be pairwise distinct.
    for (size_t i = 0; i < cpu_ids.size(); ++i) {
        for (size_t j = i + 1; j < cpu_ids.size(); ++j) {
            CHECK(cpu_ids[i] != cpu_ids[j]);
        }
    }

    std::vector<uintptr_t> ptrs;
    for (int c : cpu_ids) {
        uintptr_t p = alloc.get_host_ptr(c);
        CHECK(p != 0);
        ptrs.push_back(p);
    }

    // (a') Addresses must be pairwise distinct.
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            CHECK(ptrs[i] != ptrs[j]);
        }
    }

    // (b) Every address lies in [base, base + host_pool_bytes).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    uintptr_t base = alloc.get_device_ptr(0);
#pragma GCC diagnostic pop
    if (base != 0) {
        for (uintptr_t p : ptrs) {
            CHECK(p >= base && p < base + alloc.get_host_pool_bytes());
        }
    }

    // (c) Pairwise distance is a multiple of the block size in bytes.
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            uintptr_t hi = ptrs[i] > ptrs[j] ? ptrs[i] : ptrs[j];
            uintptr_t lo = ptrs[i] > ptrs[j] ? ptrs[j] : ptrs[i];
            CHECK(((hi - lo) % block_bytes) == 0);
        }
    }

    // (d) Writing one block must not change another.
    for (size_t k = 0; k < ptrs.size(); ++k) {
        std::memset(reinterpret_cast<void*>(ptrs[k]),
                    0x10 + static_cast<int>(k), block_bytes);
    }
    std::memset(reinterpret_cast<void*>(ptrs[0]), 0xEE, block_bytes);
    for (size_t k = 1; k < ptrs.size(); ++k) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(ptrs[k]);
        for (size_t b = 0; b < block_bytes; ++b) {
            CHECK(bytes[b] == static_cast<uint8_t>(0x10 + k));
        }
    }
}

// ---- Step 2: kill M8 (swap_in does not reset cpu_block_states_ to FREE) ----
// Contract: after swap_out -> swap_in the CPU block is observable as free
// (get_host_ptr == 0), a release on that handle throws, and the block is
// immediately reusable by another swap_out.
void test_m8_swap_in_cpu_state_consistency() {
    for (size_t pool : {static_cast<size_t>(1), static_cast<size_t>(2)}) {
        for (int cycle = 0; cycle < 3; ++cycle) {
            PageAllocator alloc(4, 4, 4, pool);
            int g0 = alloc.allocate_block();
            int cpu = alloc.swap_out(g0);
            int back = alloc.swap_in(cpu);

            // Back in the free list...
            CHECK_EQ(alloc.get_num_free_cpu_blocks(), pool);
            // ...and its state is FREE (observable through get_host_ptr).
            CHECK_EQ(alloc.get_host_ptr(cpu), static_cast<uintptr_t>(0));

            // A release on the already-freed handle must throw.
            CHECK_THROWS_AS(alloc.release_swapped_block(cpu), std::runtime_error);

            // The CPU block must be reusable: swapping another live block out
            // with the same pool size must not throw.
            CHECK_NOTHROW(alloc.swap_out(back));
            CHECK_EQ(alloc.get_num_free_cpu_blocks(), pool - 1);
        }
    }
}

// ---- Step 3: kill M11 (free_block ignores ref_count in the swapped path) ----
// Contract: free_block on a *live* block that still carries a swap entry only
// decrements ref_count; it must not release the CPU block or leave the GPU block
// allocated. The live+swapped state is forced by pool exhaustion (swap_in pops a
// swapped-out block from the free queue while a different CPU block is swapped in).
void test_m11_free_block_live_vs_swapped() {
    PageAllocator alloc(4, 4, 4, 2);
    int a = alloc.allocate_block();  // 0
    int b = alloc.allocate_block();  // 1
    int cpu_a = alloc.swap_out(a);
    int cpu_b = alloc.swap_out(b);
    CHECK(cpu_a != cpu_b);

    // swap_in pops a never-allocated block from the free queue; consume it.
    int live_a = alloc.swap_in(cpu_b);
    int c = alloc.allocate_block();
    (void)c;
    alloc.swap_out(live_a);

    // The free-queue front is now a swapped-out block; swapping in a *different*
    // CPU block makes it live while its stale map entry survives.
    int live_swapped = alloc.swap_in(cpu_b);

    // Precondition: the block is live AND still has a swap entry.
    bool has_entry = true;
    try {
        (void)alloc.get_cpu_block_id(live_swapped);
    } catch (...) {
        has_entry = false;
    }
    CHECK(has_entry);

    const size_t free_gpu_before = alloc.get_num_free_blocks();
    const size_t free_cpu_before = alloc.get_num_free_cpu_blocks();

    alloc.free_block(live_swapped);

    // Only ref_count drops: CPU pool unchanged, GPU block returned to the pool.
    CHECK_EQ(alloc.get_num_free_cpu_blocks(), free_cpu_before);
    CHECK_EQ(alloc.get_num_free_blocks(), free_gpu_before + 1);
    // ref_count reached 0, so the block can no longer be shared.
    CHECK_THROWS_AS(alloc.share_block(live_swapped), std::runtime_error);
}

} // namespace

int main() {
    std::vector<TestCase> tests = {
        {"h1_swap_out_id_reuse", test_h1_swap_out_id_reuse},
        {"h1_controlled_stale_release", test_h1_controlled_stale_release},
        {"h1_cpu_pool_accounting", test_h1_cpu_pool_accounting},
        {"h2_free_block_releases_swapped_cpu", test_h2_free_block_releases_swapped_cpu},
        {"h3_get_device_ptr_offset", test_h3_get_device_ptr_offset},
        {"h4_allocate_blocks_exception_safety", test_h4_allocate_blocks_exception_safety},
        {"h4_append_token_oom_no_leak", test_h4_append_token_oom_no_leak},
        {"h4_constructor_invalid_args", test_h4_constructor_invalid_args},
        {"h4_overflow_size", test_h4_overflow_size},
        {"h5_moved_from_state", test_h5_moved_from_state},
        {"h5_append_on_moved_from", test_h5_append_on_moved_from},
        {"h5_move_assign_releases_old", test_h5_move_assign_releases_old},
        {"h5_self_move", test_h5_self_move},
        {"h6_replace_block_no_refcount_change", test_h6_replace_block_no_refcount_change},
        {"h7_edge_case_semantics", test_h7_edge_case_semantics},
        {"h8_backend_copy_and_free", test_h8_backend_copy_and_free},
        {"boundaries_block_id", test_boundaries_block_id},
        {"swap_in_no_free_gpu_state_unchanged", test_swap_in_no_free_gpu_state_unchanged},
        {"swap_out_no_cpu_state_unchanged", test_swap_out_no_cpu_state_unchanged},
        {"full_cycle_exhaust_reuse", test_full_cycle_exhaust_reuse},
        {"zero_cpu_blocks_swap_ops", test_zero_cpu_blocks_swap_ops},
        {"dump_state_no_throw", test_dump_state_no_throw},
        {"utilization_partial", test_utilization_partial},
        {"m7_get_host_ptr_per_block", test_m7_get_host_ptr_per_block},
        {"m8_swap_in_cpu_state_consistency", test_m8_swap_in_cpu_state_consistency},
        {"m11_free_block_live_vs_swapped", test_m11_free_block_live_vs_swapped},
    };

    for (const auto& t : tests) run_test(t);

    std::cout << "allocator_audit_test: " << (tests.size() - 0) << " tests, "
              << g_checks << " checks, " << g_failures << " failures\n";
    if (g_failures) {
        std::cout << "allocator_audit_test: FAIL\n";
        return 1;
    }
    std::cout << "allocator_audit_test: PASS\n";
    return 0;
}
