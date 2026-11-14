#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "../include/page_allocator.hpp"

using namespace llm_infra;

static double mean_vec(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// ---------- Contiguous (single big pool, first-fit, no coalescing) ----------
struct ContigAlloc { size_t start; size_t bytes; };

class ContiguousPool {
public:
    ContiguousPool(size_t total_bytes) : total_bytes_(total_bytes) {}

    size_t allocate(size_t bytes) {
        if (bytes == 0) return SIZE_MAX;
        size_t pos = 0;
        // first-fit over gaps between sorted allocations
        for (const auto& a : allocs_) {
            size_t gap = a.start - pos;
            if (gap >= bytes) {
                allocs_.insert(std::lower_bound(allocs_.begin(), allocs_.end(), pos,
                    [](const ContigAlloc& c, size_t s){ return c.start < s; }),
                    {pos, bytes});
                return pos;
            }
            pos = a.start + a.bytes;
        }
        size_t gap = total_bytes_ - pos;
        if (gap >= bytes) {
            allocs_.push_back({pos, bytes});
            return pos;
        }
        return SIZE_MAX;
    }

    void free(size_t start) {
        auto it = std::lower_bound(allocs_.begin(), allocs_.end(), start,
            [](const ContigAlloc& c, size_t s){ return c.start < s; });
        if (it != allocs_.end() && it->start == start) {
            allocs_.erase(it);
        }
    }

    size_t total_free() const {
        size_t used = 0;
        for (const auto& a : allocs_) used += a.bytes;
        return total_bytes_ - used;
    }

    size_t largest_free() const {
        size_t pos = 0, best = 0;
        for (const auto& a : allocs_) {
            best = std::max(best, a.start - pos);
            pos = a.start + a.bytes;
        }
        best = std::max(best, total_bytes_ - pos);
        return best;
    }

    size_t used_bytes() const {
        size_t used = 0;
        for (const auto& a : allocs_) used += a.bytes;
        return used;
    }

    size_t total_bytes() const { return total_bytes_; }

private:
    size_t total_bytes_;
    std::vector<ContigAlloc> allocs_; // sorted by start
};

struct SeqState {
    size_t tokens = 0;
    size_t max_tokens = 0;
    bool active = false;
    // paged
    std::vector<int> block_ids;
    // contiguous
    size_t contig_start = SIZE_MAX;
    size_t contig_bytes = 0;
};

int main() {
    constexpr size_t NUM_SEQUENCES = 128;
    constexpr size_t HIDDEN_DIM = 128;
    constexpr size_t BLOCK_SIZE = 16;
    constexpr size_t MAX_TOKENS_PER_REQ = 512;
    constexpr double BYTES_PER_TOKEN = static_cast<double>(HIDDEN_DIM) * sizeof(float);
    // tight pool: enough for ~70% of max simultaneous demand -> creates pressure
    constexpr size_t TOTAL_BLOCKS = 600;
    constexpr size_t TOTAL_BYTES = TOTAL_BLOCKS * BLOCK_SIZE * HIDDEN_DIM * sizeof(float);
    constexpr size_t STEPS = 200;
    constexpr double NEW_SEQ_PROB = 0.15;   // per step, per free slot
    constexpr double END_SEQ_PROB = 0.05;   // per active seq per step

    std::mt19937 rng(42);

    std::cout << "=== FRAGMENTATION BENCHMARK (dynamic load) ===\n";
    std::cout << "Sequences: " << NUM_SEQUENCES
              << " | Hidden Dim: " << HIDDEN_DIM
              << " | Block Size: " << BLOCK_SIZE << " tokens"
              << " | Max Tokens/Req: " << MAX_TOKENS_PER_REQ
              << " | Total Blocks: " << TOTAL_BLOCKS
              << " | Steps: " << STEPS << "\n\n";

    std::ofstream csv("fragmentation_results.csv");
    csv << "iteration,length_mode,length_mean,length_stddev,"
        << "paged_peak_blocks,paged_wasted_tokens,paged_wasted_bytes,paged_peak_bytes,paged_rejected,paged_time_ms,paged_no_leak,"
        << "contig_peak_bytes,contig_wasted_external,contig_wasted_bytes,contig_peak_bytes,contig_rejected,contig_time_ms,contig_no_leak\n";

    struct LengthMode { const char* name; double min_frac; double max_frac; };
    std::vector<LengthMode> modes = {
        {"constant_max", 1.0, 1.0},
        {"low_variance", 0.8, 1.0},
        {"medium_variance", 0.4, 1.0},
        {"high_variance", 0.05, 1.0},
    };

    for (const auto& mode : modes) {
        std::vector<double> pw_list, cw_list, pp_list, cp_list, pt_list, ct_list;
        std::vector<double> lm_list, ls_list;
        std::vector<double> pr_list, cr_list;
        bool all_paged_ok = true, all_contig_ok = true;

        for (size_t rep = 0; rep < 25; ++rep) {
            PageAllocator paged(TOTAL_BLOCKS, BLOCK_SIZE, static_cast<size_t>(BYTES_PER_TOKEN));
            ContiguousPool contig(TOTAL_BYTES);

            std::vector<SeqState> seqs(NUM_SEQUENCES);

            size_t paged_peak_blocks = 0;
            size_t contig_peak_bytes = 0;
            size_t paged_rejected = 0;
            size_t contig_rejected = 0;
            size_t paged_alloc_time_ns = 0;
            size_t contig_alloc_time_ns = 0;

            double len_sum = 0.0, len_sq_sum = 0.0;
            size_t len_count = 0;

            auto add_seq = [&](size_t len) {
                // find free slot
                size_t slot = SIZE_MAX;
                for (size_t i = 0; i < NUM_SEQUENCES; ++i) {
                    if (!seqs[i].active) { slot = i; break; }
                }
                if (slot == SIZE_MAX) return;

                len_sum += static_cast<double>(len);
                len_sq_sum += static_cast<double>(len) * static_cast<double>(len);
                ++len_count;

                auto& s = seqs[slot];
                s.active = true;
                s.tokens = len;
                s.max_tokens = len;

                // ---------- PAGED ----------
                size_t need_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
                auto t0 = std::chrono::high_resolution_clock::now();
                bool paged_ok = true;
                for (size_t b = 0; b < need_blocks; ++b) {
                    try {
                        int bid = paged.allocate_block();
                        s.block_ids.push_back(bid);
                    } catch (...) {
                        paged_ok = false;
                        break;
                    }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                paged_alloc_time_ns += static_cast<size_t>(std::chrono::duration<double, std::nano>(t1 - t0).count());

                if (!paged_ok) {
                    // rollback paged
                    for (int bid : s.block_ids) paged.free_block(bid);
                    s.block_ids.clear();
                    s.active = false;
                    ++paged_rejected;
                    return;
                }

                // ---------- CONTIGUOUS ----------
                size_t need_bytes = static_cast<size_t>(std::ceil(static_cast<double>(len) * BYTES_PER_TOKEN));
                auto t2 = std::chrono::high_resolution_clock::now();
                size_t cstart = contig.allocate(need_bytes);
                auto t3 = std::chrono::high_resolution_clock::now();
                contig_alloc_time_ns += static_cast<size_t>(std::chrono::duration<double, std::nano>(t3 - t2).count());

                if (cstart == SIZE_MAX) {
                    // rollback paged
                    for (int bid : s.block_ids) paged.free_block(bid);
                    s.block_ids.clear();
                    s.active = false;
                    ++contig_rejected;
                    return;
                }
                s.contig_start = cstart;
                s.contig_bytes = need_bytes;

                size_t used_blocks = 0;
                for (const auto& sq : seqs) if (sq.active) used_blocks += sq.block_ids.size();
                if (used_blocks > paged_peak_blocks) paged_peak_blocks = used_blocks;
                size_t cused = contig.used_bytes();
                if (cused > contig_peak_bytes) contig_peak_bytes = cused;
            };

            auto decode_step = [&]() {
                for (size_t i = 0; i < NUM_SEQUENCES; ++i) {
                    auto& s = seqs[i];
                    if (!s.active) continue;
                    ++s.tokens;

                    // paged: new block when crossing block boundary
                    if (s.tokens % BLOCK_SIZE == 1 && s.tokens > 1) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        try {
                            int bid = paged.allocate_block();
                            s.block_ids.push_back(bid);
                        } catch (...) {
                            ++paged_rejected;
                            s.active = false;
                            contig.free(s.contig_start);
                            s.contig_start = SIZE_MAX;
                            s.contig_bytes = 0;
                            for (int bid : s.block_ids) paged.free_block(bid);
                            s.block_ids.clear();
                            continue;
                        }
                        auto t1 = std::chrono::high_resolution_clock::now();
                        paged_alloc_time_ns += static_cast<size_t>(std::chrono::duration<double, std::nano>(t1 - t0).count());
                    }

                    // contiguous: need +1 token worth of bytes right after current block
                    (void)static_cast<size_t>(std::ceil(1.0 * BYTES_PER_TOKEN));
                    size_t needed_total = static_cast<size_t>(std::ceil(static_cast<double>(s.tokens) * BYTES_PER_TOKEN));
                    if (needed_total > s.contig_bytes) {
                        // try to extend in place: only possible if contiguous space right after
                        // Since we don't track per-byte gaps precisely, treat as: need new allocation
                        // (simulates external fragmentation: cannot extend, must reject)
                        auto t2 = std::chrono::high_resolution_clock::now();
                        size_t cstart = contig.allocate(needed_total);
                        auto t3 = std::chrono::high_resolution_clock::now();
                        contig_alloc_time_ns += static_cast<size_t>(std::chrono::duration<double, std::nano>(t3 - t2).count());
                        if (cstart == SIZE_MAX) {
                            ++contig_rejected;
                            s.active = false;
                            contig.free(s.contig_start);
                            for (int bid : s.block_ids) paged.free_block(bid);
                            s.block_ids.clear();
                            s.contig_start = SIZE_MAX;
                            s.contig_bytes = 0;
                            continue;
                        }
                        contig.free(s.contig_start);
                        s.contig_start = cstart;
                        s.contig_bytes = needed_total;
                    }
                }
            };

            auto end_random_seqs = [&]() {
                for (size_t i = 0; i < NUM_SEQUENCES; ++i) {
                    auto& s = seqs[i];
                    if (!s.active) continue;
                    if (std::bernoulli_distribution(END_SEQ_PROB)(rng)) {
                        s.active = false;
                        contig.free(s.contig_start);
                        s.contig_start = SIZE_MAX;
                        s.contig_bytes = 0;
                        for (int bid : s.block_ids) paged.free_block(bid);
                        s.block_ids.clear();
                        s.tokens = 0;
                        s.max_tokens = 0;
                    }
                }
            };

            for (size_t step = 0; step < STEPS; ++step) {
                // new sequences arrive
                size_t free_slots = 0;
                for (const auto& s : seqs) if (!s.active) ++free_slots;
                size_t arrivals = 0;
                for (size_t i = 0; i < free_slots; ++i) {
                    if (std::bernoulli_distribution(NEW_SEQ_PROB)(rng)) ++arrivals;
                }
                std::uniform_real_distribution<double> dist(mode.min_frac, mode.max_frac);
                for (size_t a = 0; a < arrivals; ++a) {
                    size_t l = static_cast<size_t>(std::ceil(dist(rng) * static_cast<double>(MAX_TOKENS_PER_REQ)));
                    if (l < 1) l = 1;
                    if (l > MAX_TOKENS_PER_REQ) l = MAX_TOKENS_PER_REQ;
                    add_seq(l);
                }

                decode_step();
                end_random_seqs();

                size_t used_blocks = 0;
                for (const auto& s : seqs) if (s.active) used_blocks += s.block_ids.size();
                if (used_blocks > paged_peak_blocks) paged_peak_blocks = used_blocks;
                size_t cused = contig.used_bytes();
                if (cused > contig_peak_bytes) contig_peak_bytes = cused;
            }

            // wasted paged = internal padding
            double paged_wasted_tokens = 0.0;
            size_t paged_live_blocks = 0;
            for (const auto& s : seqs) {
                if (!s.active) continue;
                paged_live_blocks += s.block_ids.size();
                paged_wasted_tokens += static_cast<double>(s.block_ids.size() * BLOCK_SIZE) - static_cast<double>(s.tokens);
            }
            // sanity: free all remaining
            for (auto& s : seqs) {
                if (!s.active) continue;
                for (int bid : s.block_ids) paged.free_block(bid);
                contig.free(s.contig_start);
                s.block_ids.clear();
                s.active = false;
            }
            bool paged_no_leak = (paged.get_num_free_blocks() == paged.get_total_blocks());
            bool contig_no_leak = (contig.total_free() == contig.total_bytes());
            if (!paged_no_leak) all_paged_ok = false;
            if (!contig_no_leak) all_contig_ok = false;

            // contiguous external waste = free memory not usable as largest single chunk
            double contig_wasted_external = static_cast<double>(contig.total_free() - contig.largest_free());
            double contig_wasted_bytes = contig_wasted_external;

            double paged_peak_bytes = static_cast<double>(paged_peak_blocks) * static_cast<double>(BLOCK_SIZE) * BYTES_PER_TOKEN;
            double paged_wasted_bytes = paged_wasted_tokens * BYTES_PER_TOKEN;

            double len_mean = len_count ? len_sum / static_cast<double>(len_count) : 0.0;
            double len_var = len_count ? (len_sq_sum / static_cast<double>(len_count) - len_mean * len_mean) : 0.0;
            double len_std = std::sqrt(std::max(0.0, len_var));

            lm_list.push_back(len_mean);
            ls_list.push_back(len_std);
            pw_list.push_back(paged_wasted_tokens);
            cw_list.push_back(contig_wasted_bytes);
            pp_list.push_back(paged_peak_bytes);
            cp_list.push_back(static_cast<double>(contig_peak_bytes));
            pr_list.push_back(static_cast<double>(paged_rejected));
            cr_list.push_back(static_cast<double>(contig_rejected));
            pt_list.push_back(static_cast<double>(paged_alloc_time_ns) / 1e6);
            ct_list.push_back(static_cast<double>(contig_alloc_time_ns) / 1e6);

            csv << rep << "," << mode.name << "," << len_mean << "," << len_std << ","
                << paged_peak_blocks << "," << paged_wasted_tokens << "," << paged_wasted_bytes << "," << paged_peak_bytes << "," << paged_rejected << "," << pt_list.back() << "," << (paged_no_leak ? 1 : 0) << ","
                << contig_peak_bytes << "," << contig_wasted_external << "," << contig_wasted_bytes << "," << contig_peak_bytes << "," << contig_rejected << "," << ct_list.back() << "," << (contig_no_leak ? 1 : 0) << "\n";
        }

        std::cout << "--- Mode: " << mode.name << " ---\n";
        std::cout << "  len_mean=" << std::fixed << std::setprecision(1) << mean_vec(lm_list)
                  << " len_std=" << mean_vec(ls_list) << "\n";
        std::cout << "  paged_avg_wasted_tokens=" << mean_vec(pw_list)
                  << " contig_avg_wasted_bytes=" << mean_vec(cw_list) << "\n";
        std::cout << "  paged_avg_peak_bytes=" << mean_vec(pp_list)
                  << " contig_avg_peak_bytes=" << mean_vec(cp_list) << "\n";
        std::cout << "  paged_avg_rejected=" << mean_vec(pr_list)
                  << " contig_avg_rejected=" << mean_vec(cr_list) << "\n";
        std::cout << "  paged_avg_time_ms=" << mean_vec(pt_list)
                  << " contig_avg_time_ms=" << mean_vec(ct_list) << "\n";
        std::cout << "  paged_no_leak=" << (all_paged_ok ? "true" : "false")
                  << " contig_no_leak=" << (all_contig_ok ? "true" : "false") << "\n\n";
    }

    std::cout << "Sanity check:\n";
    std::cout << "  - constant_max: both paged and contiguous waste ~0, few rejections.\n";
    std::cout << "  - higher variance => contiguous rejected/external waste rises sharply vs paged.\n";
    std::cout << "  - paged keeps allocating from scattered blocks; contiguous needs one contiguous chunk.\n";
    std::cout << "\nResults written to fragmentation_results.csv\n";

    csv.close();
    return 0;
}
