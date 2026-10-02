#include "paged_kernels.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

void paged_attention(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int32_t layer_idx,
    const PagedAttentionDims dims,
    const int32_t* __restrict__ gather_slots,
    const uint8_t* __restrict__ attention_mask,
    const float scaling,
    float* __restrict__ output,
    float* __restrict__ scratch,
    void* stream
) {
    (void)stream;

    const int32_t num_query_heads    = dims.num_query_heads;
    const int32_t num_query_tokens   = dims.num_query_tokens;
    const int32_t head_dim           = dims.head_dim;
    const int32_t total_blocks       = dims.total_blocks;
    const int32_t block_size         = dims.block_size;
    const int32_t num_kv_heads       = dims.num_kv_heads;
    const int32_t num_context_tokens = dims.num_context_tokens;

    if (num_context_tokens == 0) {
        std::memset(output, 0,
            static_cast<size_t>(num_query_tokens) * num_query_heads * head_dim * sizeof(float));
        return;
    }

    const int64_t kv_head_dim  = static_cast<int64_t>(num_kv_heads) * head_dim;
    const int64_t layer_items  = static_cast<int64_t>(total_blocks) * block_size * kv_head_dim;
    const float* __restrict__ layer_k = key_cache + static_cast<int64_t>(layer_idx) * layer_items;
    const float* __restrict__ layer_v = value_cache + static_cast<int64_t>(layer_idx) * layer_items;

    const int32_t num_groups = num_query_heads / num_kv_heads;
    const int64_t q_head_stride = static_cast<int64_t>(num_query_tokens) * head_dim;
    const int64_t out_head_stride = static_cast<int64_t>(num_query_heads) * head_dim;
    const int64_t scratch_per_head = static_cast<int64_t>(num_context_tokens);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int32_t q = 0; q < num_query_tokens; ++q) {
        for (int32_t h_q = 0; h_q < num_query_heads; ++h_q) {
            const int32_t h_kv = h_q / num_groups;
            const float* __restrict__ q_ptr = query + static_cast<int64_t>(h_q) * q_head_stride + q * head_dim;
            float* __restrict__ my_scores = scratch + (static_cast<int64_t>(q) * num_query_heads + h_q) * scratch_per_head;

            float max_score = -std::numeric_limits<float>::infinity();

            for (int32_t c = 0; c < num_context_tokens; ++c) {
                const uint8_t attend = attention_mask[static_cast<int64_t>(q) * num_context_tokens + c];
                if (!attend) {
                    my_scores[c] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const int32_t slot = gather_slots[c];
                const float* __restrict__ k_ptr = layer_k + static_cast<int64_t>(slot) * kv_head_dim + h_kv * head_dim;

                float dot = 0.0f;
                for (int32_t d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
                const float score = dot * scaling;
                my_scores[c] = score;
                if (score > max_score) max_score = score;
            }

            if (max_score == -std::numeric_limits<float>::infinity()) max_score = 0.0f;

            float sum_exp = 0.0f;
            for (int32_t c = 0; c < num_context_tokens; ++c) {
                if (my_scores[c] > -1e30f) {
                    my_scores[c] = std::exp(my_scores[c] - max_score);
                    sum_exp += my_scores[c];
                } else {
                    my_scores[c] = 0.0f;
                }
            }
            if (sum_exp > 0.0f) {
                const float inv_sum = 1.0f / sum_exp;
                for (int32_t c = 0; c < num_context_tokens; ++c) {
                    my_scores[c] *= inv_sum;
                }
            }

            float* __restrict__ out_ptr = output + static_cast<int64_t>(q) * out_head_stride + h_q * head_dim;
            for (int32_t d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int32_t c = 0; c < num_context_tokens; ++c) {
                    const int32_t slot = gather_slots[c];
                    const float* __restrict__ v_ptr = layer_v + static_cast<int64_t>(slot) * kv_head_dim + h_kv * head_dim;
                    val += my_scores[c] * v_ptr[d];
                }
                out_ptr[d] = val;
            }
        }
    }
}

#if LLM_INFRA_HIP_ENABLED
#error "HIP paged attention kernel must be implemented in a separate .hip/.cu file with FlashAttention-style algorithms (shared memory, warp reductions, vectorized loads). Do not compile CPU code paths as GPU kernels."
#endif
