#ifndef PAGED_KERNELS_H
#define PAGED_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t num_query_heads;
    int32_t num_query_tokens;
    int32_t head_dim;
    int32_t num_layers;
    int32_t total_blocks;
    int32_t block_size;
    int32_t num_kv_heads;
    int32_t num_context_tokens;
} PagedAttentionDims;

void paged_attention(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int32_t layer_idx,
    const PagedAttentionDims dims,
    const int32_t* gather_slots,
    const uint8_t* attention_mask,
    const float scaling,
    float* output,
    float* scratch,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif /* PAGED_KERNELS_H */
