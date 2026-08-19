import torch
import llm_allocator_cpp

class PagedKVCacheManager:
    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.allocator = llm_allocator_cpp.PageAllocator(total_blocks, block_size)
        
        
        self.key_cache = torch.empty(
            (total_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.value_cache = torch.empty(
            (total_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

    def create_sequence_table(self) -> llm_allocator_cpp.SequenceBlockTable:
        return llm_allocator_cpp.SequenceBlockTable(self.allocator)

    def write_kv_slot(
        self,
        table: llm_allocator_cpp.SequenceBlockTable,
        key_token: torch.Tensor,
        value_token: torch.Tensor
    ):
       
        table.append_token(self.allocator)
        
        token_index = table.get_tokens_count() - 1
        physical_blocks = table.get_physical_blocks()
        
        block_idx = physical_blocks[token_index // self.block_size]
        slot_idx = token_index % self.block_size
        
        self.key_cache[block_idx, slot_idx] = key_token
        self.value_cache[block_idx, slot_idx] = value_token

    def free_sequence(self, table: llm_allocator_cpp.SequenceBlockTable):
        table.release(self.allocator)