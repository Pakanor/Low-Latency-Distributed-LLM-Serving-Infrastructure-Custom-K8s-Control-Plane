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


    def build_block_tables(
        self,
        tables: list[llm_allocator_cpp.SequenceBlockTable]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = len(tables)
        if batch_size == 0:
            return (
                torch.empty((0, 0), dtype=torch.int32, device=self.key_cache.device),
                torch.empty((0,), dtype=torch.int32, device=self.key_cache.device)
            )

        raw_block_tables = [table.get_physical_blocks() for table in tables]
        context_lens_list = [table.get_tokens_count() for table in tables]

        max_blocks = max(len(blocks) for blocks in raw_block_tables)

        block_tables_cpu = torch.full(
            (batch_size, max_blocks),
            fill_value=-1,
            dtype=torch.int32
        )

        for i, blocks in enumerate(raw_block_tables):
            if blocks:
                block_tables_cpu[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)

        context_lens_cpu = torch.tensor(context_lens_list, dtype=torch.int32)

        return (
            block_tables_cpu.to(self.key_cache.device),
            context_lens_cpu.to(self.key_cache.device)
        )