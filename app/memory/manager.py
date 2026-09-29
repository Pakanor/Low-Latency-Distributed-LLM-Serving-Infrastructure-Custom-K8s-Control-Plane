import torch
import llm_allocator_cpp
from typing import List, Tuple, Optional, Union
from app.scheduler.sequence import Sequence, SequenceStatus
from app.memory.strategies import (
    KVCacheMemoryStrategy,
    CpuKVCacheStrategy,
    CudaKVCacheStrategy,
    detect_strategy,
)


class PagedKVCacheManager:
    def __init__(
        self,
        total_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        total_cpu_blocks: int = 32,
        dtype: torch.dtype = torch.float32,
        num_layers: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        strategy: Optional[KVCacheMemoryStrategy] = None,
    ) -> None:
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype

        if strategy is not None:
            self.strategy = strategy
        elif device is not None:
            self.strategy = _strategy_from_device(device)
        else:
            self.strategy = detect_strategy()

        self.allocator = llm_allocator_cpp.PageAllocator(total_blocks, block_size, total_cpu_blocks)
        self.swapped_seqs = set()

        self.key_cache, self.value_cache = self.strategy.allocate_cache(
            total_blocks, block_size, num_layers, num_heads, head_dim, dtype
        )

    @property
    def device(self) -> str:
        return str(self.strategy.device)

    def get_allocator(self) -> llm_allocator_cpp.PageAllocator:
        return self.allocator

    def init_sequence_table(self, seq: Sequence) -> None:
        if seq.block_table is None:
            seq.block_table = llm_allocator_cpp.SequenceBlockTable(self.allocator)

    def allocate_prefix_blocks(self, seq: Sequence) -> None:
        self.init_sequence_table(seq)
        assert seq.block_table is not None

        tokens_count = seq.block_table.get_tokens_count()
        needed_tokens = seq.total_len - tokens_count
        for _ in range(needed_tokens):
            seq.block_table.append_token(self.allocator)

    def allocate_slot_for_next_token(self, seq: Sequence) -> None:
        assert seq.block_table is not None
        seq.block_table.append_token(self.allocator)

    def free_sequence(self, seq: Sequence) -> None:
        if seq.block_table is not None:
            seq.block_table.release(self.allocator)
            seq.block_table = None

    def write_prefix_kv(
        self,
        seq: Sequence,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        if keys.shape != values.shape:
            raise ValueError(f"Keys shape {keys.shape} != Values shape {values.shape}")

        seq_len = keys.shape[1]
        if seq_len == 0 or seq.block_table is None:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        physical_blocks = seq.block_table.get_physical_blocks()

        block_offsets = torch.tensor(
            physical_blocks, dtype=torch.long, device=self.device
        ) * self.block_size
        slot_offsets = torch.arange(
            self.block_size, dtype=torch.long, device=self.device
        )

        all_slots = (block_offsets.unsqueeze(1) + slot_offsets.unsqueeze(0)).flatten()
        slot_mapping = all_slots[:seq_len]

        self.key_cache.view(-1, self.num_layers, self.num_heads, self.head_dim)[slot_mapping] = keys.transpose(0, 1)
        self.value_cache.view(-1, self.num_layers, self.num_heads, self.head_dim)[slot_mapping] = values.transpose(0, 1)

        return slot_mapping

    def write_single_token_kv(
        self,
        seq: Sequence,
        key_token: torch.Tensor,
        value_token: torch.Tensor
    ) -> None:
        assert seq.block_table is not None
        token_index = seq.block_table.get_tokens_count() - 1
        physical_blocks = seq.block_table.get_physical_blocks()

        block_idx = token_index // self.block_size
        slot_idx = token_index % self.block_size
        physical_block_id = physical_blocks[block_idx]

        self.key_cache[physical_block_id, slot_idx] = key_token
        self.value_cache[physical_block_id, slot_idx] = value_token

    def gather_kv_cache(
        self,
        seq: Sequence,
        read_len: Optional[int] = None,
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        if seq.block_table is None:
            return None

        physical_blocks = seq.block_table.get_physical_blocks()
        tokens_count = seq.block_table.get_tokens_count()

        if read_len is None:
            read_len = tokens_count

        if read_len <= 0:
            return None

        block_data = torch.stack([self.key_cache[bid] for bid in physical_blocks])
        block_data_v = torch.stack([self.value_cache[bid] for bid in physical_blocks])

        flat_k = block_data.view(-1, self.num_layers, self.num_heads, self.head_dim)[:read_len]
        flat_v = block_data_v.view(-1, self.num_layers, self.num_heads, self.head_dim)[:read_len]

        result = []
        for layer_idx in range(self.num_layers):
            key = flat_k[:, layer_idx, :, :]
            value = flat_v[:, layer_idx, :, :]
            key = key.unsqueeze(0).transpose(1, 2)
            value = value.unsqueeze(0).transpose(1, 2)
            result.append((key, value))

        return tuple(result)

    def build_block_tables(
        self,
        sequences: List[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sequences)
        if batch_size == 0:
            return (
                torch.empty((0, 0), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.int32, device=self.device)
            )

        raw_block_tables = [
            seq.block_table.get_physical_blocks() if seq.block_table else []
            for seq in sequences
        ]
        context_lens_list = [
            seq.block_table.get_tokens_count() if seq.block_table else 0
            for seq in sequences
        ]

        max_blocks = max(len(b) for b in raw_block_tables) if raw_block_tables else 0
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
            block_tables_cpu.to(self.device, non_blocking=True),
            context_lens_cpu.to(self.device, non_blocking=True)
        )

    def swap_out_sequence(self, seq: Sequence) -> None:
        assert seq.block_table is not None
        physical_blocks = seq.block_table.get_physical_blocks()

        if self.strategy.requires_copy_on_swap:
            pass

        for block_id in physical_blocks:
            self.allocator.swap_out(block_id)
        self.swapped_seqs.add(seq.seq_id)
        seq.status = SequenceStatus.SWAPPED

    def swap_in_sequence(self, seq: Sequence) -> None:
        assert seq.block_table is not None
        old_physical_blocks = seq.block_table.get_physical_blocks()
        new_physical_blocks = []
        for block_id in old_physical_blocks:
            new_block = self.allocator.swap_in(block_id)
            new_physical_blocks.append(new_block)
        for old_id, new_id in zip(old_physical_blocks, new_physical_blocks):
            seq.block_table.replace_block(old_id, new_id)
        self.swapped_seqs.discard(seq.seq_id)
        seq.status = SequenceStatus.RUNNING


def _strategy_from_device(device: Union[str, torch.device]) -> KVCacheMemoryStrategy:
    dev = torch.device(device)
    if dev.type == "cuda":
        return CudaKVCacheStrategy(device_id=dev.index or 0)
    return CpuKVCacheStrategy()
