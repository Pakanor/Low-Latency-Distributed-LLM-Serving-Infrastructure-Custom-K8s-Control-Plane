from dataclasses import dataclass
from typing import List
from app.scheduler.sequence import Sequence, SequenceStatus
from app.memory.manager import PagedKVCacheManager


@dataclass
class SchedulerOutputs:
    scheduled_prefills: List[Sequence]
    scheduled_decodes: List[Sequence]
    ignored_sequences: List[Sequence]


class Scheduler:
    def __init__(
        self,
        kv_cache_manager: PagedKVCacheManager,
        max_batch_size: int = 8,
        max_num_batched_tokens: int = 2048
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.allocator = kv_cache_manager.get_allocator()
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def schedule(self) -> SchedulerOutputs:
        scheduled_decodes: List[Sequence] = []
        block_size = self.kv_cache_manager.block_size

        running_to_keep: List[Sequence] = []
        for seq in self.running:
            if seq.status == SequenceStatus.FINISHED:
                self.kv_cache_manager.free_sequence(seq)
                continue

            current_tokens = seq.total_len
            need_new_block = (current_tokens > 0) and (current_tokens % block_size == 0)

            if need_new_block and self.allocator.get_num_free_blocks() < 1:
                seq.status = SequenceStatus.WAITING
                self.kv_cache_manager.free_sequence(seq)
                self.waiting.insert(0, seq)
            else:
                if need_new_block:
                    self.kv_cache_manager.allocate_slot_for_next_token(seq)
                
                scheduled_decodes.append(seq)
                running_to_keep.append(seq)

        self.running = running_to_keep

        num_free_blocks = self.allocator.get_num_free_blocks()
        scheduled_prefills: List[Sequence] = []
        curr_batch_size = len(scheduled_decodes)
        curr_tokens = 0

        while self.waiting and curr_batch_size < self.max_batch_size:
            seq = self.waiting[0]
            seq_len = seq.total_len
            blocks_needed = (seq_len + block_size - 1) // block_size

            if blocks_needed > num_free_blocks:
                break

            if curr_tokens + seq_len > self.max_num_batched_tokens:
                break

            seq = self.waiting.pop(0)
            
            self.kv_cache_manager.allocate_prefix_blocks(seq)

            seq.status = SequenceStatus.RUNNING

            num_free_blocks -= blocks_needed
            curr_tokens += seq_len
            curr_batch_size += 1

            scheduled_prefills.append(seq)
            self.running.append(seq)

        return SchedulerOutputs(
            scheduled_prefills=scheduled_prefills,
            scheduled_decodes=scheduled_decodes,
            ignored_sequences=[]
        )