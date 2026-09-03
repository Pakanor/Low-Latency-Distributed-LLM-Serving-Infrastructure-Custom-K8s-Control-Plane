from dataclasses import dataclass
from typing import List
from app.scheduler.sequence import Sequence, SequenceStatus
import llm_allocator_cpp


@dataclass
class SchedulerOutputs:
    scheduled_prefills: List[Sequence]
    scheduled_decodes: List[Sequence]
    ignored_sequences: List[Sequence]


class Scheduler:
    def __init__(
        self,
        allocator: llm_allocator_cpp.PageAllocator,
        max_batch_size: int = 8,
        max_num_batched_tokens: int = 2048
    ) -> None:
        self.allocator = allocator
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
        block_size = self.allocator.get_block_size()

        running_to_keep: List[Sequence] = []
        for seq in self.running:
            if seq.status == SequenceStatus.FINISHED:
                seq.free_blocks(self.allocator)
                continue

            current_tokens = seq.total_len
            need_new_block = (current_tokens > 0) and (current_tokens % block_size == 0)

            if need_new_block and self.allocator.get_num_free_blocks() < 1:
                seq.status = SequenceStatus.WAITING
                seq.free_blocks(self.allocator)
                self.waiting.insert(0, seq)
            else:
                if need_new_block:
                    assert seq.block_table is not None
                    block_id = self.allocator.allocate_block()
                    seq.block_table.append_block(block_id)
                
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
            seq.init_blocks(self.allocator)
            assert seq.block_table is not None

            currently_allocated = len(seq.block_table.get_physical_blocks())
            needed = blocks_needed - currently_allocated

            if needed > 0:
                block_ids = self.allocator.allocate_blocks(needed)
                for b_id in block_ids:
                    seq.block_table.append_block(b_id)

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