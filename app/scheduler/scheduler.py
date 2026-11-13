from dataclasses import dataclass
from typing import List, Optional

from app.config import DEFAULT_CONFIG
from app.scheduler.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutputs:
    scheduled_prefills: List[Sequence]
    scheduled_decodes: List[Sequence]
    ignored_sequences: List[Sequence]


class Scheduler:
    def __init__(
        self,
        kv_cache_manager=None,
        max_batch_size: int = 8,
        max_num_batched_tokens: int = DEFAULT_CONFIG.max_model_len,
        page_allocator=None,
        block_size: Optional[int] = None,
        allocator=None,
        max_model_len: Optional[int] = DEFAULT_CONFIG.max_model_len,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.allocator = (
            allocator
            or page_allocator
            or (kv_cache_manager.get_allocator() if kv_cache_manager is not None else None)
        )
        if self.allocator is None:
            raise ValueError("Scheduler requires a PageAllocator or KV cache manager")
        self.block_size = block_size or getattr(kv_cache_manager, "block_size", None)
        if not self.block_size or self.block_size <= 0:
            raise ValueError("Scheduler requires a positive block_size")
        if max_model_len is None or max_model_len <= 0:
            raise ValueError("Scheduler requires max_model_len")
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.preemption_count = 0
        self.preemption_counts: dict[int, int] = {}

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        if self.max_model_len is not None and seq.total_len + seq.max_tokens > self.max_model_len:
            raise ValueError(
                f"Sequence {seq.seq_id} exceeds max_model_len: "
                f"prompt+max_tokens ({seq.total_len + seq.max_tokens}) > {self.max_model_len}"
            )
        total_blocks = self.allocator.get_total_blocks()
        required_blocks = (
            seq.total_len + seq.max_tokens + self.block_size - 1
        ) // self.block_size
        if required_blocks > total_blocks:
            raise ValueError(
                f"Sequence {seq.seq_id} requires {required_blocks} blocks for "
                f"prompt+max_tokens ({seq.total_len + seq.max_tokens}) but the "
                f"KV cache pool has only {total_blocks} blocks"
            )
        self.waiting.append(seq)

    def has_unfinished_sequences(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def _allocate_prefix(self, seq: Sequence) -> None:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.allocate_prefix_blocks(seq)
            return

        raise RuntimeError("Scheduler requires kv_cache_manager for block allocation")

    def _free_blocks(self, seq: Sequence) -> None:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.free_sequence(seq)
            return

        raise RuntimeError("Scheduler requires kv_cache_manager for block freeing")

    def schedule(self) -> SchedulerOutputs:
        scheduled_decodes: List[Sequence] = []
        scheduled_prefills: List[Sequence] = []

        active_running: List[Sequence] = []
        for seq in self.running:
            if seq.is_finished():
                self._free_blocks(seq)
                continue

            if len(scheduled_decodes) >= self.max_batch_size:
                active_running.append(seq)
                continue

            if seq.block_table is not None:
                current_tokens = seq.block_table.get_tokens_count()
                needs_new_block = current_tokens > 0 and current_tokens % self.block_size == 0
            else:
                needs_new_block = False

            if needs_new_block and self.allocator.get_num_free_blocks() < 1:
                seq.status = SequenceStatus.WAITING
                self._free_blocks(seq)
                self.preemption_count += 1
                self.preemption_counts[seq.seq_id] = (
                    self.preemption_counts.get(seq.seq_id, 0) + 1
                )
                self.waiting.insert(0, seq)
                continue

            if self.kv_cache_manager is not None and seq.block_table is not None:
                self.kv_cache_manager.allocate_slot_for_next_token(seq)

            active_running.append(seq)
            scheduled_decodes.append(seq)

        self.running = active_running
        current_batch_tokens = len(scheduled_decodes)
        ignored: List[Sequence] = []

        while self.waiting:
            seq = self.waiting[0]
            seq_len = seq.total_len
            needed_blocks = (seq_len + self.block_size - 1) // self.block_size
            if needed_blocks > self.allocator.get_num_free_blocks():
                ignored.append(seq)
                break
            if current_batch_tokens + seq_len > self.max_num_batched_tokens:
                ignored.append(seq)
                break

            self.waiting.pop(0)
            self._allocate_prefix(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_prefills.append(seq)
            current_batch_tokens += seq_len

        return SchedulerOutputs(
            scheduled_prefills=scheduled_prefills,
            scheduled_decodes=scheduled_decodes,
            ignored_sequences=ignored,
        )

    def free_sequence(self, seq: Sequence) -> None:
        self._free_blocks(seq)
        seq.status = SequenceStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.swapped:
            self.swapped.remove(seq)

    def is_full(self) -> bool:
        if self.allocator is None:
            return False
        return self.allocator.get_num_free_blocks() < 1