import time
from enum import Enum
from typing import Optional

import llm_allocator_cpp


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED = "FINISHED"


class Sequence:
    def __init__(self, seq_id: int, prompt_token_ids: list[int]) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: list[int] = []
        self.status = SequenceStatus.WAITING
        self.created_time = time.time()
        self.block_table: Optional[llm_allocator_cpp.SequenceBlockTable] = None

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def init_blocks(self, allocator: llm_allocator_cpp.PageAllocator) -> None:
        if self.block_table is not None:
            raise RuntimeError("Blocks already initialized for this sequence")
        self.block_table = llm_allocator_cpp.SequenceBlockTable(allocator)
        for _ in range(len(self.prompt_token_ids)):
            self.block_table.append_token(allocator)

    def append_token(self, token_id: int, allocator: llm_allocator_cpp.PageAllocator) -> None:
        if self.block_table is None:
            raise RuntimeError("Blocks not initialized - call init_blocks() first")
        self.output_token_ids.append(token_id)
        self.block_table.append_token(allocator)

    def free_blocks(self, allocator: llm_allocator_cpp.PageAllocator) -> None:
        if self.block_table is not None:
            self.block_table.release(allocator)
            self.block_table = None