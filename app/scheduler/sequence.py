import time
from enum import Enum
from typing import List, Optional
import llm_allocator_cpp


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED = "FINISHED"


class Sequence:
    def __init__(self, seq_id: int, prompt_token_ids: List[int], max_tokens: int = 128) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING
        self.created_time = time.time()
        self.max_tokens = max_tokens
        
        self.block_table: Optional[llm_allocator_cpp.SequenceBlockTable] = None

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) >= self.max_tokens:
            self.status = SequenceStatus.FINISHED