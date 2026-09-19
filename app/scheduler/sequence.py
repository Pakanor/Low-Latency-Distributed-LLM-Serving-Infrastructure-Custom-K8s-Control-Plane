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
    def __init__(
        self,
        seq_id: int,
        prompt: str = "",
        prompt_token_ids: Optional[List[int]] = None,
        max_tokens: int = 128,
        eos_token_id: Optional[int] = None
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids or [])
        if not prompt_token_ids and not prompt:
            raise ValueError(f"Sequence {seq_id}: Prompt and prompt_token_ids cannot be both empty.")

        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING
        self.created_time = time.time()
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        self.block_table: Optional[llm_allocator_cpp.SequenceBlockTable] = None
        self.logical_block_table: List[int] = []

    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.get_len()

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if token_id == self.eos_token_id or len(self.output_token_ids) >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED