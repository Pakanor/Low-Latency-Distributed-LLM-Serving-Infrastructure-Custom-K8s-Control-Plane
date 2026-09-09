from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler, SchedulerOutputs
from app.memory.manager import PagedKVCacheManager


class LLMEngine:
    def __init__(
        self,
        model: nn.Module,
        kv_cache_manager: PagedKVCacheManager,
        scheduler: Scheduler
    ) -> None:
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = scheduler
        self.seq_counter = 0

    def add_request(self, prompt_token_ids: List[int], max_tokens: int = 128) -> int:
        self.seq_counter += 1
        seq = Sequence(
            seq_id=self.seq_counter,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens
        )
        self.scheduler.add_sequence(seq)
        return seq.seq_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_sequences()

    def step(self) -> List[Tuple[int, int]]:
       
        outputs: SchedulerOutputs = self.scheduler.schedule()
        
        results: List[Tuple[int, int]] = []

        if outputs.scheduled_prefills:
            for seq in outputs.scheduled_prefills:
                input_ids = torch.tensor(
                    [seq.prompt_token_ids], dtype=torch.long
                )
                
                # Forward pass przez model (w rzeczywistym modelu używamy tu PagedAttention)
                # Dlla potrzeb prostego silnika: pobieramy logity dla ostatniego tokena
                with torch.no_grad():
                    logits = self.model(input_ids)
                    next_token_logits = logits[0, -1, :]
                    next_token_id = int(torch.argmax(next_token_logits).item())

                seq.append_token(next_token_id)
                results.append((seq.seq_id, next_token_id))

        if outputs.scheduled_decodes:
            for seq in outputs.scheduled_decodes:
                last_token_id = seq.output_token_ids[-1]
                input_ids = torch.tensor([[last_token_id]], dtype=torch.long)

                with torch.no_grad():
                    logits = self.model(input_ids)
                    next_token_logits = logits[0, -1, :]
                    next_token_id = int(torch.argmax(next_token_logits).item())

                seq.append_token(next_token_id)
                results.append((seq.seq_id, next_token_id))

        return results