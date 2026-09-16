from typing import List, Dict, Optional
import torch
from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager


class LLMEngine:
    def __init__(
        self,
        kv_cache_manager: PagedKVCacheManager,
        scheduler: Scheduler,
        model: Optional[torch.nn.Module] = None
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = scheduler
        self.model = model
        self.seq_counter = 0

    def add_request(
        self, 
        prompt_token_ids: List[int], 
        max_tokens: int = 128
    ) -> Sequence:
        self.seq_counter += 1
        seq = Sequence(
            seq_id=self.seq_counter, 
            prompt_token_ids=prompt_token_ids, 
            max_tokens=max_tokens
        )
        self.scheduler.add_sequence(seq)
        return seq

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_sequences()

    def step(self) -> Dict[str, List[Sequence]]:
        outputs = self.scheduler.schedule()
        prefills = outputs.scheduled_prefills
        decodes = outputs.scheduled_decodes

        if not prefills and not decodes:
            return {"finished": [], "running": []}

        all_active_seqs = prefills + decodes

        block_tables, context_lens = self.kv_cache_manager.build_block_tables(all_active_seqs)

        for seq in prefills:
            seq_len = seq.total_len
            mock_keys = torch.randn(
                seq_len,
                self.kv_cache_manager.num_heads,
                self.kv_cache_manager.head_dim,
                device=self.kv_cache_manager.device,
                dtype=self.kv_cache_manager.dtype
            )
            mock_values = torch.randn(
                seq_len,
                self.kv_cache_manager.num_heads,
                self.kv_cache_manager.head_dim,
                device=self.kv_cache_manager.device,
                dtype=self.kv_cache_manager.dtype
            )

            self.kv_cache_manager.write_prefix_kv(seq, mock_keys, mock_values)

            generated_token = 100 + seq.seq_id
            seq.append_token(generated_token)

        for seq in decodes:
            mock_key_token = torch.randn(
                self.kv_cache_manager.num_heads,
                self.kv_cache_manager.head_dim,
                device=self.kv_cache_manager.device,
                dtype=self.kv_cache_manager.dtype
            )
            mock_value_token = torch.randn(
                self.kv_cache_manager.num_heads,
                self.kv_cache_manager.head_dim,
                device=self.kv_cache_manager.device,
                dtype=self.kv_cache_manager.dtype
            )

            self.kv_cache_manager.write_single_token_kv(seq, mock_key_token, mock_value_token)

            generated_token = 200 + seq.total_len
            seq.append_token(generated_token)

        finished = [s for s in all_active_seqs if s.status == SequenceStatus.FINISHED]
        running = [s for s in all_active_seqs if s.status == SequenceStatus.RUNNING]

        return {"finished": finished, "running": running}