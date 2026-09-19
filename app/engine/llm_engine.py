from typing import List, Dict, Optional
import torch
from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import K8sModelClient

class LLMEngine:
    def __init__(
        self,
        kv_cache_manager: PagedKVCacheManager,
        scheduler: Scheduler,
        model_client: Optional[K8sModelClient] = None
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = scheduler
        self.model_client = model_client or K8sModelClient()
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

            sequence_positions = {id(seq): index for index, seq in enumerate(all_active_seqs)}

            for seq in prefills:
                row = sequence_positions[id(seq)]
                seq_block_table = block_tables[row].tolist()
                ctx_len = int(context_lens[row].item())

                next_token = self.model_client.generate_step_tensor(
                    input_ids=seq.prompt_token_ids,
                    block_table=seq_block_table,
                    context_len=ctx_len
                )
                seq.append_token(next_token)

            for seq in decodes:
                row = sequence_positions[id(seq)]
                seq_block_table = block_tables[row].tolist()
                ctx_len = int(context_lens[row].item())

                last_token = [seq.output_token_ids[-1]] if seq.output_token_ids else [seq.prompt_token_ids[-1]]

                next_token = self.model_client.generate_step_tensor(
                    input_ids=last_token,
                    block_table=seq_block_table,
                    context_len=ctx_len
                )
                seq.append_token(next_token)

            finished = [s for s in all_active_seqs if s.status == SequenceStatus.FINISHED]
            running = [s for s in all_active_seqs if s.status == SequenceStatus.RUNNING]

            return {"finished": finished, "running": running}