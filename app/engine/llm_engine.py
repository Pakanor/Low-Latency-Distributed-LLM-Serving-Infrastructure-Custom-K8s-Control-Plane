import logging
from typing import List, Dict, Optional
from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import K8sModelClient

logger = logging.getLogger(__name__)

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
        try:
            outputs = self.scheduler.schedule()
        except RuntimeError as e:
            logger.warning(f"Scheduler OOM, skipping step: {e}")
            return {"finished": [], "running": []}

        prefills = outputs.scheduled_prefills
        decodes = outputs.scheduled_decodes

        if not prefills and not decodes:
            return {"finished": [], "running": []}

        all_active_seqs = prefills + decodes

        block_tables, context_lens = self.kv_cache_manager.build_block_tables(all_active_seqs)

        for i, seq in enumerate(prefills):
            row = i
            seq_block_table = block_tables[row].tolist()
            ctx_len = int(context_lens[row].item())

            result = self.model_client.generate_step_tensor(
                input_ids=seq.prompt_token_ids,
                block_table=seq_block_table,
                context_len=ctx_len
            )
            next_token = result["next_token_id"]
            seq.append_token(next_token)
            keys_tensor = result.get("keys")
            values_tensor = result.get("values")
            if keys_tensor is not None and values_tensor is not None:
                self.kv_cache_manager.write_prefix_kv(seq, keys_tensor, values_tensor)

        for i, seq in enumerate(decodes):
            row = len(prefills) + i
            seq_block_table = block_tables[row].tolist()
            ctx_len = int(context_lens[row].item())

            last_token = [seq.output_token_ids[-1]] if seq.output_token_ids else [seq.prompt_token_ids[-1]]

            result = self.model_client.generate_step_tensor(
                input_ids=last_token,
                block_table=seq_block_table,
                context_len=ctx_len
            )
            next_token = result["next_token_id"]
            seq.append_token(next_token)
            key_tensor = result.get("key")
            value_tensor = result.get("value")
            if key_tensor is not None and value_tensor is not None:
                self.kv_cache_manager.write_single_token_kv(seq, key_tensor, value_tensor)

        finished = [s for s in all_active_seqs if s.status == SequenceStatus.FINISHED]
        running = [s for s in all_active_seqs if s.status == SequenceStatus.RUNNING]

        return {"finished": finished, "running": running}
