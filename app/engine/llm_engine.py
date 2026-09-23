import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import K8sModelClient
import torch

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(
        self,
        kv_cache_manager: PagedKVCacheManager,
        scheduler: Scheduler,
        model_client: Optional[K8sModelClient] = None,
        model=None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = scheduler
        self.model_client = model_client
        self.model = model
        self.seq_counter = 0
        self.update_event: Optional[asyncio.Event] = None
        self._past_key_values: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

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

    def _local_generate_step(
        self,
        seq_id: int,
        input_ids: List[int],
    ) -> int:
        if self.model is None or not callable(getattr(self.model, "forward", None)):
            assert self.model_client is not None
            result = self.model_client.generate_step_tensor(
                seq_id=seq_id,
                input_ids=input_ids,
            )
            return result["next_token_id"]

        import torch as _torch
        with _torch.no_grad():
            input_tensor = _torch.tensor(input_ids, dtype=_torch.long).unsqueeze(0)
            past_kv = self._past_key_values.get(seq_id, None)
            outputs = self.model(input_tensor, past_key_values=past_kv, use_cache=True)
            next_token_id = int(_torch.argmax(outputs.logits[0, -1, :]))
            self._past_key_values[seq_id] = list(outputs.past_key_values)
        return next_token_id

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

        for seq in prefills:
            next_token = self._local_generate_step(seq.seq_id, seq.prompt_token_ids)
            seq.append_token(next_token)

        for seq in decodes:
            last_token = [seq.output_token_ids[-1]] if seq.output_token_ids else [seq.prompt_token_ids[-1]]
            next_token = self._local_generate_step(seq.seq_id, last_token)
            seq.append_token(next_token)

        finished = [s for s in all_active_seqs if s.status == SequenceStatus.FINISHED]
        running = [s for s in all_active_seqs if s.status == SequenceStatus.RUNNING]

        for seq in finished:
            self._past_key_values.pop(seq.seq_id, None)

        if self.update_event is not None:
            self.update_event.set()

        return {"finished": finished, "running": running}
