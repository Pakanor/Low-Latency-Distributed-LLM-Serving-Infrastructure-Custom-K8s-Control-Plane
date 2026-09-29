import logging
import asyncio
import torch
from typing import List, Dict, Optional, Tuple
from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import K8sModelClient
from app.model.paged_attention import (
    PagedStepContext,
    register_paged_attention,
    run_paged_forward,
)

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
        self._paged_attention_registered = False

    @property
    def model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

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
        is_prefill: bool = False,
        seq: Optional[Sequence] = None,
    ) -> Tuple[int, Optional[Tuple[torch.Tensor, ...]]]:
        past_kv = None
        if not is_prefill and seq is not None and self.kv_cache_manager is not None:
            read_len = seq.block_table.get_tokens_count() - 1
            past_kv = self.kv_cache_manager.gather_kv_cache(seq, read_len=read_len)

        if self.model is None or not callable(getattr(self.model, "forward", None)):
            assert self.model_client is not None
            result = self.model_client.generate_step_tensor(
                seq_id=seq_id,
                input_ids=input_ids,
                past_key_values=past_kv,
            )
            new_past_kv = result.get("past_key_values")
            if seq is not None and new_past_kv is not None and self.kv_cache_manager is not None:
                self._write_paged_kv_cache(seq, new_past_kv, is_prefill=is_prefill)
            return result["next_token_id"], new_past_kv

        with torch.no_grad():
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            outputs = self.model(input_tensor, past_key_values=past_kv, use_cache=True)
            next_token_id = int(torch.argmax(outputs.logits[0, -1, :]))
            self._write_paged_kv_cache(seq, outputs.past_key_values, is_prefill=is_prefill)
        return next_token_id, outputs.past_key_values

    def _write_paged_kv_cache(
        self,
        seq: Sequence,
        past_key_values: Tuple[torch.Tensor, ...],
        is_prefill: bool = False,
    ) -> None:
        """Write model's KV cache to PagedKVCacheManager's contiguous tensors."""
        if self.kv_cache_manager is None or seq.block_table is None:
            return

        num_layers = len(past_key_values)
        keys_list = []
        values_list = []

        for layer_idx in range(num_layers):
            key = past_key_values[layer_idx][0]
            value = past_key_values[layer_idx][1]
            key = key.squeeze(0).transpose(0, 1)
            value = value.squeeze(0).transpose(0, 1)
            keys_list.append(key)
            values_list.append(value)

        keys = torch.stack(keys_list)
        values = torch.stack(values_list)

        if is_prefill:
            self.kv_cache_manager.write_prefix_kv(seq, keys, values)
        else:
            last_keys = keys[:, -1]
            last_values = values[:, -1]
            self.kv_cache_manager.write_single_token_kv(seq, last_keys, last_values)

    def _group_inputs(
        self,
        sequences: List[Sequence],
        is_prefill: bool,
    ) -> Tuple[List[List[int]], List[int]]:
        """Tokens each sequence contributes to this step, and where they land."""
        if is_prefill:
            token_lists = [seq.prompt_token_ids for seq in sequences]
            start_positions = [0] * len(sequences)
        else:
            token_lists = [self._last_decode_token(seq) for seq in sequences]
            start_positions = [
                seq.block_table.get_tokens_count() - 1 for seq in sequences
            ]
        return token_lists, start_positions

    def _last_decode_token(self, seq: Sequence) -> List[int]:
        if seq.output_token_ids:
            return [seq.output_token_ids[-1]]
        return [seq.prompt_token_ids[-1]]

    def _ensure_paged_attention(self) -> None:
        if self._paged_attention_registered:
            return
        register_paged_attention(self.model)
        self._paged_attention_registered = True

    def _paged_forward_group(
        self,
        sequences: List[Sequence],
        is_prefill: bool,
    ) -> List[int]:
        self._ensure_paged_attention()

        token_lists, start_positions = self._group_inputs(sequences, is_prefill)
        new_token_counts = [len(tokens) for tokens in token_lists]

        flat_input = torch.tensor(
            [token for tokens in token_lists for token in tokens],
            dtype=torch.long,
            device=self.model_device,
        ).unsqueeze(0)

        context = PagedStepContext(
            self.kv_cache_manager,
            sequences,
            start_positions,
            new_token_counts,
        )
        logits = run_paged_forward(self.model, context, flat_input)

        cu_seqlens = context.cu_seqlens
        return [
            int(logits[0, int(cu_seqlens[index + 1]) - 1].argmax())
            for index in range(len(sequences))
        ]

    def _forward_group(
        self,
        sequences: List[Sequence],
        is_prefill: bool,
    ) -> List[int]:
        if self.model is None or not callable(getattr(self.model, "forward", None)):
            token_lists, _ = self._group_inputs(sequences, is_prefill)
            return [
                self._local_generate_step(
                    seq.seq_id, tokens, is_prefill=is_prefill, seq=seq
                )[0]
                for seq, tokens in zip(sequences, token_lists)
            ]
        return self._paged_forward_group(sequences, is_prefill)

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

        for sequences, is_prefill in ((prefills, True), (decodes, False)):
            if not sequences:
                continue
            for seq, next_token in zip(sequences, self._forward_group(sequences, is_prefill)):
                seq.append_token(next_token)

        finished = [s for s in all_active_seqs if s.status == SequenceStatus.FINISHED]
        running = [s for s in all_active_seqs if s.status == SequenceStatus.RUNNING]

        for seq in finished:
            self.scheduler.free_sequence(seq)

        if self.update_event is not None:
            self.update_event.set()

        return {"finished": finished, "running": running}
