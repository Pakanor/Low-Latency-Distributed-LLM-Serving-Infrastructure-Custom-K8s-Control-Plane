from typing import List, Optional, Sequence as SequenceABC

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from app.memory.manager import PagedKVCacheManager
from app.model.attention_backends import AttentionBackend, TorchSdpBackend, select_backend
from app.scheduler.sequence import Sequence

PAGED_ATTENTION_NAME = "paged"

_FALLBACK_BACKEND = TorchSdpBackend()
_BACKEND: Optional[AttentionBackend] = None


def _backend() -> AttentionBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = select_backend()
    return _BACKEND


def set_backend(backend: Optional[AttentionBackend]) -> None:
    global _BACKEND
    _BACKEND = backend


class PagedStepContext:

    def __init__(
        self,
        store: PagedKVCacheManager,
        sequences: List[Sequence],
        start_positions: List[int],
        new_token_counts: List[int],
    ) -> None:
        if not (len(sequences) == len(start_positions) == len(new_token_counts)):
            raise ValueError(
                f"sequences ({len(sequences)}), start_positions ({len(start_positions)}) "
                f"and new_token_counts ({len(new_token_counts)}) must have the same length"
            )

        self.store = store
        self.sequences = list(sequences)
        self.start_positions = list(start_positions)
        self.new_token_counts = list(new_token_counts)
        self.context_lens = [
            start + count for start, count in zip(start_positions, new_token_counts)
        ]

        self.cu_seqlens = _cumulative_offsets(new_token_counts)
        self.cu_context_offsets = _cumulative_offsets(self.context_lens)

        self.scatter_slots = store.build_slot_mapping(
            self.sequences, self.start_positions, self.new_token_counts
        )
        self.gather_slots = store.build_slot_mapping(
            self.sequences, [0] * len(self.sequences), self.context_lens
        )

        self._mask: Optional[torch.Tensor] = None

    @property
    def num_query_tokens(self) -> int:
        return int(self.cu_seqlens[-1])

    @property
    def num_context_tokens(self) -> int:
        return int(self.cu_context_offsets[-1])

    def position_ids(self, device: torch.device) -> torch.Tensor:
        positions = []
        for start, count in zip(self.start_positions, self.new_token_counts):
            positions.append(
                torch.arange(start, start + count, dtype=torch.long, device=device)
            )
        if not positions:
            return torch.empty((1, 0), dtype=torch.long, device=device)
        return torch.cat(positions).unsqueeze(0)

    def attention_mask(self, device: torch.device) -> torch.Tensor:
        if self._mask is not None and self._mask.device == device:
            return self._mask

        query_index = _sequence_index(self.new_token_counts, device)
        context_index = _sequence_index(self.context_lens, device)

        cu_seqlens = self.cu_seqlens.to(device)
        cu_context_offsets = self.cu_context_offsets.to(device)
        starts = torch.tensor(self.start_positions, dtype=torch.long, device=device)

        query_position = torch.arange(self.num_query_tokens, device=device)
        query_position = query_position - cu_seqlens[query_index] + starts[query_index]

        context_position = torch.arange(self.num_context_tokens, device=device)
        context_position = context_position - cu_context_offsets[context_index]

        same_sequence = context_index.unsqueeze(0) == query_index.unsqueeze(1)
        within_causal_window = context_position.unsqueeze(0) <= query_position.unsqueeze(1)

        self._mask = (same_sequence & within_causal_window).unsqueeze(0).unsqueeze(0)
        return self._mask


def _cumulative_offsets(counts: List[int]) -> torch.Tensor:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.long)


def _sequence_index(counts: List[int], device: torch.device) -> torch.Tensor:
    if not counts:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.long, device=device),
        torch.tensor(counts, dtype=torch.long, device=device),
    )


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    context: PagedStepContext = module.paged_context
    store: PagedKVCacheManager = context.store
    layer_idx = module.layer_idx

    new_keys = key.squeeze(0).transpose(0, 1)
    new_values = value.squeeze(0).transpose(0, 1)
    store.scatter_layer_kv(layer_idx, new_keys, new_values, context.scatter_slots)

    scale = scaling if scaling is not None else module.scaling
    backend = _backend()
    if not backend.supports(context, query.shape[2]):
        backend = _FALLBACK_BACKEND

    attn_output = backend.attend(query, store, layer_idx, context, scale, dropout)
    return attn_output, None


def register_paged_attention(model) -> None:
    ALL_ATTENTION_FUNCTIONS.register(PAGED_ATTENTION_NAME, paged_attention_forward)
    for layer in _decoder_layers(model):
        layer.self_attn.paged_context = None
    model.config._attn_implementation = PAGED_ATTENTION_NAME


def run_paged_forward(
    model,
    context: PagedStepContext,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    decoder = _decoder(model)
    for layer in decoder.layers:
        layer.self_attn.paged_context = context

    position_ids = context.position_ids(input_ids.device)
    outputs = model(
        input_ids,
        position_ids=position_ids,
        use_cache=False,
    )
    return outputs.logits


def _decoder(model) -> torch.nn.Module:
    for attribute in ("model", "transformer", "base_model"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    return model


def _decoder_layers(model) -> SequenceABC[torch.nn.Module]:
    return _decoder(model).layers
