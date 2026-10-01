import os

import pytest
import torch
import torch.nn.functional as F

import app.model.paged_attention as paged_attention
from app.memory.manager import PagedKVCacheManager
from app.model.attention_backends import (
    NativePagedBackend,
    TorchSdpBackend,
    native_kernel_available,
    select_backend,
)
from app.model.paged_attention import PagedStepContext
from app.scheduler.sequence import Sequence

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 4


@pytest.fixture
def store():
    return PagedKVCacheManager(
        total_blocks=8,
        block_size=BLOCK_SIZE,
        num_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        device="cpu",
    )


def _context(store, prompt_lens):
    sequences = []
    for index, prompt_len in enumerate(prompt_lens):
        seq = Sequence(seq_id=index + 1, prompt_token_ids=list(range(prompt_len)))
        store.allocate_prefix_blocks(seq)
        sequences.append(seq)
    return sequences, PagedStepContext(store, sequences, [0] * len(sequences), list(prompt_lens))


def test_default_backend_is_reference_without_native_kernel(monkeypatch):
    monkeypatch.delenv("LLM_PAGED_BACKEND", raising=False)
    monkeypatch.setattr(
        "app.model.attention_backends.native_kernel_available", lambda: False
    )
    assert isinstance(select_backend(), TorchSdpBackend)


def test_default_backend_prefers_native_when_available(monkeypatch):
    monkeypatch.delenv("LLM_PAGED_BACKEND", raising=False)
    monkeypatch.setattr(
        "app.model.attention_backends.native_kernel_available", lambda: True
    )
    assert isinstance(select_backend(), NativePagedBackend)


def test_explicit_torch_backend_is_always_allowed(monkeypatch):
    monkeypatch.setenv("LLM_PAGED_BACKEND", "torch")
    assert isinstance(select_backend(), TorchSdpBackend)


def test_explicit_native_backend_fails_loudly_without_kernel(monkeypatch):
    monkeypatch.setenv("LLM_PAGED_BACKEND", "native")
    monkeypatch.setattr(
        "app.model.attention_backends.native_kernel_available", lambda: False
    )
    with pytest.raises(RuntimeError):
        select_backend()


def test_unknown_backend_name_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PAGED_BACKEND", "flash")
    with pytest.raises(ValueError):
        select_backend()


def test_hip_alias_maps_to_native(monkeypatch):
    monkeypatch.setenv("LLM_PAGED_BACKEND", "hip")
    monkeypatch.setattr(
        "app.model.attention_backends.native_kernel_available", lambda: True
    )
    assert isinstance(select_backend(), NativePagedBackend)


def test_native_backend_only_supports_decode_shapes(store):
    sequences, prefill = _context(store, [4, 2])
    assert NativePagedBackend().supports(prefill, prefill.num_query_tokens) is False

    _, decode = _context(store, [4, 2])
    decode.start_positions = [4, 2]
    decode.new_token_counts = [1, 1]
    assert NativePagedBackend().supports(decode, 2) is True


def test_reference_backend_supports_every_shape(store):
    _, prefill = _context(store, [4, 2])
    assert TorchSdpBackend().supports(prefill, prefill.num_query_tokens) is True


def test_native_backend_attend_reports_missing_kernel(store, monkeypatch):
    sequences, context = _context(store, [4, 2])
    query = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)
    monkeypatch.setattr(
        "app.model.attention_backends.native_kernel_available", lambda: False
    )
    with pytest.raises(RuntimeError):
        NativePagedBackend().attend(query, store, 0, context, 0.35, 0.0)


def test_dispatch_falls_back_to_reference_for_unsupported_shape(store, monkeypatch):
    sequences, context = _context(store, [4, 2])
    query = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)
    key = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)
    value = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)

    class RecordingBackend(NativePagedBackend):
        def __init__(self):
            self.calls = 0

        def supports(self, context, num_query_tokens):
            return False

        def attend(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("must not be called for unsupported shapes")

    native = RecordingBackend()
    monkeypatch.setattr(paged_attention, "_BACKEND", native)

    class FakeModule:
        layer_idx = 0
        scaling = 0.35
        paged_context = context

    output, weights = paged_attention.paged_attention_forward(
        FakeModule(), query, key, value, None, dropout=0.0, scaling=0.35
    )

    assert native.calls == 0
    assert output.shape == (1, 6, NUM_KV_HEADS, HEAD_DIM)
    assert weights is None


def test_dispatch_routes_supported_shape_to_backend(store, monkeypatch):
    sequences, context = _context(store, [4, 2])
    query = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)
    key = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)
    value = torch.randn(1, NUM_KV_HEADS, 6, HEAD_DIM)

    class RecordingBackend(TorchSdpBackend):
        def __init__(self):
            self.calls = 0

        def attend(self, *args, **kwargs):
            self.calls += 1
            return super().attend(*args, **kwargs)

    recorder = RecordingBackend()
    monkeypatch.setattr(paged_attention, "_BACKEND", recorder)

    class FakeModule:
        layer_idx = 0
        scaling = 0.35
        paged_context = context

    paged_attention.paged_attention_forward(
        FakeModule(), query, key, value, None, dropout=0.0, scaling=0.35
    )

    assert recorder.calls == 1


def test_cu_seqlens_and_context_offsets_for_ragged_batch(store):
    sequences, context = _context(store, [4, 2, 3])
    assert context.cu_seqlens.tolist() == [0, 4, 6, 9]
    assert context.cu_context_offsets.tolist() == [0, 4, 6, 9]
    assert context.context_lens == [4, 2, 3]
    assert context.num_query_tokens == 9
    assert context.num_context_tokens == 9


def test_position_ids_for_prefill_preserves_positions_across_boundaries(store):
    sequences, context = _context(store, [4, 2])
    pos_ids = context.position_ids(torch.device("cpu"))
    assert pos_ids.shape == (1, 6)
    assert pos_ids.tolist() == [[0, 1, 2, 3, 0, 1]]


def test_position_ids_for_decode_step(store):
    sequences = []
    for index, prompt_len in enumerate([4, 2]):
        seq = Sequence(seq_id=index + 1, prompt_token_ids=list(range(prompt_len)))
        store.allocate_prefix_blocks(seq)
        store.allocate_slot_for_next_token(seq)
        sequences.append(seq)

    start_positions = [seq.block_table.get_tokens_count() - 1 for seq in sequences]
    new_token_counts = [1, 1]
    context = PagedStepContext(store, sequences, start_positions, new_token_counts)

    pos_ids = context.position_ids(torch.device("cpu"))
    assert pos_ids.shape == (1, 2)
    assert pos_ids.tolist() == [[4, 2]]


def test_attention_mask_is_block_diagonal_causal(store):
    sequences, context = _context(store, [4, 2])
    mask = context.attention_mask(torch.device("cpu"))
    assert mask.shape == (1, 1, 6, 6)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask[0, 0], expected)


def test_scatter_then_gather_round_trips_kv(store):
    sequences, context = _context(store, [4, 2])
    context_tokens = context.num_context_tokens

    torch.manual_seed(123)
    original_kv = torch.randn(context_tokens, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)

    store.scatter_kv(original_kv, original_kv, context.scatter_slots)

    for layer_idx in range(NUM_LAYERS):
        gathered_keys = store.gather_layer_keys(layer_idx, context.gather_slots)
        assert torch.allclose(gathered_keys, original_kv[:, layer_idx])

        gathered_values = store.gather_layer_values(layer_idx, context.gather_slots)
        assert torch.allclose(gathered_values, original_kv[:, layer_idx])


def test_gqa_reference_backend_attend_shape(store):
    num_query_heads = 6
    sequences, context = _context(store, [4, 2])
    query_tokens = context.num_query_tokens
    context_tokens = context.num_context_tokens

    kv = torch.randn(context_tokens, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, num_query_heads, query_tokens, HEAD_DIM)
    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, 0.35, 0.0)

    assert output.shape == (1, query_tokens, num_query_heads, HEAD_DIM)


def test_reference_backend_matches_manual_sdpa(store):
    sequences, context = _context(store, [4, 2])
    context_tokens = context.num_context_tokens
    query_tokens = context.num_query_tokens

    torch.manual_seed(42)
    kv = torch.randn(context_tokens, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, NUM_KV_HEADS, query_tokens, HEAD_DIM)
    scale = 0.35

    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, scale, 0.0)

    context_keys = store.gather_layer_keys(0, context.gather_slots)
    context_values = store.gather_layer_values(0, context.gather_slots)
    gathered_keys = context_keys.unsqueeze(0).transpose(1, 2)
    gathered_values = context_values.unsqueeze(0).transpose(1, 2)

    manual_output = F.scaled_dot_product_attention(
        query,
        gathered_keys,
        gathered_values,
        attn_mask=context.attention_mask(query.device),
        dropout_p=0.0,
        scale=scale,
        enable_gqa=False,
    )
    manual_output = manual_output.transpose(1, 2).contiguous()

    assert torch.allclose(output, manual_output, atol=1e-6)


def test_gqa_reference_backend_matches_manual_sdpa(store):
    num_query_heads = 6
    sequences, context = _context(store, [4, 2])
    context_tokens = context.num_context_tokens
    query_tokens = context.num_query_tokens

    torch.manual_seed(42)
    kv = torch.randn(context_tokens, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, num_query_heads, query_tokens, HEAD_DIM)
    scale = 0.35

    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, scale, 0.0)

    context_keys = store.gather_layer_keys(0, context.gather_slots)
    context_values = store.gather_layer_values(0, context.gather_slots)
    gathered_keys = context_keys.unsqueeze(0).transpose(1, 2)
    gathered_values = context_values.unsqueeze(0).transpose(1, 2)

    manual_output = F.scaled_dot_product_attention(
        query,
        gathered_keys,
        gathered_values,
        attn_mask=context.attention_mask(query.device),
        dropout_p=0.0,
        scale=scale,
        enable_gqa=gathered_keys.shape[1] != query.shape[1],
    )
    manual_output = manual_output.transpose(1, 2).contiguous()

    assert torch.allclose(output, manual_output, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
