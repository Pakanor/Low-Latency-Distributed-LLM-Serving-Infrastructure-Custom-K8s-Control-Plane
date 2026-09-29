import os

import pytest
import torch

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
