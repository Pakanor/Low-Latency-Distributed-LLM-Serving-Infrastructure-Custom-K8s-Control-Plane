import pytest
import torch
import torch.nn.functional as F

from app.memory.manager import PagedKVCacheManager
from app.model.attention_backends import TorchSdpBackend
from app.model.paged_attention import PagedStepContext
from app.scheduler.sequence import Sequence


def _make_store(total_blocks=16, block_size=4, num_heads=2, head_dim=8, num_layers=1):
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
    )


def _make_context(store, prompt_lens):
    sequences = []
    for idx, plen in enumerate(prompt_lens):
        seq = Sequence(seq_id=idx + 1, prompt_token_ids=list(range(plen)))
        store.allocate_prefix_blocks(seq)
        sequences.append(seq)
    return sequences, PagedStepContext(store, sequences, [0] * len(sequences), list(prompt_lens))


def _manual_attention(query, key, value, mask, scale, enable_gqa):
    if enable_gqa and query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


@pytest.mark.parametrize("num_query_heads,kv_heads,head_dim,scale", [
    (4, 4, 8, 0.35),
    (6, 2, 8, 0.35),   # GQA
    (8, 2, 16, 0.25),  # different head_dim
    (2, 2, 8, 1.0),    # scale=1.0
])
def test_attention_matches_manual_softmax(store_fixture, num_query_heads, kv_heads, head_dim, scale):
    store = store_fixture(kv_heads=kv_heads, head_dim=head_dim)
    sequences, context = _make_context(store, [4, 2, 3])
    context_tokens = context.num_context_tokens
    query_tokens = context.num_query_tokens

    torch.manual_seed(1234)
    kv = torch.randn(context_tokens, 1, kv_heads, head_dim)
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, num_query_heads, query_tokens, head_dim)
    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, scale, 0.0)

    context_keys = store.gather_layer_keys(0, context.gather_slots)
    context_values = store.gather_layer_values(0, context.gather_slots)
    gathered_keys = context_keys.unsqueeze(0).transpose(1, 2)
    gathered_values = context_values.unsqueeze(0).transpose(1, 2)

    manual = _manual_attention(
        query, gathered_keys, gathered_values,
        context.attention_mask(query.device), scale,
        enable_gqa=(gathered_keys.shape[1] != query.shape[1]),
    )
    manual = manual.transpose(1, 2).contiguous()

    assert torch.allclose(output, manual, atol=1e-6, rtol=0)


@pytest.mark.parametrize("num_query_heads,kv_heads,head_dim", [
    (4, 4, 8),
    (6, 2, 8),
])
def test_attention_stable_with_large_and_negative_values(store_fixture, num_query_heads, kv_heads, head_dim):
    store = store_fixture(kv_heads=kv_heads, head_dim=head_dim)
    sequences, context = _make_context(store, [4, 2, 3])
    context_tokens = context.num_context_tokens
    query_tokens = context.num_query_tokens

    torch.manual_seed(777)
    kv = torch.randn(context_tokens, 1, kv_heads, head_dim) * 50.0
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, num_query_heads, query_tokens, head_dim) * 50.0
    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, 0.35, 0.0)

    context_keys = store.gather_layer_keys(0, context.gather_slots)
    context_values = store.gather_layer_values(0, context.gather_slots)
    gathered_keys = context_keys.unsqueeze(0).transpose(1, 2)
    gathered_values = context_values.unsqueeze(0).transpose(1, 2)

    manual = _manual_attention(
        query, gathered_keys, gathered_values,
        context.attention_mask(query.device), 0.35,
        enable_gqa=(gathered_keys.shape[1] != query.shape[1]),
    )
    manual = manual.transpose(1, 2).contiguous()

    assert torch.isfinite(output).all()
    assert torch.allclose(output, manual, atol=1e-4, rtol=0)


@pytest.mark.parametrize("prompt_lens", [
    [1],
    [2, 1],
    [1, 1, 1],
    [5],
    [3, 3, 3],
    [7, 2, 5, 1],
])
def test_attention_various_prompt_lengths(store_fixture, prompt_lens):
    store = store_fixture(kv_heads=2, head_dim=8)
    sequences, context = _make_context(store, prompt_lens)
    context_tokens = context.num_context_tokens
    query_tokens = context.num_query_tokens

    torch.manual_seed(99)
    kv = torch.randn(context_tokens, 1, 2, 8)
    store.scatter_kv(kv, kv, context.scatter_slots)

    query = torch.randn(1, 4, query_tokens, 8)
    backend = TorchSdpBackend()
    output = backend.attend(query, store, 0, context, 0.35, 0.0)

    context_keys = store.gather_layer_keys(0, context.gather_slots)
    context_values = store.gather_layer_values(0, context.gather_slots)
    gathered_keys = context_keys.unsqueeze(0).transpose(1, 2)
    gathered_values = context_values.unsqueeze(0).transpose(1, 2)

    manual = _manual_attention(
        query, gathered_keys, gathered_values,
        context.attention_mask(query.device), 0.35,
        enable_gqa=(gathered_keys.shape[1] != query.shape[1]),
    )
    manual = manual.transpose(1, 2).contiguous()

    assert torch.allclose(output, manual, atol=1e-6, rtol=0)


@pytest.fixture
def store_fixture():
    def _make(kv_heads=2, head_dim=8, total_blocks=16, block_size=4):
        return PagedKVCacheManager(
            total_blocks=total_blocks,
            block_size=block_size,
            num_heads=kv_heads,
            head_dim=head_dim,
            num_layers=1,
            device="cpu",
        )
    return _make