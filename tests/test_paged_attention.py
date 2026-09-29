import os

import pytest
import torch

from app.memory.manager import PagedKVCacheManager
from app.model.paged_attention import (
    PagedStepContext,
    register_paged_attention,
    run_paged_forward,
)
from app.scheduler.sequence import Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
NUM_LAYERS = 30
NUM_KV_HEADS = 3
HEAD_DIM = 64
BLOCK_SIZE = 8
MAX_NEW_TOKENS = 12


@pytest.fixture(scope="module")
def model_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _make_store(model, total_blocks=64):
    config = model.config
    return PagedKVCacheManager.from_model_config(
        config,
        total_blocks=total_blocks,
        block_size=BLOCK_SIZE,
        dtype=model.dtype,
        device="cpu",
    )


def _generate_paged(model, tokenizer, prompt):
    store = _make_store(model)
    register_paged_attention(model)

    prompt_ids = tokenizer(prompt).input_ids
    seq = Sequence(seq_id=1, prompt_token_ids=prompt_ids, max_tokens=MAX_NEW_TOKENS)
    store.allocate_prefix_blocks(seq)

    context = PagedStepContext(store, [seq], [0], [len(prompt_ids)])
    logits = run_paged_forward(model, context, torch.tensor([prompt_ids], dtype=torch.long))
    next_token = int(logits[0, -1].argmax())

    generated = [next_token]
    for _ in range(MAX_NEW_TOKENS - 1):
        start = seq.block_table.get_tokens_count()
        store.allocate_slot_for_next_token(seq)
        context = PagedStepContext(store, [seq], [start], [1])
        logits = run_paged_forward(model, context, torch.tensor([[next_token]], dtype=torch.long))
        next_token = int(logits[0, -1].argmax())
        generated.append(next_token)

    return generated


def test_paged_generation_matches_hf_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "The capital of France is"

    reference = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    reference_tokens = reference[0, -MAX_NEW_TOKENS:].tolist()

    paged_tokens = _generate_paged(model, tokenizer, prompt)

    assert paged_tokens == reference_tokens


def test_paged_cache_holds_history_for_every_layer(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Counting: one two three"

    total_blocks = 64
    store = _make_store(model, total_blocks=total_blocks)
    register_paged_attention(model)

    prompt_ids = tokenizer(prompt).input_ids
    seq = Sequence(seq_id=1, prompt_token_ids=prompt_ids, max_tokens=4)
    store.allocate_prefix_blocks(seq)

    context = PagedStepContext(store, [seq], [0], [len(prompt_ids)])
    run_paged_forward(model, context, torch.tensor([prompt_ids], dtype=torch.long))

    assert store.key_cache.shape == (
        NUM_LAYERS,
        total_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )

    gather_slots = PagedStepContext(store, [seq], [0], [0]).gather_slots
    assert gather_slots.numel() == 0

    full_context = PagedStepContext(store, [seq], [0], [len(prompt_ids)])
    for layer_idx in (0, NUM_LAYERS // 2, NUM_LAYERS - 1):
        layer_keys = store.gather_layer_keys(layer_idx, full_context.gather_slots)
        assert layer_keys.shape == (len(prompt_ids), NUM_KV_HEADS, HEAD_DIM)
        assert torch.isfinite(layer_keys).all()
        assert layer_keys.abs().sum() > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
