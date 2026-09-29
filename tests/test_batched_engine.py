import os

import pytest
import torch

from app.engine.llm_engine import LLMEngine
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
BLOCK_SIZE = 8
TOTAL_BLOCKS = 256
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a forest",
    "Water boils at",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _build_engine(model, max_batch_size=3):
    store = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=TOTAL_BLOCKS,
        block_size=BLOCK_SIZE,
        dtype=model.dtype,
        device="cpu",
    )
    scheduler = Scheduler(
        kv_cache_manager=store,
        max_batch_size=max_batch_size,
        max_num_batched_tokens=512,
        block_size=BLOCK_SIZE,
        allocator=store.get_allocator(),
    )
    return LLMEngine(kv_cache_manager=store, scheduler=scheduler, model=model), store


def _generate_reference(model, tokenizer, prompt, max_tokens):
    output = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    return output[0, -max_tokens:].tolist()


def test_batched_prompts_match_single_sequence_generation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    references = [
        _generate_reference(model, tokenizer, prompt, MAX_TOKENS) for prompt in PROMPTS
    ]

    engine, _ = _build_engine(model, max_batch_size=len(PROMPTS))
    sequences = [
        engine.add_request(tokenizer(prompt).input_ids, max_tokens=MAX_TOKENS)
        for prompt in PROMPTS
    ]

    steps = 0
    while not all(seq.is_finished() for seq in sequences) and steps < MAX_TOKENS * 2:
        engine.step()
        steps += 1

    assert all(seq.is_finished() for seq in sequences)
    for expected, seq in zip(references, sequences):
        assert seq.output_token_ids == expected


def test_batched_step_serves_all_prompts_in_one_prefill(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    engine, _ = _build_engine(model, max_batch_size=len(PROMPTS))

    sequences = [
        engine.add_request(tokenizer(prompt).input_ids, max_tokens=MAX_TOKENS)
        for prompt in PROMPTS
    ]

    engine.step()

    for seq in sequences:
        assert len(seq.output_token_ids) == 1
        assert seq.block_table is not None
        assert seq.block_table.get_tokens_count() == len(seq.prompt_token_ids)


def test_sequences_do_not_leak_into_each_other(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    engine_a, _ = _build_engine(model, max_batch_size=1)
    engine_b, _ = _build_engine(model, max_batch_size=1)

    prompt = PROMPTS[1]
    tokens = tokenizer(prompt).input_ids

    alone = engine_a.add_request(tokens, max_tokens=MAX_TOKENS)
    while not alone.is_finished():
        engine_a.step()

    together = [
        engine_b.add_request(tokenizer(other).input_ids, max_tokens=MAX_TOKENS)
        for other in PROMPTS
    ]
    while not all(seq.is_finished() for seq in together):
        engine_b.step()

    target = next(seq for seq in together if seq.prompt_token_ids == tokens)
    assert target.output_token_ids == alone.output_token_ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
