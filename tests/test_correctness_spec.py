"""Correctness tests for paged KV/attention/scheduler (deterministic, CPU-friendly where possible).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=. venv/bin/python -m pytest tests/test_correctness_spec.py -q
"""
import math
import os

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _has_hf_model():
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        return os.path.isdir(cache)
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _has_hf_model(), reason="HF model cache unavailable (offline test)"
)


@pytest.fixture(scope="function")
def model_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct", dtype=torch.float32
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    return model, tok


def _run_engine(model, prompt_ids, max_tokens, block_size, total_blocks, batch=8):
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    mgr = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
        total_cpu_blocks=max(1, min(16, total_blocks)),
    )
    sch = Scheduler(
        kv_cache_manager=mgr,
        max_batch_size=batch,
        max_num_batched_tokens=4096,
        block_size=block_size,
        allocator=mgr.get_allocator(),
        max_model_len=getattr(model.config, "max_position_embeddings", 2048),
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    seq = eng.add_request(prompt_ids, max_tokens=max_tokens)
    steps = 0
    while eng.has_unfinished_requests() and steps < 5000:
        eng.step()
        steps += 1
    return seq, mgr, sch, eng


@requires_model
def test_output_equivalence_paged_contiguous_hf(model_tok):
    """Paged vs contiguous-style (large block) vs HF generate, greedy.

    Computes HF references before engine runs (engine registers paged attention
    on the model, which breaks HF generate on the same instance).
    """
    model, tok = model_tok
    cases = [
        ("The capital of France is", 12),
        ("Hello world", 8),
        ("In Python, a list is", 10),
    ]
    refs = {}
    for text, n in cases:
        refs[(text, n)] = model.generate(
            tok(text, return_tensors="pt").input_ids, max_new_tokens=n, do_sample=False
        )[0, -n:].tolist()
    for text, n in cases:
        ids = tok(text).input_ids
        paged, *_ = _run_engine(model, ids, n, 8, 64)
        contig, *_ = _run_engine(model, ids, n, 2048, 4)
        assert paged.output_token_ids == refs[(text, n)]
        assert contig.output_token_ids == refs[(text, n)]


@requires_model
@pytest.mark.parametrize("block_size", [1, 4, 16, 32, 64])
def test_block_size_equivalence(model_tok, block_size):
    """Compute HF ref before engine registers paged attention on model."""
    model, tok = model_tok
    ids = tok("The capital of France is").input_ids
    n = 12
    ref = model.generate(
        tok("The capital of France is", return_tensors="pt").input_ids,
        max_new_tokens=n,
        do_sample=False,
    )[0, -n:].tolist()
    tb = max(96, math.ceil((len(ids) + n) / block_size) + 16)
    seq, *_ = _run_engine(model, ids, n, block_size, tb)
    assert seq.output_token_ids == ref


@requires_model
def test_edge_cases_boundaries(model_tok):
    """Edge cases; refs computed before engine mutates model attention.

    Uses tokenizer-produced IDs so inputs are in-distribution for SmolLM.
    """
    model, tok = model_tok
    cases = []
    # prompt length exactly equals block_size
    txt = tok("one two three four five six seven").input_ids
    cases.append((txt[:8] if len(txt) >= 8 else txt + [1] * (8 - len(txt)), 4, 8, 24))
    txt2 = tok("a b c d e f g h i j k l m n o p").input_ids
    cases.append((txt2[:16], 4, 16, 24))
    # 1 token prompt
    cases.append(([tok("Hi").input_ids[0]], 4, 8, 16))
    # prompt+output ends on block boundary (4+4, bs=4)
    txt3 = tok("alpha beta gamma delta").input_ids
    cases.append((txt3[:4], 4, 4, 16))
    # crosses multiple blocks (16+8, bs=8)
    txt4 = tok("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen").input_ids
    cases.append((txt4[:16], 8, 8, 16))
    refs = []
    for ids, n, _bs, _tb in cases:
        refs.append(model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False)[0, -n:].tolist())
    for (ids, n, bs, tb), ref in zip(cases, refs):
        seq, *_ = _run_engine(model, ids, n, bs, tb)
        assert seq.output_token_ids == ref, f"ids={ids} bs={bs}"


def test_max_model_len_rejected():
    """Hard limit: prompt_len + max_tokens must not exceed max_model_len."""
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    mgr = PagedKVCacheManager(
        total_blocks=64, block_size=8, num_heads=2, head_dim=8, device="cpu", num_layers=1
    )
    sch = Scheduler(
        kv_cache_manager=mgr,
        max_batch_size=4,
        max_num_batched_tokens=4096,
        block_size=8,
        allocator=mgr.get_allocator(),
        max_model_len=16,
    )
    with pytest.raises(ValueError, match="max_model_len"):
        sch.add_sequence(__import__("app.scheduler.sequence", fromlist=["Sequence"]).Sequence(1, prompt_token_ids=list(range(10)), max_tokens=10))


@requires_model
def test_no_leaks_after_engine_run(model_tok):
    model, tok = model_tok
    ids = tok("Hello").input_ids
    seq, mgr, _sch, _eng = _run_engine(model, ids, 6, 8, 32)
    assert seq.is_finished()
    assert mgr.get_allocator().get_num_free_blocks() == 32


def test_sequence_isolation_distinct_kv():
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.sequence import Sequence

    mgr = PagedKVCacheManager(
        total_blocks=16, block_size=4, num_heads=2, head_dim=4, device="cpu", num_layers=2
    )
    s1 = Sequence(1, prompt_token_ids=list(range(6)))
    s2 = Sequence(2, prompt_token_ids=list(range(6)))
    mgr.allocate_prefix_blocks(s1)
    mgr.allocate_prefix_blocks(s2)
    mgr.write_prefix_kv(s1, torch.full((2, 6, 2, 4), 7.0), torch.full((2, 6, 2, 4), 8.0))
    mgr.write_prefix_kv(s2, torch.full((2, 6, 2, 4), -3.0), torch.full((2, 6, 2, 4), -4.0))
    kv1 = mgr.gather_kv_cache(s1, read_len=6)
    assert torch.allclose(kv1[0][0], torch.full_like(kv1[0][0], 7.0))
    assert torch.allclose(kv1[0][1], torch.full_like(kv1[0][1], 8.0))
    assert set(s1.block_table.get_physical_blocks()).isdisjoint(
        set(s2.block_table.get_physical_blocks())
    )


@requires_model
def test_preemption_preserves_output_equivalence(model_tok):
    model, tok = model_tok
    ids1 = tok("The capital of France is").input_ids
    ids2 = tok("Hello world").input_ids
    base, *_ = _run_engine(model, ids1, 12, 8, 8, batch=1)
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    mgr = PagedKVCacheManager.from_model_config(
        model.config, total_blocks=2, block_size=8, dtype=torch.float32, device="cpu", total_cpu_blocks=1
    )
    sch = Scheduler(
        kv_cache_manager=mgr, max_batch_size=2, max_num_batched_tokens=64, block_size=8,
        allocator=mgr.get_allocator(), max_model_len=2048,
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    a = eng.add_request(ids1, max_tokens=12)
    b = eng.add_request(ids2, max_tokens=12)
    while eng.has_unfinished_requests():
        eng.step()
    assert sch.preemption_count > 0
    assert a.output_token_ids == base.output_token_ids
    assert b.is_finished()


def test_generate_after_register_paged_attention_fails_clearly():
    """register_paged_attention mutates model globally; HF generate must not silently corrupt.

    Uses a fresh model instance so it does not pollute shared fixtures.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.model.paged_attention import register_paged_attention

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct", dtype=torch.float32
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    _ = model.generate(ids, max_new_tokens=4, do_sample=False)
    register_paged_attention(model)
    with pytest.raises((AttributeError, RuntimeError, ValueError, TypeError)):
        model.generate(ids, max_new_tokens=4, do_sample=False)
