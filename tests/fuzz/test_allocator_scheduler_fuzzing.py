"""Fuzzing allocator and scheduler state transitions.

Covers: scaled fuzz (100 seeds x 1000 steps, seed on failure), hypothesis
RuleBasedStateMachine for allocator invariants, preemption+resume ops via
engine/scheduler (add_sequence, step, preempt/re-admit), parameterized
preemption equivalence under pressure.
"""
import os
import random
import gc
from collections import deque

import pytest
import psutil
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from hypothesis import settings, HealthCheck
from hypothesis.stateful import (
    RuleBasedStateMachine,
    Bundle,
    initialize,
    invariant,
    multiple,
    rule,
    precondition,
)
import hypothesis.strategies as st

from tests.helpers import check_allocator_invariants


pytestmark = pytest.mark.fuzz


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


class AllocatorStateMachine(RuleBasedStateMachine):
    """Hypothesis state machine: random alloc/append/free with per-op invariants."""

    live = Bundle("live")

    def __init__(self):
        super().__init__()
        from app.memory.manager import PagedKVCacheManager

        self.total_blocks = 24
        self.block_size = 4
        self.mgr = PagedKVCacheManager(
            total_blocks=self.total_blocks,
            block_size=self.block_size,
            num_heads=1,
            head_dim=4,
            device="cpu",
            num_layers=1,
        )
        self.alloc = self.mgr.get_allocator()
        self._live = {}  # sid -> (seq, tokens_count)
        self._next_sid = 1

    @rule(target=live, prompt_len=st.integers(min_value=1, max_value=10))
    @precondition(lambda self: len(self._live) < 8)
    def add_sequence(self, prompt_len):
        from app.scheduler.sequence import Sequence

        sid = self._next_sid
        self._next_sid += 1
        s = Sequence(sid, prompt_token_ids=list(range(prompt_len)))
        try:
            self.mgr.allocate_prefix_blocks(s)
        except Exception:
            if s.block_table is not None:
                self.mgr.free_sequence(s)
            return multiple()
        self._live[sid] = (s, prompt_len)
        return sid

    @rule(sid=live)
    def append_token(self, sid):
        if sid not in self._live:
            return
        s, n = self._live[sid]
        try:
            self.mgr.allocate_slot_for_next_token(s)
            self._live[sid] = (s, n + 1)
        except Exception:
            pass

    @rule(sid=live)
    def free_sequence(self, sid):
        if sid not in self._live:
            return
        s, _n = self._live.pop(sid)
        self.mgr.free_sequence(s)

    @invariant()
    def inv_allocator(self):
        check_allocator_invariants(
            self.alloc,
            [seq for seq, _n in self._live.values()],
            self.total_blocks,
            self.block_size,
        )

    def teardown(self):
        for s, _n in list(self._live.values()):
            self.mgr.free_sequence(s)
        self._live.clear()
        assert self.alloc.get_num_free_blocks() == self.total_blocks


TestAllocatorStateMachine = AllocatorStateMachine.TestCase
TestAllocatorStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class SchedulerPreemptionStateMachine(RuleBasedStateMachine):
    """Hypothesis state machine over engine ops: add/step with small pool.

    Uses MockModelClient (no HF model). Checks per-step invariants and that
    finished sequences release all blocks.
    """

    def __init__(self):
        super().__init__()
        from app.engine.llm_engine import LLMEngine
        from app.memory.manager import PagedKVCacheManager
        from app.model.client import MockModelClient
        from app.scheduler.scheduler import Scheduler

        self.total_blocks = 10
        self.block_size = 4
        self.mgr = PagedKVCacheManager(
            total_blocks=self.total_blocks,
            block_size=self.block_size,
            num_heads=1,
            head_dim=4,
            device="cpu",
            num_layers=1,
        )
        self.sch = Scheduler(
            kv_cache_manager=self.mgr,
            max_batch_size=4,
            max_num_batched_tokens=64,
            block_size=self.block_size,
            allocator=self.mgr.get_allocator(),
            max_model_len=128,
        )
        self.eng = LLMEngine(
            kv_cache_manager=self.mgr,
            scheduler=self.sch,
            model_client=MockModelClient(num_layers=1, num_heads=1, head_dim=4),
        )
        self.seqs = []

    @rule(
        prompt_len=st.integers(min_value=1, max_value=8),
        max_tokens=st.integers(min_value=1, max_value=6),
    )
    @precondition(lambda self: len([s for s in self.seqs if not s.is_finished()]) < 5)
    def add(self, prompt_len, max_tokens):
        ids = list(range(1, prompt_len + 1))
        try:
            s = self.eng.add_request(ids, max_tokens=max_tokens)
            self.seqs.append(s)
        except Exception:
            pass

    @rule(steps=st.integers(min_value=1, max_value=5))
    @precondition(lambda self: self.eng.has_unfinished_requests())
    def step(self, steps):
        for _ in range(steps):
            if not self.eng.has_unfinished_requests():
                break
            self.eng.step()
            self._check_invariants()

    @invariant()
    def inv_allocator(self):
        self._check_invariants()

    def _check_invariants(self):
        check_allocator_invariants(
            self.mgr.get_allocator(),
            self.seqs,
            self.total_blocks,
            self.block_size,
        )

    def teardown(self):
        for _ in range(500):
            if not self.eng.has_unfinished_requests():
                break
            self.eng.step()
        for seq in self.seqs:
            if seq.block_table is not None:
                self.mgr.free_sequence(seq)
        assert self.mgr.get_allocator().get_num_free_blocks() == self.total_blocks


TestSchedulerStateMachine = SchedulerPreemptionStateMachine.TestCase
TestSchedulerStateMachine.settings = settings(
    max_examples=30,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def test_allocator_fuzz_scaled_invariants():
    """100 seeds x 1000 steps; failure reports reproducible seed."""
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.sequence import Sequence

    def run_one(seed):
        rng = random.Random(seed)
        total_blocks = 24
        block_size = 4
        mgr = PagedKVCacheManager(
            total_blocks=total_blocks, block_size=block_size, num_heads=1, head_dim=4,
            device="cpu", num_layers=1,
        )
        alloc = mgr.get_allocator()
        live = {}
        recent_ops = deque(maxlen=20)

        def check():
            check_allocator_invariants(
                alloc,
                [seq for seq, _n in live.values()],
                total_blocks,
                block_size,
            )

        for i in range(1000):
            op = rng.choice(["add", "append", "free", "add", "append"])
            recent_ops.append(op)
            check()
            if op == "add" and len(live) < 6 and alloc.get_num_free_blocks() > 0:
                plen = rng.randint(1, 10)
                s = Sequence(1000 + i, prompt_token_ids=list(range(plen)))
                try:
                    mgr.allocate_prefix_blocks(s)
                    live[s.seq_id] = (s, plen)
                except Exception:
                    if s.block_table is not None:
                        mgr.free_sequence(s)
            elif op == "append" and live:
                sid = rng.choice(list(live))
                s, n = live[sid]
                try:
                    mgr.allocate_slot_for_next_token(s)
                    live[sid] = (s, n + 1)
                except Exception:
                    pass
            elif op == "free" and live:
                sid = rng.choice(list(live))
                s, _n = live.pop(sid)
                mgr.free_sequence(s)
        try:
            for s, _n in list(live.values()):
                mgr.free_sequence(s)
            live.clear()
            check()
            assert alloc.get_num_free_blocks() == total_blocks
        except AssertionError as e:
            raise AssertionError(f"seed={seed}, recent_ops={list(recent_ops)}: {e}") from e
        finally:
            del live, alloc, mgr
            gc.collect()

    process = psutil.Process()
    first_rss = None
    for seed in range(100):
        try:
            run_one(20000 + seed)
        except AssertionError as e:
            raise AssertionError(f"scaled fuzz failed seed={20000+seed}: {e}") from e
        rss = process.memory_info().rss
        if first_rss is None:
            first_rss = rss
    assert first_rss is not None
    assert rss <= first_rss + 300 * 1024 * 1024


def _baseline_tokens(model, prompt_ids, max_tokens):
    return model.generate(
        torch.tensor([prompt_ids]), max_new_tokens=max_tokens, do_sample=False
    )[0, -max_tokens:].tolist()


def _tiny_gqa_model(vocab_size):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _run_engine(model, prompt_specs, total_blocks, block_size, max_model_len=2048):
    """Run multiple requests through engine; prompt_specs=[(ids, max_tokens)]."""
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    mgr = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
        total_cpu_blocks=max(1, min(8, total_blocks)),
    )
    sch = Scheduler(
        kv_cache_manager=mgr,
        max_batch_size=len(prompt_specs),
        max_num_batched_tokens=4096,
        block_size=block_size,
        allocator=mgr.get_allocator(),
        max_model_len=max_model_len,
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    seqs = [eng.add_request(ids, max_tokens=n) for ids, n in prompt_specs]
    max_steps = 20 * sum(len(ids) + n for ids, n in prompt_specs)
    for _ in range(max_steps):
        if not eng.has_unfinished_requests():
            break
        eng.step()
    else:
        raise AssertionError(
            f"engine did not finish within {max_steps} steps; livelock? "
            f"total_blocks={total_blocks}, block_size={block_size}"
        )
    return seqs, mgr, sch


def test_scheduler_rejects_request_larger_than_pool():
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler
    from app.scheduler.sequence import Sequence

    mgr = PagedKVCacheManager(
        total_blocks=3,
        block_size=4,
        num_heads=1,
        head_dim=4,
        device="cpu",
        num_layers=1,
    )
    scheduler = Scheduler(
        kv_cache_manager=mgr,
        block_size=4,
        allocator=mgr.get_allocator(),
        max_model_len=128,
    )

    with pytest.raises(ValueError, match="requires 4 blocks.*only 3 blocks"):
        scheduler.add_sequence(Sequence(1, prompt_token_ids=list(range(5)), max_tokens=10))


@pytest.mark.hf
@requires_model
def test_scheduler_preemption_hf_equivalence():
    """HF preemption equivalence: multiple pool/block configs, repeated preemption,
    random lengths. All baselines computed before engine registers paged attention.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct", dtype=torch.float32
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

    ids1 = tok("The capital of France is").input_ids
    ids2 = tok("Hello world").input_ids
    ids_b = tok("A short test").input_ids
    pool = [tok(t).input_ids for t in [
        "The capital of France is", "Hello world", "In Python, a list is",
        "Water boils at", "The sun is",
    ]]

    # All baselines before any engine/paged-attention registration.
    bases = {
        ("p1", 10): _baseline_tokens(model, ids1, 10),
        ("p2", 8): _baseline_tokens(model, ids2, 8),
        ("p1", 14): _baseline_tokens(model, ids1, 14),
        ("pb", 10): _baseline_tokens(model, ids_b, 10),
    }
    rng_specs = []
    rng_bases = []
    for seed in [30001, 30002]:
        rng = random.Random(seed)
        specs = []
        for _ in range(3):
            p = list(rng.choice(pool))
            p = p[: max(1, rng.randint(1, min(len(p), 6)))]
            n = rng.randint(4, 8)
            specs.append((p, n))
        rng_specs.append((seed, specs))

    # Keep the cached SmolLM comparison above and use a tiny random GQA model
    # for the 100-seed-style randomized pressure loop below.
    fuzz_model = _tiny_gqa_model(tok.vocab_size)
    rng_bases = [
        [_baseline_tokens(fuzz_model, p, n) for p, n in specs]
        for _seed, specs in rng_specs
    ]

    # (1) parameterized pool/block configs
    # Each request needs 4/3 blocks for block_size=4, or 2/2 for block_size=8.
    # Every pool fits either request alone but not both, forcing preemption.
    for total_blocks, block_size in [(4, 4), (5, 4), (3, 8)]:
        seqs, mgr, sch = _run_engine(
            model, [(ids1, 10), (ids2, 8)], total_blocks, block_size
        )
        assert seqs[0].is_finished() and seqs[1].is_finished()
        assert seqs[0].output_token_ids == bases[("p1", 10)]
        assert seqs[1].output_token_ids == bases[("p2", 8)]
        assert mgr.get_allocator().get_num_free_blocks() == total_blocks
        del seqs, sch, mgr
        gc.collect()

    # (2) repeated preemption of same sequences under extreme pressure
    mgr = PagedKVCacheManager.from_model_config(
        model.config, total_blocks=10, block_size=2, dtype=torch.float32,
        device="cpu", total_cpu_blocks=1,
    )
    sch = Scheduler(
        kv_cache_manager=mgr, max_batch_size=2, max_num_batched_tokens=128,
        block_size=2, allocator=mgr.get_allocator(), max_model_len=2048,
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    a = eng.add_request(ids1, max_tokens=14)
    b = eng.add_request(ids_b, max_tokens=10)
    c = eng.add_request(ids2, max_tokens=8)
    d = eng.add_request(ids2, max_tokens=8)
    e = eng.add_request(ids2, max_tokens=8)
    f = eng.add_request(ids2, max_tokens=8)
    for _ in range(12000):
        if not eng.has_unfinished_requests():
            break
        eng.step()
        free = mgr.get_allocator().get_num_free_blocks()
        used = set()
        for sq in (a, b, c, d, e, f):
            if sq.block_table is None:
                continue
            phys = list(sq.block_table.get_physical_blocks())
            assert not (used & set(phys))
            used.update(phys)
            assert sq.block_table.get_tokens_count() <= len(phys) * 2
        assert free + len(used) == 10
    assert a.is_finished() and b.is_finished()
    assert c.is_finished() and d.is_finished() and e.is_finished() and f.is_finished()
    assert sch.preemption_count >= 1
    assert max(sch.preemption_counts.values()) >= 2
    assert a.output_token_ids == bases[("p1", 14)]
    assert b.output_token_ids == bases[("pb", 10)]
    assert c.output_token_ids == bases[("p2", 8)]
    assert d.output_token_ids == bases[("p2", 8)]
    assert e.output_token_ids == bases[("p2", 8)]
    assert f.output_token_ids == bases[("p2", 8)]
    assert mgr.get_allocator().get_num_free_blocks() == 10
    del a, b, c, d, e, f, eng, sch, mgr
    gc.collect()

    # (3) random lengths/seeds under pressure
    for (seed, specs), blist in zip(rng_specs, rng_bases):
        seqs, mgr, sch = _run_engine(fuzz_model, specs, total_blocks=5, block_size=4)
        try:
            for s, base, (p, n) in zip(seqs, blist, specs):
                assert s.is_finished(), f"seed={seed} unfinished prompt_len={len(p)} n={n}"
                assert s.output_token_ids == base, f"seed={seed} mismatch"
            assert mgr.get_allocator().get_num_free_blocks() == 5
        finally:
            del seqs, sch, mgr
            gc.collect()
    del fuzz_model
    gc.collect()


def test_scheduler_fuzz_scaled_invariants():
    """100 seeds of random add/step/free patterns with MockModelClient; invariants+no leak."""
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.model.client import MockModelClient
    from app.scheduler.scheduler import Scheduler
    from app.scheduler.sequence import SequenceStatus

    def run_one(seed):
        rng = random.Random(seed)
        total_blocks = 12
        block_size = 4
        mgr = PagedKVCacheManager(
            total_blocks=total_blocks, block_size=block_size, num_heads=1, head_dim=4,
            device="cpu", num_layers=1,
        )
        sch = Scheduler(
            kv_cache_manager=mgr, max_batch_size=4, max_num_batched_tokens=64,
            block_size=block_size, allocator=mgr.get_allocator(), max_model_len=128,
        )
        eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model_client=MockModelClient(num_layers=1, num_heads=1, head_dim=4))
        seqs = []
        next_id = 1
        recent_ops = deque(maxlen=20)

        def check():
            check_allocator_invariants(
                mgr.get_allocator(),
                seqs,
                total_blocks,
                block_size,
            )

        for _ in range(200):
            op = rng.choice(["add", "step", "step", "add", "step"])
            recent_ops.append(op)
            check()
            if op == "add" and len([s for s in seqs if not s.is_finished()]) < 5:
                plen = rng.randint(1, 8)
                mxt = rng.randint(1, 6)
                try:
                    seqs.append(eng.add_request(list(range(1, plen + 1)), max_tokens=mxt))
                except Exception:
                    pass
            elif op == "step" and eng.has_unfinished_requests():
                eng.step()

        for _ in range(2000):
            if not eng.has_unfinished_requests():
                break
            eng.step()
            check()
        check()
        # sequences may be waiting if pool too small for prompt — drain by freeing
        try:
            for s in seqs:
                if not s.is_finished() and s.block_table is not None:
                    sch.free_sequence(s)
            assert mgr.get_allocator().get_num_free_blocks() == total_blocks
        except AssertionError as e:
            raise AssertionError(f"seed={seed}, recent_ops={list(recent_ops)}: {e}") from e
        finally:
            del seqs, eng, sch, mgr
            gc.collect()

    process = psutil.Process()
    first_rss = None
    for seed in range(100):
        try:
            run_one(40000 + seed)
        except AssertionError as e:
            raise AssertionError(f"scheduler fuzz failed seed={40000+seed}: {e}") from e
        rss = process.memory_info().rss
        if first_rss is None:
            first_rss = rss
    assert first_rss is not None
    assert rss <= first_rss + 300 * 1024 * 1024
