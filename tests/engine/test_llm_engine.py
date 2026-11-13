import pytest
import time

from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import MockModelClient
from app.scheduler.sequence import SequenceStatus

pytestmark = pytest.mark.engine


def _make_engine(total_blocks=10, max_batch_size=2, max_tokens=2):
    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=16,
    )
    scheduler = Scheduler(
        kv_cache_manager=manager,
        max_batch_size=max_batch_size,
        max_num_batched_tokens=32,
        block_size=4,
        allocator=manager.get_allocator(),
    )
    engine = LLMEngine(
        kv_cache_manager=manager,
        scheduler=scheduler,
        model_client=MockModelClient(),
    )
    return engine, manager


def test_llm_engine_execution():
    engine, manager = _make_engine()
    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=2)
    engine.step()

    assert len(seq.output_token_ids) == 1
    assert seq.output_token_ids[0] == 103
    assert seq.block_table is not None


def test_llm_engine_prefill_kv_write():
    engine, manager = _make_engine()
    seq = engine.add_request(prompt_token_ids=[10, 20, 30], max_tokens=2)
    engine.step()

    assert len(seq.output_token_ids) == 1
    initial_free = manager.get_allocator().get_num_free_blocks()

    seq2 = engine.add_request(prompt_token_ids=[40, 50], max_tokens=2)
    engine.step()

    assert len(seq2.output_token_ids) == 1


def test_sweep_abandoned_frees_deadline_sequence():
    engine, manager = _make_engine()
    past = time.monotonic() - 10
    seq = engine.add_request(
        prompt_token_ids=[1, 2, 3], max_tokens=5, deadline=past
    )
    engine.step()
    assert seq.block_table is not None
    assert seq.status == SequenceStatus.RUNNING

    freed = engine.sweep_abandoned()
    assert seq in freed
    assert seq.status == SequenceStatus.FINISHED
    assert seq.block_table is None


def test_sweep_abandoned_frees_client_disconnect():
    engine, manager = _make_engine()
    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=5)
    engine.step()
    assert seq.block_table is not None

    seq.abandon_requested = True
    freed = engine.sweep_abandoned()
    assert seq in freed
    assert seq.status == SequenceStatus.FINISHED


def test_sweep_abandoned_leaves_active_sequences_alone():
    engine, manager = _make_engine()
    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=5)
    engine.step()
    assert seq.block_table is not None

    freed = engine.sweep_abandoned()
    assert freed == []
    assert seq.status == SequenceStatus.RUNNING


def test_scheduler_is_full_when_no_free_blocks():
    engine, manager = _make_engine(total_blocks=2, max_batch_size=4)
    # Each prompt is 3 tokens; with max_tokens=2 the total is 5 tokens -> 2
    # blocks needed (block_size=4), which exactly fills the 2-block pool.
    # max_tokens must be >= 2 so neither sequence finishes after the first step
    # and frees its block before the second is scheduled.
    seq1 = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=2)
    seq2 = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=2)
    assert engine.scheduler.is_full() is False
    engine.step()
    assert engine.scheduler.is_full() is True