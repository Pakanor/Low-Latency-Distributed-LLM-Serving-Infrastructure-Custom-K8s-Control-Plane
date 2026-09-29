import pytest
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import MockModelClient
from app.scheduler.sequence import SequenceStatus


def test_immediate_finish_max_tokens_one():

    kv_mgr = PagedKVCacheManager(
        total_blocks=5,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )
    scheduler = Scheduler(
        kv_cache_manager=kv_mgr,
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler, model_client=MockModelClient(num_layers=1))

    seq = engine.add_request(prompt_token_ids=[10, 20], max_tokens=1)

    res = engine.step()

    assert seq.status == SequenceStatus.FINISHED
    assert seq in res["finished"]
    assert len(seq.output_token_ids) == 1


def test_client_abort_mid_generation():

    kv_mgr = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )
    allocator = kv_mgr.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    scheduler = Scheduler(
        kv_cache_manager=kv_mgr,
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler, model_client=MockModelClient(num_layers=1))

    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=10)
    engine.step()

    assert allocator.get_num_free_blocks() == initial_free - 1

    seq.status = SequenceStatus.FINISHED
    if seq.block_table is not None:
        seq.block_table.release(kv_mgr.get_allocator())
        seq.block_table = None

    res = engine.step()

    assert seq not in res["running"]
    assert allocator.get_num_free_blocks() == initial_free


def test_prompt_larger_than_total_allocator_capacity():

    kv_mgr = PagedKVCacheManager(
        total_blocks=2,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )
    scheduler = Scheduler(
        kv_cache_manager=kv_mgr,
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler, model_client=MockModelClient(num_layers=1))

    seq = engine.add_request(prompt_token_ids=list(range(12)), max_tokens=5)

    engine.step()

    assert seq.status == SequenceStatus.WAITING
    assert kv_mgr.get_allocator().get_num_free_blocks() == 2
