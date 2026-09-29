import pytest
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import MockModelClient
from app.scheduler.sequence import SequenceStatus


@pytest.fixture
def constrained_engine():
    
    kv_mgr = PagedKVCacheManager(
        total_blocks=4,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )
    scheduler = Scheduler(
        kv_cache_manager=kv_mgr,
        max_batch_size=4,
        max_num_batched_tokens=32,
        block_size=4,
    )
    return LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler, model_client=MockModelClient(num_layers=1))


def test_prompt_exceeds_total_memory(constrained_engine):
    
    allocator = constrained_engine.kv_cache_manager.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    big_prompt = list(range(20))
    seq = constrained_engine.add_request(prompt_token_ids=big_prompt, max_tokens=5)

    constrained_engine.step()

    assert seq.status == SequenceStatus.WAITING
    assert allocator.get_num_free_blocks() == initial_free


def test_decode_oom_preemption_and_recovery(constrained_engine):
    
    allocator = constrained_engine.kv_cache_manager.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    s1 = constrained_engine.add_request(prompt_token_ids=[1, 2, 3, 4], max_tokens=2)
    s2 = constrained_engine.add_request(prompt_token_ids=[5, 6, 7, 8], max_tokens=2)

    constrained_engine.step()
    assert allocator.get_num_free_blocks() == 2

    for _ in range(10):
        if not constrained_engine.has_unfinished_requests():
            break
        constrained_engine.step()

    assert allocator.get_num_free_blocks() == initial_free
