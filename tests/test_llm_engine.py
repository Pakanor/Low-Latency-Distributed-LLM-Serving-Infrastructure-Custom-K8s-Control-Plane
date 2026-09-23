import pytest
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import MockModelClient


def test_llm_engine_execution():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(
        kv_cache_manager=manager,
        max_batch_size=2,
        max_num_batched_tokens=32,
        block_size=4,
        allocator=manager.get_allocator()
    )
    
    engine = LLMEngine(
        kv_cache_manager=manager,
        scheduler=scheduler,
        model_client=MockModelClient()
    )

    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=2)
    res = engine.step()

    assert len(seq.output_token_ids) == 1
    assert seq.output_token_ids[0] == 103
    assert seq.block_table is not None


def test_llm_engine_prefill_kv_write():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(
        kv_cache_manager=manager,
        max_batch_size=2,
        max_num_batched_tokens=32,
        block_size=4,
        allocator=manager.get_allocator()
    )
    
    engine = LLMEngine(
        kv_cache_manager=manager, 
        scheduler=scheduler,
        model_client=MockModelClient()
    )

    seq = engine.add_request(prompt_token_ids=[10, 20, 30], max_tokens=2)
    res = engine.step()

    assert len(seq.output_token_ids) == 1
    initial_free = manager.get_allocator().get_num_free_blocks()

    seq2 = engine.add_request(prompt_token_ids=[40, 50], max_tokens=2)
    res = engine.step()

    assert len(seq2.output_token_ids) == 1
