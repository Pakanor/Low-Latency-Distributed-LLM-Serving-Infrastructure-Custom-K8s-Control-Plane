import gc
import pytest
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.scheduler.sequence import SequenceStatus


def test_double_free_protection():
   
    kv_mgr = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    allocator = kv_mgr.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    scheduler = Scheduler(
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
        allocator=allocator
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler)

    seq = engine.add_request(prompt_token_ids=[1, 2, 3, 4], max_tokens=1)
    engine.step()  

    assert allocator.get_num_free_blocks() == initial_free - 1

    if seq.block_table is not None:
        seq.block_table.release()
        seq.block_table.release()

    assert allocator.get_num_free_blocks() == initial_free


def test_allocator_lifetime_with_garbage_collection():
    
    kv_mgr = PagedKVCacheManager(
        total_blocks=5,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    allocator = kv_mgr.get_allocator()
    scheduler = Scheduler(
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
        allocator=allocator
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler)

    seq = engine.add_request(prompt_token_ids=[10, 20, 30, 40], max_tokens=5)
    engine.step()  

    block_table_ref = seq.block_table

    del engine
    del kv_mgr
    del scheduler
    gc.collect()

    assert block_table_ref is not None
    assert block_table_ref.get_num_blocks() > 0
    
    block_table_ref.release()

