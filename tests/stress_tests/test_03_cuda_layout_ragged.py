import pytest
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine


def test_ragged_batch_block_table_structure():
    
    kv_mgr = PagedKVCacheManager(
        total_blocks=20,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(
        max_batch_size=4,
        max_num_batched_tokens=200,
        block_size=4,
        allocator=kv_mgr.get_allocator()
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler)

    s1 = engine.add_request(prompt_token_ids=[10], max_tokens=2)
    s2 = engine.add_request(prompt_token_ids=list(range(10, 17)), max_tokens=2)
    s3 = engine.add_request(prompt_token_ids=list(range(20, 33)), max_tokens=2)

    engine.step()

    active_seqs = [s1, s2, s3]
    block_tables, context_lens = kv_mgr.build_block_tables(active_seqs)

    assert block_tables.shape == (3, 4)
    assert context_lens.tolist() == [1, 7, 13]

    
    assert block_tables[0, 1:].tolist() == [-1, -1, -1]
    assert block_tables[1, 2:].tolist() == [-1, -1]
    assert -1 not in block_tables[2].tolist()


def test_block_boundary_exact_crossing():
    
    kv_mgr = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(
        max_batch_size=2,
        max_num_batched_tokens=32,
        block_size=4,
        allocator=kv_mgr.get_allocator()
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler)

    seq = engine.add_request(prompt_token_ids=[1, 2, 3, 4], max_tokens=3)

    engine.step()

    block_tables, context_lens = kv_mgr.build_block_tables([seq])
    
    assert context_lens.tolist() == [5]
    assert block_tables.shape == (1, 2)
    assert block_tables[0, 0] != -1
    assert block_tables[0, 1] != -1


def test_empty_prompt_rejection():
    
    kv_mgr = PagedKVCacheManager(
        total_blocks=5,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(
        max_batch_size=2,
        max_num_batched_tokens=16,
        block_size=4,
        allocator=kv_mgr.get_allocator()
    )
    engine = LLMEngine(kv_cache_manager=kv_mgr, scheduler=scheduler)

    seq = engine.add_request(prompt_token_ids=[], max_tokens=5)
    
    res = engine.step()
    
    assert seq.total_len == 1  # 0 prompt + 1 podglad z generated_token po step