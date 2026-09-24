import pytest
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import MockModelClient


def test_mock_model_generation():
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

    client = MockModelClient()

    engine = LLMEngine(
        kv_cache_manager=kv_mgr,
        scheduler=scheduler,
        model_client=client
    )

    seq = engine.add_request(prompt_token_ids=[1, 2, 3], max_tokens=3)

    res = engine.step()

    assert len(seq.output_token_ids) == 1
    assert seq.output_token_ids[0] == 103 