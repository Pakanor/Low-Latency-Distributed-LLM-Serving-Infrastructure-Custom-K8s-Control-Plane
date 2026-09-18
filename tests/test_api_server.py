import pytest
from fastapi.testclient import TestClient
from app.api.server import app, set_engine
from app.engine.llm_engine import LLMEngine
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import MockModelClient


def test_async_sse_completion_endpoint():
    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(kv_cache_manager=manager, max_num_seqs=4)
    client = MockModelClient()
    engine = LLMEngine(scheduler=scheduler, model_client=client, kv_cache_manager=manager)

    set_engine(engine)
    test_client = TestClient(app)

    response = test_client.post(
        "/v1/completions",
        json={"prompt": "Hi", "max_tokens": 3}
    )
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]