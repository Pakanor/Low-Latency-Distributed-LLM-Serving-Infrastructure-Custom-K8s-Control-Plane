import pytest
from fastapi.testclient import TestClient
from app.api.server import app, set_engine
from app.engine.llm_engine import LLMEngine
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import MockModelClient


def test_async_sse_completion_endpoint():
    import app.api.server as server_module

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]
        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(t) for t in token_ids if 0 <= t < 128)
        @property
        def eos_token_id(self):
            return 0

    server_module.tokenizer = _FakeTokenizer()
    server_module.model = object()

    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )
    scheduler = Scheduler(kv_cache_manager=manager, max_batch_size=4)
    client = MockModelClient()
    engine = LLMEngine(scheduler=scheduler, model_client=client, kv_cache_manager=manager)

    set_engine(engine)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/completions",
            json={"prompt": "Hi", "max_tokens": 3}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
