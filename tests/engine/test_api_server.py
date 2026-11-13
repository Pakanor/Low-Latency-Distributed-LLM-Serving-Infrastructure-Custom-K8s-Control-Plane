import pytest
import time
from fastapi.testclient import TestClient

import app.api.server as server_module
from app.api.server import app, set_engine
from app.engine.llm_engine import LLMEngine
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager
from app.model.client import MockModelClient
from app.scheduler.sequence import SequenceStatus

pytestmark = pytest.mark.engine


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) for t in token_ids if 0 <= t < 128)

    @property
    def eos_token_id(self):
        return 0


def _build_engine():
    server_module.tokenizer = _FakeTokenizer()
    server_module.model = object()

    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=16,
    )
    scheduler = Scheduler(kv_cache_manager=manager, max_batch_size=4)
    client = MockModelClient()
    engine = LLMEngine(scheduler=scheduler, model_client=client, kv_cache_manager=manager)
    set_engine(engine)
    return engine, manager


def test_async_sse_completion_endpoint():
    _build_engine()
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/completions",
            json={"prompt": "Hi", "max_tokens": 3},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


def test_completions_rejects_when_kv_full():
    engine, manager = _build_engine()
    # Replace the scheduler's allocator with a 1-block pool that is already
    # occupied, so is_full() returns True regardless of the real manager.
    import llm_allocator_cpp

    tiny = llm_allocator_cpp.PageAllocator(1, 4, 16, 0)
    tiny.allocate_block()
    engine.scheduler.allocator = tiny
    assert engine.scheduler.is_full() is True

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/completions",
            json={"prompt": "Hi", "max_tokens": 3},
        )
        assert response.status_code == 503
        assert "full" in response.json()["detail"].lower()


def test_generate_rejects_when_kv_full():
    engine, manager = _build_engine()
    import llm_allocator_cpp

    tiny = llm_allocator_cpp.PageAllocator(1, 4, 16, 0)
    tiny.allocate_block()
    engine.scheduler.allocator = tiny
    assert engine.scheduler.is_full() is True

    with TestClient(app) as test_client:
        response = test_client.post(
            "/generate",
            json={"prompt": "Hi", "max_tokens": 3},
        )
        assert response.status_code == 503
        assert "full" in response.json()["detail"].lower()


def test_generate_rejects_sequence_over_model_limit():
    _build_engine()

    with TestClient(app) as test_client:
        response = test_client.post(
            "/generate",
            json={"prompt": "a" * 2048, "max_tokens": 1},
        )

        assert response.status_code == 400
        assert "max_model_len" in response.json()["detail"]


def test_deadline_abandon_frees_kv_blocks():
    engine, manager = _build_engine()
    past = time.monotonic() - 10
    seq = engine.add_request(
        prompt_token_ids=[1, 2, 3], max_tokens=5, deadline=past
    )
    engine.step()
    assert seq.block_table is not None

    freed = engine.sweep_abandoned()
    assert seq in freed
    assert seq.status == SequenceStatus.FINISHED
    assert seq.block_table is None