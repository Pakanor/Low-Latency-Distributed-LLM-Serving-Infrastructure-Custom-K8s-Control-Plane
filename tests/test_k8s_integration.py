import pytest
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.model.client import K8sModelClient


def test_live_k8s_model_generation():
   
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

   
    client = K8sModelClient(
        endpoint_url="http://localhost:8000/generate_step",
        model_name="HuggingFaceTB/SmolLM-135M-Instruct"
    )

    engine = LLMEngine(
        kv_cache_manager=kv_mgr,
        scheduler=scheduler,
        model_client=client
    )

    prompt_tokens = client.tokenizer.encode("Hello", add_special_tokens=False)
    seq = engine.add_request(prompt_token_ids=prompt_tokens, max_tokens=3)

    res = engine.step()

    assert len(seq.output_token_ids) == 1
    assert seq.output_token_ids[0] != client.tokenizer.eos_token_id