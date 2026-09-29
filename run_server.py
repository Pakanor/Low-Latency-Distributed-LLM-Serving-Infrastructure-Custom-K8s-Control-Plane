from app.memory.manager import PagedKVCacheManager
from app.memory.strategies import detect_strategy
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.api.server import set_engine, app, load_model
import uvicorn

manager = PagedKVCacheManager(
    total_blocks=10,
    block_size=4,
    num_heads=8,
    head_dim=64,
    num_layers=16,
)
scheduler = Scheduler(
    kv_cache_manager=manager,
    max_batch_size=2,
    max_num_batched_tokens=32,
    block_size=4,
    allocator=manager.get_allocator(),
)

load_model()

from app.api.server import model
engine = LLMEngine(
    kv_cache_manager=manager,
    scheduler=scheduler,
    model=model,
)
set_engine(engine)

print(f"KV cache strategy: {type(manager.strategy).__name__}")
print(f"KV cache device: {manager.device}")
print(f"KV cache swap pool zero-copy: {manager.get_allocator().is_zero_copy()}")

uvicorn.run(app, host="0.0.0.0", port=8000)
