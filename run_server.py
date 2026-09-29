import uvicorn

from app.api import server
from app.engine.llm_engine import LLMEngine
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler

TOTAL_BLOCKS = 10
BLOCK_SIZE = 4

if not server.load_model():
    raise SystemExit(f"Fatal: could not load model {server.MODEL_NAME}")

manager = PagedKVCacheManager.from_model_config(
    server.model.config,
    total_blocks=TOTAL_BLOCKS,
    block_size=BLOCK_SIZE,
    dtype=server.model.dtype,
    device=server.model_device,
)
scheduler = Scheduler(
    kv_cache_manager=manager,
    max_batch_size=2,
    max_num_batched_tokens=32,
    block_size=BLOCK_SIZE,
    allocator=manager.get_allocator(),
)
engine = LLMEngine(
    kv_cache_manager=manager,
    scheduler=scheduler,
    model=server.model,
)
server.set_engine(engine)

print(f"KV cache strategy: {type(manager.strategy).__name__}")
print(f"KV cache device: {manager.device}")
print(f"KV cache geometry: {manager.num_layers} layers, {manager.num_heads} kv heads, {manager.head_dim} head dim")
print(f"KV cache swap pool zero-copy: {manager.get_allocator().is_zero_copy()}")

uvicorn.run(app=server.app, host="0.0.0.0", port=8000)
