import argparse
import os

import uvicorn

from app.api import server
from app.config import DEFAULT_CONFIG, RuntimeConfig, calculate_kv_blocks
from app.engine.llm_engine import LLMEngine
from app.memory.manager import PagedKVCacheManager
from app.memory.strategies import resolve_dtype
from app.scheduler.scheduler import Scheduler

parser = argparse.ArgumentParser(description="Run the LLM API server")
parser.add_argument("--device", default=DEFAULT_CONFIG.device)
parser.add_argument("--num-threads", type=int, default=DEFAULT_CONFIG.num_threads)
parser.add_argument("--max-batch-size", type=int, default=DEFAULT_CONFIG.max_batch_size)
parser.add_argument(
    "--max-num-batched-tokens", type=int, default=None
)
args = parser.parse_args()
runtime_config = RuntimeConfig(
    model_name=server.MODEL_NAME,
    dtype=os.getenv("LLM_DTYPE", DEFAULT_CONFIG.dtype),
    device=args.device,
    budget_mib=DEFAULT_CONFIG.budget_mib,
    max_model_len=DEFAULT_CONFIG.max_model_len,
    block_size=DEFAULT_CONFIG.block_size,
    num_threads=args.num_threads,
    seed=DEFAULT_CONFIG.seed,
    max_batch_size=args.max_batch_size,
    max_num_batched_tokens=args.max_num_batched_tokens,
)

if not server.load_model(runtime_config):
    raise SystemExit(f"Fatal: could not load model {server.MODEL_NAME}")

total_blocks = calculate_kv_blocks(
    server.model.config,
    resolve_dtype(runtime_config.dtype),
    runtime_config.budget_mib,
    runtime_config.block_size,
)
manager = PagedKVCacheManager.from_model_config(
    server.model.config,
    total_blocks=total_blocks,
    block_size=runtime_config.block_size,
    dtype=resolve_dtype(runtime_config.dtype),
    device=server.model_device,
)
scheduler = Scheduler(
    kv_cache_manager=manager,
    max_batch_size=runtime_config.max_batch_size,
    max_num_batched_tokens=runtime_config.max_num_batched_tokens or runtime_config.max_model_len,
    block_size=runtime_config.block_size,
    allocator=manager.get_allocator(),
    max_model_len=runtime_config.max_model_len,
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

uvicorn.run(app=server.app, host="0.0.0.0", port=8000)
