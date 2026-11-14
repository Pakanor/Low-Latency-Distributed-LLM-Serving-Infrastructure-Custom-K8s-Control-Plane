import time
import tracemalloc
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence

# simulate manager like SmolLM: 30 layers, 3 kv heads, 64 head dim, block_size 16, fp32
num_layers, num_heads, head_dim, block_size = 30, 3, 64, 16

def run(length, warmup=2, iters=10):
    blocks_needed = (length + block_size - 1) // block_size
    mgr = PagedKVCacheManager(
        total_blocks=blocks_needed + 8,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        total_cpu_blocks=0,
        dtype=torch.float32,
        num_layers=num_layers,
        device="cpu",
    )
    seq = Sequence(seq_id=1, prompt_token_ids=list(range(length)), max_tokens=1)
    mgr.allocate_prefix_blocks(seq)
    # fill with data
    keys = torch.randn(num_layers, length, num_heads, head_dim)
    vals = torch.randn(num_layers, length, num_heads, head_dim)
    mgr.write_prefix_kv(seq, keys, vals)
    for _ in range(warmup):
        mgr.gather_kv_cache(seq, read_len=length)
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = mgr.gather_kv_cache(seq, read_len=length)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # keep ref to out
    _ = out
    pool_bytes = mgr.key_cache.nelement() * mgr.key_cache.element_size() * 2
    print(f"len={length} blocks={blocks_needed} avg_ms={(t1-t0)/iters*1000:.4f} peak_py_alloc_bytes={peak} pool_bytes={pool_bytes}")

for L in (128, 512, 2048):
    run(L)
