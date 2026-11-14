import os, psutil, torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence

proc = psutil.Process(os.getpid())
def rss():
    return proc.memory_info().rss

print("rss_start", rss())
# SmolLM-like geometry, 512 MiB budget equiv ~ many blocks; use fixed pool
num_layers, num_heads, head_dim, block_size = 30, 3, 64, 16
total_blocks = 256  # preallocated pool
mgr = PagedKVCacheManager(
    total_blocks=total_blocks, block_size=block_size,
    num_heads=num_heads, head_dim=head_dim,
    total_cpu_blocks=0, dtype=torch.float32, num_layers=num_layers, device="cpu",
)
print("rss_after_pool_alloc", rss())
print("pool_bytes", mgr.key_cache.nelement()*mgr.key_cache.element_size()*2)
print("dump0", mgr.get_allocator().get_num_free_blocks(), mgr.get_allocator().get_total_blocks())

for i, L in enumerate([16, 64, 128, 256], start=1):
    seq = Sequence(seq_id=i, prompt_token_ids=list(range(L)), max_tokens=8)
    mgr.allocate_prefix_blocks(seq)
    keys = torch.randn(num_layers, L, num_heads, head_dim)
    vals = torch.randn(num_layers, L, num_heads, head_dim)
    mgr.write_prefix_kv(seq, keys, vals)
    print(f"after_seq len={L} rss={rss()} free_blocks={mgr.get_allocator().get_num_free_blocks()} tokens={seq.block_table.get_tokens_count()}")
