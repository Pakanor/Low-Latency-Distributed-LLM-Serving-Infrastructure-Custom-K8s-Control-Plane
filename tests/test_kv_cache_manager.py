import torch
from app.memory.manager import PagedKVCacheManager

def test_paged_kv_cache_manager():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=16,
        num_heads=4,
        head_dim=64
    )
    
    seq_table = manager.create_sequence_table()
    
    for _ in range(20):
        fake_k = torch.randn(4, 64)
        fake_v = torch.randn(4, 64)
        manager.write_kv_slot(seq_table, fake_k, fake_v)
        
    blocks = seq_table.get_physical_blocks()
    assert len(blocks) == 2, f"Oczekiwano 2 bloków, otrzymano {len(blocks)}"
    assert manager.allocator.get_num_free_blocks() == 8
    
    manager.free_sequence(seq_table)
    assert manager.allocator.get_num_free_blocks() == 10

if __name__ == "__main__":
    test_paged_kv_cache_manager()