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

def test_build_block_tables():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=16,
        num_heads=4,
        head_dim=64
    )

    table1 = manager.create_sequence_table()
    for _ in range(20):
        table1.append_token(manager.allocator)

    table2 = manager.create_sequence_table()
    for _ in range(5):
        table2.append_token(manager.allocator)

    block_tables, context_lens = manager.build_block_tables([table1, table2])

    assert block_tables.shape == (2, 2) 
    assert context_lens.tolist() == [20, 5]
    
    assert block_tables[0, 0].item() != -1
    assert block_tables[0, 1].item() != -1

    assert block_tables[1, 0].item() != -1
    assert block_tables[1, 1].item() == -1

    print("\n[TEST BUILD_BLOCK_TABLES PASSED]")
    print("Block Tables Tensor:\n", block_tables)
    print("Context Lens Tensor:\n", context_lens)


if __name__ == "__main__":
    test_paged_kv_cache_manager()