import pytest
from app.memory.manager import PagedKVCacheManager


def test_kv_cache_swapping_lifecycle():
    manager = PagedKVCacheManager(
        total_blocks=4,
        block_size=4,
        num_heads=2,
        head_dim=8,
        total_cpu_blocks=8,
        device="cpu"
    )
    allocator = manager.get_allocator()
    
    b1 = allocator.allocate_block()
    b2 = allocator.allocate_block()
    assert allocator.get_num_free_blocks() == 2
    assert allocator.get_num_free_cpu_blocks() == 8
    
    # 2. Wykonujemy Swap Out dla bloku b1
    manager.swap_out_sequence(seq_id=101, block_table=[b1])
    assert 101 in manager.swapped_seqs
    assert allocator.get_num_free_blocks() == 3 
    assert allocator.get_num_free_cpu_blocks() == 7  

    manager.swap_in_sequence(seq_id=101, block_table=[b1])
    assert 101 not in manager.swapped_seqs
    assert allocator.get_num_free_blocks() == 2
    assert allocator.get_num_free_cpu_blocks() == 8