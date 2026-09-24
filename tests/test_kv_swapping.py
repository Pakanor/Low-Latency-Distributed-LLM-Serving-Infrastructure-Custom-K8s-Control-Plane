import pytest
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence


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

    seq = Sequence(seq_id=101, prompt_token_ids=[1, 2, 3, 4])
    manager.allocate_prefix_blocks(seq)

    b1 = seq.block_table.get_physical_blocks()[0]
    assert allocator.get_num_free_blocks() == 3
    assert allocator.get_num_free_cpu_blocks() == 8

    manager.swap_out_sequence(seq)
    assert 101 in manager.swapped_seqs
    assert allocator.get_num_free_blocks() == 4
    assert allocator.get_num_free_cpu_blocks() == 7

    manager.swap_in_sequence(seq)
    assert 101 not in manager.swapped_seqs
    assert allocator.get_num_free_blocks() == 3
    assert allocator.get_num_free_cpu_blocks() == 8