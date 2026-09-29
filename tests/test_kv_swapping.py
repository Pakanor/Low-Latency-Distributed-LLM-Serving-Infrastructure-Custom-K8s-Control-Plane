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
        device="cpu",
        num_layers=1,
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


def test_swapped_block_buffers_are_device_addressable():
    manager = PagedKVCacheManager(
        total_blocks=4,
        block_size=4,
        num_heads=2,
        head_dim=8,
        total_cpu_blocks=8,
        device="cpu",
        num_layers=1,
    )
    allocator = manager.get_allocator()

    seq = Sequence(seq_id=102, prompt_token_ids=[1, 2, 3, 4])
    manager.allocate_prefix_blocks(seq)
    block_id = seq.block_table.get_physical_blocks()[0]

    assert allocator.get_host_ptr(block_id) == 0
    assert allocator.get_device_ptr(block_id) == 0

    manager.swap_out_sequence(seq)

    host_ptr = allocator.get_host_ptr(block_id)
    device_ptr = allocator.get_device_ptr(block_id)
    assert host_ptr != 0
    assert (device_ptr != 0) == allocator.is_zero_copy()

    if allocator.is_zero_copy():
        second_seq = Sequence(seq_id=103, prompt_token_ids=[5, 6, 7, 8])
        manager.allocate_prefix_blocks(second_seq)
        manager.swap_out_sequence(second_seq)
        second_block_id = second_seq.block_table.get_physical_blocks()[0]
        assert allocator.get_device_ptr(second_block_id) - device_ptr == allocator.get_block_size()
