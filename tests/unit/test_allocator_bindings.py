import pytest

import llm_allocator_cpp


def test_page_allocator_invalid_geometry():
    with pytest.raises(ValueError, match="tokens_per_block"):
        llm_allocator_cpp.PageAllocator(1, 0, 4, 1)

    with pytest.raises(ValueError, match="bytes_per_token"):
        llm_allocator_cpp.PageAllocator(1, 4, 0, 1)


def test_page_allocator_zero_cpu_pool_disables_device_ptr():
    allocator = llm_allocator_cpp.PageAllocator(2, 4, 4, 0)

    assert allocator.is_zero_copy() is True
    assert allocator.get_device_ptr(0) == 0
    assert allocator.get_num_free_cpu_blocks() == 0


def test_page_allocator_refcounts_and_out_of_range_errors():
    allocator = llm_allocator_cpp.PageAllocator(3, 4, 4, 1)

    first = allocator.allocate_block()
    second = allocator.allocate_block()
    assert first == 0
    assert second == 1
    assert allocator.get_num_free_blocks() == 1

    allocator.share_block(first)
    allocator.free_block(first)
    assert allocator.get_num_free_blocks() == 1
    allocator.free_block(first)
    assert allocator.get_num_free_blocks() == 2

    with pytest.raises(RuntimeError, match="Double-free"):
        allocator.free_block(first)
    with pytest.raises(RuntimeError, match="Cannot share"):
        allocator.share_block(first)
    with pytest.raises(IndexError, match="Block ID out of range"):
        allocator.free_block(3)
    with pytest.raises(IndexError, match="Block ID out of range"):
        allocator.free_block(-1)


def test_page_allocator_swap_roundtrip_and_host_pointer():
    allocator = llm_allocator_cpp.PageAllocator(2, 4, 4, 1)

    gpu_block = allocator.allocate_block()
    # get_host_ptr requires cpu_block_id, not gpu_block_id
    # Before swap_out, there's no CPU mapping for this GPU block

    cpu_block = allocator.swap_out(gpu_block)
    assert cpu_block == 0
    assert allocator.get_cpu_block_id(gpu_block) == 0
    assert allocator.get_host_ptr(cpu_block) != 0
    assert allocator.get_num_free_blocks() == 2
    assert allocator.get_num_free_cpu_blocks() == 0

    with pytest.raises(RuntimeError, match="ref_count != 1"):
        allocator.swap_out(gpu_block)

    swapped_in = allocator.swap_in(cpu_block)
    assert swapped_in == 1
    assert allocator.get_num_free_blocks() == 1
    assert allocator.get_num_free_cpu_blocks() == 1

    with pytest.raises(RuntimeError, match="is not swapped out"):
        allocator.get_cpu_block_id(gpu_block)

    allocator.free_block(swapped_in)
    assert allocator.get_num_free_blocks() == 2


def test_page_allocator_release_swapped_block_returns_cpu_block():
    allocator = llm_allocator_cpp.PageAllocator(1, 4, 4, 1)
    gpu_block = allocator.allocate_block()
    cpu_block = allocator.swap_out(gpu_block)

    assert allocator.get_num_free_cpu_blocks() == 0
    allocator.release_swapped_block(cpu_block)
    assert allocator.get_num_free_cpu_blocks() == 1

    with pytest.raises(RuntimeError, match="is not swapped out"):
        allocator.get_cpu_block_id(gpu_block)


def test_sequence_block_table_release_uses_stored_allocator():
    allocator = llm_allocator_cpp.PageAllocator(8, 4, 4, 2)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)

    for _ in range(5):
        table.append_token()

    assert table.get_physical_blocks() == [0, 1]
    assert table.get_tokens_count() == 5
    assert allocator.get_num_free_blocks() == 6

    table.release()

    assert table.get_physical_blocks() == []
    assert table.get_tokens_count() == 0
    assert allocator.get_num_free_blocks() == 8

    table.release()
    assert allocator.get_num_free_blocks() == 8


def test_sequence_block_table_release_rejects_allocator_argument():
    allocator = llm_allocator_cpp.PageAllocator(4, 4, 4, 0)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)
    table.append_token()

    with pytest.raises(TypeError):
        table.release(allocator)


def test_sequence_block_table_append_token_and_release():
    allocator = llm_allocator_cpp.PageAllocator(4, 4, 4, 0)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)

    for _ in range(4):
        table.append_token()
    assert table.get_tokens_count() == 4
    assert table.get_physical_blocks() == [0]

    table.append_token()
    assert table.get_tokens_count() == 5
    assert table.get_physical_blocks() == [0, 1]
    assert allocator.get_num_free_blocks() == 2

    table.release()
    assert allocator.get_num_free_blocks() == 4


def test_sequence_block_table_validation_and_replace():
    allocator = llm_allocator_cpp.PageAllocator(8, 4, 4, 0)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)

    with pytest.raises(IndexError, match="Block ID out of range"):
        table.append_block(-1)
    with pytest.raises(IndexError, match="Block ID out of range"):
        table.append_block(8)

    old_block = allocator.allocate_block()
    new_block = allocator.allocate_block()
    table.append_block(old_block)

    with pytest.raises(IndexError, match="Block ID out of range"):
        table.replace_block(old_block, 8)
    with pytest.raises(RuntimeError, match="not found in block table"):
        table.replace_block(99, new_block)

    table.replace_block(old_block, new_block)
    assert table.get_physical_blocks() == [new_block]

    table.release()
    allocator.free_block(old_block)
    assert allocator.get_num_free_blocks() == 8


def test_sequence_block_table_keeps_allocator_alive():
    import gc

    allocator = llm_allocator_cpp.PageAllocator(4, 4, 4, 0)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)
    table.append_token()

    del allocator
    gc.collect()

    table.release()
    assert table.get_physical_blocks() == []
    assert table.get_tokens_count() == 0