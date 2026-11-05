

def check_allocator_invariants(allocator, sequences, total_blocks, block_size):
    free = allocator.get_num_free_blocks()
    used_blocks = set()
    total_capacity = 0
    for seq in sequences:
        if seq.block_table is None:
            continue
        phys = list(seq.block_table.get_physical_blocks())
        for b in phys:
            assert b not in used_blocks, f"block {b} owned twice"
            used_blocks.add(b)
        assert seq.block_table.get_tokens_count() <= len(phys) * block_size
        total_capacity += len(phys) * block_size
    assert free + len(used_blocks) == total_blocks
    assert sum(
        seq.block_table.get_tokens_count() for seq in sequences if seq.block_table is not None
    ) <= total_capacity
