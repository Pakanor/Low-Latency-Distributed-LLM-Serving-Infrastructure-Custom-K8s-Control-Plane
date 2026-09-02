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
    assert len(blocks) == 2
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


def test_allocate_and_write_prefix_partial_block():
    total_blocks = 10
    block_size = 16
    num_heads = 4
    head_dim = 32
    seq_len = 25  

    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu"
    )

    table = manager.create_sequence_table()

    keys = torch.randn((seq_len, num_heads, head_dim), dtype=torch.float32)
    values = torch.randn((seq_len, num_heads, head_dim), dtype=torch.float32)

    slot_mapping = manager.allocate_and_write_prefix(table, keys, values)

    physical_blocks = table.get_physical_blocks()
    assert len(physical_blocks) == 2, f"Oczekiwano 2 bloków, dostano {len(physical_blocks)}"

    assert slot_mapping.shape[0] == seq_len, f"Oczekiwano {seq_len} slotów, dostano {slot_mapping.shape[0]}"

   
    expected_block_0 = torch.arange(0, 16)
    expected_block_1 = torch.arange(16, 25)
    expected_slots = torch.cat([expected_block_0, expected_block_1])
    assert torch.equal(slot_mapping, expected_slots), "Błąd w wyliczaniu wektora slot_mapping!"

    flat_key_cache = manager.key_cache.view(-1, num_heads, head_dim)
    flat_val_cache = manager.value_cache.view(-1, num_heads, head_dim)

    assert torch.equal(flat_key_cache[slot_mapping], keys), "Key cache zgubił dane przy zapisie!"
    assert torch.equal(flat_val_cache[slot_mapping], values), "Value cache zgubił dane przy zapisie!"

    print("\n[OK] Test Prefill (Bulk Write) przeszedł pomyślnie!")


if __name__ == "__main__":
    test_paged_kv_cache_manager()
    test_build_block_tables()
    print("✓ All tests passed")