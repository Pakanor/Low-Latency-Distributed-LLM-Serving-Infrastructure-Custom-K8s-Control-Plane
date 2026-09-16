import pytest
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence


@pytest.fixture
def kv_manager():
    return PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )


def test_allocate_and_write_prefix(kv_manager):
    seq = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30, 40, 50])  # 5 tokenów -> 2 bloki
    kv_manager.allocate_prefix_blocks(seq)

    assert seq.block_table is not None
    assert seq.block_table.get_tokens_count() == 5
    assert len(seq.block_table.get_physical_blocks()) == 2

    keys = torch.randn(5, 2, 8)
    values = torch.randn(5, 2, 8)

    slot_mapping = kv_manager.write_prefix_kv(seq, keys, values)
    assert len(slot_mapping) == 5


def test_single_token_kv_write(kv_manager):
    seq = Sequence(seq_id=2, prompt_token_ids=[10, 20, 30, 40])  # 4 tokeny -> 1 blok
    kv_manager.allocate_prefix_blocks(seq)

    seq.append_token(50)
    kv_manager.allocate_slot_for_next_token(seq)

    key_token = torch.randn(2, 8)
    value_token = torch.randn(2, 8)

    kv_manager.write_single_token_kv(seq, key_token, value_token)
    assert seq.block_table.get_tokens_count() == 5
    assert len(seq.block_table.get_physical_blocks()) == 2


def test_free_sequence(kv_manager):
    allocator = kv_manager.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    seq = Sequence(seq_id=3, prompt_token_ids=[1, 2, 3, 4])
    kv_manager.allocate_prefix_blocks(seq)
    assert allocator.get_num_free_blocks() == initial_free - 1

    kv_manager.free_sequence(seq)
    assert allocator.get_num_free_blocks() == initial_free
    assert seq.block_table is None