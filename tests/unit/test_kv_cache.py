import pytest
import torch
import llm_allocator_cpp
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence


@pytest.fixture
def kv_manager():
    return PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )


def test_allocate_and_write_prefix(kv_manager):
    seq = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30, 40, 50])
    kv_manager.allocate_prefix_blocks(seq)

    assert seq.block_table is not None
    assert seq.block_table.get_tokens_count() == 5
    assert len(seq.block_table.get_physical_blocks()) == 2

    keys = torch.randn(1, 5, 2, 8)
    values = torch.randn(1, 5, 2, 8)

    slot_mapping = kv_manager.write_prefix_kv(seq, keys, values)
    assert len(slot_mapping) == 5


def test_single_token_kv_write(kv_manager):
    seq = Sequence(seq_id=2, prompt_token_ids=[10, 20, 30, 40])
    kv_manager.allocate_prefix_blocks(seq)

    seq.append_token(50)
    kv_manager.allocate_slot_for_next_token(seq)

    key_token = torch.randn(1, 2, 8)
    value_token = torch.randn(1, 2, 8)

    kv_manager.write_single_token_kv(seq, key_token, value_token)
    assert seq.block_table.get_tokens_count() == 5
    assert len(seq.block_table.get_physical_blocks()) == 2


def test_single_token_kv_write_requires_block_table(kv_manager):
    seq = Sequence(seq_id=4, prompt_token_ids=[1])
    token = torch.zeros(1, 2, 8)

    with pytest.raises(ValueError, match="no block table"):
        kv_manager.write_single_token_kv(seq, token, token)


def test_slot_mapping_rejects_sequence_without_block_table(kv_manager):
    seq = Sequence(seq_id=5, prompt_token_ids=[1])

    with pytest.raises(ValueError, match="no block table"):
        kv_manager.build_slot_mapping([seq], [0], [1])


def test_free_sequence(kv_manager):
    allocator = kv_manager.get_allocator()
    initial_free = allocator.get_num_free_blocks()

    seq = Sequence(seq_id=3, prompt_token_ids=[1, 2, 3, 4])
    kv_manager.allocate_prefix_blocks(seq)
    assert allocator.get_num_free_blocks() == initial_free - 1

    kv_manager.free_sequence(seq)
    assert allocator.get_num_free_blocks() == initial_free
    assert seq.block_table is None


def test_gather_kv_cache_round_trip():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=1,
    )
    seq = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30, 40])
    manager.allocate_prefix_blocks(seq)

    keys = torch.randn(1, 4, 2, 8)
    values = torch.randn(1, 4, 2, 8)
    manager.write_prefix_kv(seq, keys, values)

    gathered = manager.gather_kv_cache(seq, read_len=4)
    assert gathered is not None
    assert len(gathered) == 1

    gathered_key, gathered_value = gathered[0]
    assert gathered_key.shape == (1, 2, 4, 8)
    assert gathered_value.shape == (1, 2, 4, 8)

    original_key = keys[0].unsqueeze(0).transpose(1, 2)
    original_value = values[0].unsqueeze(0).transpose(1, 2)
    assert torch.allclose(gathered_key, original_key)
    assert torch.allclose(gathered_value, original_value)


def test_block_size_is_tokens_not_bytes():
    num_heads = 2
    head_dim = 8
    dtype = torch.float32
    bytes_per_token = num_heads * head_dim * dtype.itemsize * 2
    tokens_per_block = 4
    total_cpu_blocks = 32

    allocator = llm_allocator_cpp.PageAllocator(
        total_blocks=10,
        tokens_per_block=tokens_per_block,
        bytes_per_token=bytes_per_token,
        total_cpu_blocks=total_cpu_blocks,
    )

    assert allocator.get_block_size() == tokens_per_block
    assert allocator.get_bytes_per_token() == bytes_per_token
    expected_pool_bytes = total_cpu_blocks * tokens_per_block * bytes_per_token
    assert allocator.get_host_pool_bytes() == expected_pool_bytes


def _make_sequence(manager, seq_id, prompt_len):
    seq = Sequence(seq_id=seq_id, prompt_token_ids=list(range(prompt_len)))
    manager.allocate_prefix_blocks(seq)
    return seq


def test_slot_mapping_targets_physical_slots(kv_manager):
    seq = _make_sequence(kv_manager, seq_id=1, prompt_len=6)
    physical_blocks = seq.block_table.get_physical_blocks()

    slot_mapping = kv_manager.build_slot_mapping([seq], [0], [6])

    expected = torch.tensor(
        [physical_blocks[0] * 4 + i for i in range(4)]
        + [physical_blocks[1] * 4 + i for i in range(2)]
    )
    assert torch.equal(slot_mapping.cpu(), expected)


def test_slot_mapping_spans_multiple_sequences_in_order(kv_manager):
    first = _make_sequence(kv_manager, seq_id=1, prompt_len=3)
    second = _make_sequence(kv_manager, seq_id=2, prompt_len=5)

    slot_mapping = kv_manager.build_slot_mapping([first, second], [0, 0], [3, 5])

    first_slots = kv_manager.build_slot_mapping([first], [0], [3])
    second_slots = kv_manager.build_slot_mapping([second], [0], [5])
    assert torch.equal(slot_mapping.cpu(), torch.cat([first_slots, second_slots]))


def test_slot_mapping_offset_targets_decode_slot(kv_manager):
    seq = _make_sequence(kv_manager, seq_id=1, prompt_len=4)
    physical_blocks = seq.block_table.get_physical_blocks()

    slot_mapping = kv_manager.build_slot_mapping([seq], [3], [1])

    assert torch.equal(slot_mapping.cpu(), torch.tensor([physical_blocks[0] * 4 + 3]))


def test_scatter_kv_writes_flat_ragged_batch(kv_manager):
    first = _make_sequence(kv_manager, seq_id=1, prompt_len=3)
    second = _make_sequence(kv_manager, seq_id=2, prompt_len=2)

    slot_mapping = kv_manager.build_slot_mapping([first, second], [0, 0], [3, 2])
    keys = torch.randn(5, 1, 2, 8)
    values = torch.randn(5, 1, 2, 8)

    kv_manager.scatter_kv(keys, values, slot_mapping)

    first_blocks = first.block_table.get_physical_blocks()
    second_blocks = second.block_table.get_physical_blocks()
    for token in range(3):
        assert torch.equal(kv_manager.key_cache[0, first_blocks[0], token], keys[token, 0])
        assert torch.equal(kv_manager.value_cache[0, first_blocks[0], token], values[token, 0])
    for token in range(2):
        assert torch.equal(kv_manager.key_cache[0, second_blocks[0], token], keys[3 + token, 0])
        assert torch.equal(kv_manager.value_cache[0, second_blocks[0], token], values[3 + token, 0])


def test_scatter_kv_does_not_touch_other_sequences(kv_manager):
    first = _make_sequence(kv_manager, seq_id=1, prompt_len=4)
    second = _make_sequence(kv_manager, seq_id=2, prompt_len=4)

    first_keys = torch.randn(4, 1, 2, 8)
    kv_manager.scatter_kv(
        first_keys, first_keys.clone(), kv_manager.build_slot_mapping([first], [0], [4])
    )
    second_keys = torch.randn(4, 1, 2, 8)
    kv_manager.scatter_kv(
        second_keys, second_keys.clone(), kv_manager.build_slot_mapping([second], [0], [4])
    )

    second_blocks = second.block_table.get_physical_blocks()
    for token in range(4):
        assert torch.equal(kv_manager.key_cache[0, second_blocks[0], token], second_keys[token, 0])


def test_scatter_kv_rejects_wrong_token_shape(kv_manager):
    seq = _make_sequence(kv_manager, seq_id=1, prompt_len=2)
    slot_mapping = kv_manager.build_slot_mapping([seq], [0], [2])

    with pytest.raises(ValueError):
        kv_manager.scatter_kv(torch.randn(2, 2, 8), torch.randn(2, 2, 8), slot_mapping)


def test_build_slot_mapping_rejects_unallocated_tokens(kv_manager):
    seq = _make_sequence(kv_manager, seq_id=1, prompt_len=4)

    with pytest.raises(ValueError):
        kv_manager.build_slot_mapping([seq], [0], [5])


def test_build_slot_mapping_rejects_mismatched_lengths(kv_manager):
    seq = _make_sequence(kv_manager, seq_id=1, prompt_len=4)

    with pytest.raises(ValueError):
        kv_manager.build_slot_mapping([seq], [0, 1], [4])
