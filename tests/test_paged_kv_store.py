import pytest
import torch
from app.memory.manager import PagedKVCacheManager
from app.scheduler.sequence import Sequence


@pytest.fixture
def kv_manager():
    return PagedKVCacheManager(
        total_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        num_layers=2,
    )


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
    keys = torch.randn(5, 2, 2, 8)
    values = torch.randn(5, 2, 2, 8)

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

    first_keys = torch.randn(4, 2, 2, 8)
    kv_manager.scatter_kv(
        first_keys, first_keys.clone(), kv_manager.build_slot_mapping([first], [0], [4])
    )
    second_keys = torch.randn(4, 2, 2, 8)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
