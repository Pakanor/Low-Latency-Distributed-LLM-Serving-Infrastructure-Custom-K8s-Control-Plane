import llm_allocator_cpp
from app.memory.sequence import Sequence, SequenceStatus


def test_sequence_lifecycle_and_allocation():
    allocator = llm_allocator_cpp.PageAllocator(total_blocks=5, block_size=4)
    prompt = [101, 102, 103, 104, 105, 106]
    seq = Sequence(seq_id=1, prompt_token_ids=prompt)

    assert seq.status == SequenceStatus.WAITING
    assert seq.total_len == 6
    assert seq.block_table is None

    seq.status = SequenceStatus.RUNNING
    seq.init_blocks(allocator)
    assert allocator.get_num_free_blocks() == 3

    seq.append_token(201, allocator)
    assert allocator.get_num_free_blocks() == 3

    seq.append_token(202, allocator)
    assert allocator.get_num_free_blocks() == 3

    seq.append_token(203, allocator)
    assert seq.total_len == 9
    assert allocator.get_num_free_blocks() == 2

    seq.free_blocks(allocator)
    assert seq.block_table is None
    assert allocator.get_num_free_blocks() == 5


if __name__ == "__main__":
    test_sequence_lifecycle_and_allocation()