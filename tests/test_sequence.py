from app.scheduler.sequence import Sequence, SequenceStatus
import llm_allocator_cpp


def test_sequence_lifecycle_and_allocation():
    allocator = llm_allocator_cpp.PageAllocator(total_blocks=4, block_size=16)
    seq = Sequence(seq_id=1, prompt_token_ids=[10] * 20)  

    assert seq.status == SequenceStatus.WAITING
    assert seq.block_table is None

    seq.init_blocks(allocator)
    assert seq.block_table is not None
    assert allocator.get_num_free_blocks() == 2

    # Zwalnianie zasobów
    seq.free_blocks(allocator)
    assert seq.block_table is None
    assert allocator.get_num_free_blocks() == 4


if __name__ == "__main__":
    test_sequence_lifecycle_and_allocation()
    print("✓ Test sequence lifecycle przeszedł pomyślnie!")