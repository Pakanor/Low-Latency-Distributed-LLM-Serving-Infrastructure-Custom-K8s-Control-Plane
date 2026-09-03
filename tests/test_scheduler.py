from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
import llm_allocator_cpp


def test_scheduler_prefill_and_decode_limits():
    allocator = llm_allocator_cpp.PageAllocator(total_blocks=4, block_size=16)
    scheduler = Scheduler(allocator=allocator, max_batch_size=2)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[100] * 20) 
    seq2 = Sequence(seq_id=2, prompt_token_ids=[100] * 20) 
    seq3 = Sequence(seq_id=3, prompt_token_ids=[100] * 20)

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)
    scheduler.add_sequence(seq3)

    out1 = scheduler.schedule()
    assert len(out1.scheduled_prefills) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
    assert seq3.status == SequenceStatus.WAITING
    assert allocator.get_num_free_blocks() == 0

    out2 = scheduler.schedule()
    assert len(out2.scheduled_decodes) == 2
    assert len(out2.scheduled_prefills) == 0
    assert seq3.status == SequenceStatus.WAITING

    seq1.status = SequenceStatus.FINISHED

    out3 = scheduler.schedule()
    assert len(out3.scheduled_decodes) == 1  
    assert len(out3.scheduled_prefills) == 1  
    assert seq3.status == SequenceStatus.RUNNING


def test_scheduler_token_limit_guard():
    allocator = llm_allocator_cpp.PageAllocator(total_blocks=10, block_size=16)
    scheduler = Scheduler(allocator=allocator, max_batch_size=4, max_num_batched_tokens=30)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[10] * 20)  
    seq2 = Sequence(seq_id=2, prompt_token_ids=[10] * 20)  

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    out = scheduler.schedule()
    assert len(out.scheduled_prefills) == 1
    assert out.scheduled_prefills[0].seq_id == 1
    assert seq2.status == SequenceStatus.WAITING


if __name__ == "__main__":
    test_scheduler_prefill_and_decode_limits()
    test_scheduler_token_limit_guard()
    print("✓ Wszystkie testy Schedulera zakończone sukcesem!")