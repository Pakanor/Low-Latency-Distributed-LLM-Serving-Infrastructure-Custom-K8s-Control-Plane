from app.scheduler.sequence import Sequence, SequenceStatus
from app.scheduler.scheduler import Scheduler
from app.memory.manager import PagedKVCacheManager


def test_scheduler_prefill_and_decode_limits():
    manager = PagedKVCacheManager(
        total_blocks=10,
        block_size=16,
        num_heads=8,
        head_dim=64
    )
    
    scheduler = Scheduler(kv_cache_manager=manager, max_batch_size=2)

    seq1 = Sequence(seq_id=1, prompt_token_ids=[1] * 10, max_tokens=5)
    seq2 = Sequence(seq_id=2, prompt_token_ids=[2] * 20, max_tokens=5)
    
    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    assert scheduler.has_unfinished_sequences() is True

    # 1. Krok: Prefill
    outputs = scheduler.schedule()
    assert len(outputs.scheduled_prefills) == 2
    assert len(outputs.scheduled_decodes) == 0
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING

    seq1.append_token(100)
    seq2.append_token(200)

    outputs_decode = scheduler.schedule()
    assert len(outputs_decode.scheduled_prefills) == 0
    assert len(outputs_decode.scheduled_decodes) == 2


if __name__ == "__main__":
    test_scheduler_prefill_and_decode_limits()
    print("test_scheduler.py PASSED")