from app.scheduler.sequence import Sequence, SequenceStatus


def test_sequence_lifecycle():
    seq = Sequence(seq_id=1, prompt_token_ids=[101, 102, 103], max_tokens=2)
    
    assert seq.seq_id == 1
    assert seq.total_len == 3
    assert seq.status == SequenceStatus.WAITING
    assert seq.block_table is None

    seq.append_token(201)
    assert seq.total_len == 4
    assert seq.status == SequenceStatus.WAITING

    # Dodanie 2. tokena wyjściowego (osiągnięcie max_tokens=2)
    seq.append_token(202)
    assert seq.total_len == 5
    assert seq.status == SequenceStatus.FINISHED


if __name__ == "__main__":
    test_sequence_lifecycle()
    print("test_sequence.py PASSED")