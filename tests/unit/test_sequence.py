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

    seq.append_token(202)
    assert seq.total_len == 5
    assert seq.status == SequenceStatus.FINISHED


def test_sequence_deadline_and_abandon_defaults():
    seq = Sequence(seq_id=2, prompt_token_ids=[1], max_tokens=5)
    assert seq.deadline is None
    assert seq.abandon_requested is False


def test_sequence_accepts_explicit_deadline():
    seq = Sequence(
        seq_id=3,
        prompt_token_ids=[1],
        max_tokens=5,
        deadline=999.0,
    )
    assert seq.deadline == 999.0


def test_sequence_abandon_flag_is_mutable():
    seq = Sequence(seq_id=4, prompt_token_ids=[1], max_tokens=5)
    seq.abandon_requested = True
    assert seq.abandon_requested is True