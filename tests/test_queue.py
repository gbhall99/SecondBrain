from secondbrain.pipeline import queue as q


def test_enqueue_claim_complete(conn):
    jid = q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    assert jid is not None
    job = q.claim_next(conn, "transcribe")
    assert job is not None and job.payload["audio_file_id"] == 1
    q.complete(conn, job.id)
    assert q.claim_next(conn, "transcribe") is None
    assert q.counts(conn).get("done") == 1


def test_dedupe_by_key(conn):
    first = q.enqueue(conn, "transcribe", {"audio_file_id": 7}, dedupe_key="audio_file_id")
    second = q.enqueue(conn, "transcribe", {"audio_file_id": 7}, dedupe_key="audio_file_id")
    assert first is not None
    assert second is None  # deduped while first is still pending


def test_fail_backs_off_then_dead_letters(conn):
    q.enqueue(conn, "transcribe", {"audio_file_id": 9}, max_attempts=2)
    job = q.claim_next(conn, "transcribe")
    q.fail(conn, job, "boom")
    # re-queued (attempts 1 < max 2) but with backoff → not immediately claimable
    assert q.counts(conn).get("pending") == 1
    assert q.claim_next(conn, "transcribe") is None  # scheduled in the future
    # simulate the backoff window elapsing
    conn.execute("UPDATE jobs SET scheduled_at='2000-01-01T00:00:00.000Z'")
    job = q.claim_next(conn, "transcribe")
    q.fail(conn, job, "boom again")
    assert q.counts(conn).get("failed") == 1


def test_reclaim_stale_running_jobs(conn):
    q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    job = q.claim_next(conn, "transcribe")  # now 'running'
    assert job is not None
    conn.execute("UPDATE jobs SET started_at='2000-01-01T00:00:00.000Z'")  # long stuck
    assert q.reclaim_stale(conn) == 1
    assert q.counts(conn).get("pending") == 1
    # a fresh running job is NOT reclaimed
    q.claim_next(conn, "transcribe")
    assert q.reclaim_stale(conn) == 0


def test_priority_ordering(conn):
    q.enqueue(conn, "t", {"n": 1}, priority=0)
    q.enqueue(conn, "t", {"n": 2}, priority=5)
    job = q.claim_next(conn, "t")
    assert job.payload["n"] == 2  # higher priority first
