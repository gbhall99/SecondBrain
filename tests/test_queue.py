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


def test_backoff_schedule_lengthens(conn):
    """Retries back off 1min → 5min → 30min so a transient outage rides through."""
    from secondbrain.storage import models

    q.enqueue(conn, "t", {"n": 1}, max_attempts=4)
    for expected_min in (1, 5, 30):
        conn.execute("UPDATE jobs SET scheduled_at='2000-01-01T00:00:00.000Z'")
        job = q.claim_next(conn, "t")
        before = models.parse_iso(models.utcnow_iso())
        q.fail(conn, job, "boom")
        sched = models.parse_iso(
            conn.execute("SELECT scheduled_at FROM jobs").fetchone()["scheduled_at"]
        )
        delay_s = (sched - before).total_seconds()
        assert abs(delay_s - expected_min * 60) < 5, f"expected ~{expected_min}min backoff"


def test_transient_prone_types_get_more_attempts(conn):
    q.enqueue(conn, "diarize_conversation", {"conversation_id": 1})
    q.enqueue(conn, "extract_knowledge", {"conversation_id": 1})
    q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    q.enqueue(conn, "diarize_conversation", {"conversation_id": 2}, max_attempts=1)
    rows = {
        (r["type"], r["max_attempts"])
        for r in conn.execute("SELECT type, max_attempts FROM jobs").fetchall()
    }
    assert ("diarize_conversation", 5) in rows   # transient-prone default raised
    assert ("extract_knowledge", 5) in rows
    assert ("transcribe", 3) in rows             # normal default unchanged
    assert ("diarize_conversation", 1) in rows   # explicit override respected


def test_requeue_failed_resets_and_reschedules(conn):
    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    job = q.claim_next(conn, "transcribe")
    q.fail(conn, job, "boom")  # dead-lettered (attempts == max)
    assert q.counts(conn).get("failed") == 1

    assert q.requeue_failed(conn) == 1
    row = conn.execute(
        "SELECT state, attempts, error, started_at, finished_at FROM jobs"
    ).fetchone()
    assert row["state"] == "pending"
    assert row["attempts"] == 0            # full retry budget again
    assert row["error"] == "boom"          # last error kept for context
    assert row["started_at"] is None and row["finished_at"] is None
    # immediately claimable (scheduled now, not backed off)
    assert q.claim_next(conn, "transcribe") is not None


def test_requeue_failed_filters_by_type(conn):
    for jtype in ("transcribe", "diarize_conversation"):
        q.enqueue(conn, jtype, {"x": 1}, max_attempts=1)
        job = q.claim_next(conn, jtype)
        q.fail(conn, job, "boom")
    assert q.requeue_failed(conn, "transcribe") == 1
    states = {
        r["type"]: r["state"] for r in conn.execute("SELECT type, state FROM jobs").fetchall()
    }
    assert states["transcribe"] == "pending"
    assert states["diarize_conversation"] == "failed"  # untouched


def test_cli_queue_retry_failed(conn, settings, monkeypatch):
    from typer.testing import CliRunner

    from secondbrain import cli

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    assert q.counts(conn).get("failed") == 1

    result = CliRunner().invoke(cli.app, ["queue", "--retry-failed"])
    assert result.exit_code == 0, result.output
    assert "Re-queued 1 failed job(s)." in result.output
    assert q.counts(conn).get("pending") == 1


def test_prune_done_jobs_deletes_only_old_done(conn):
    # an old done job, a fresh done job, and an old failed job
    q.enqueue(conn, "t", {"n": 1})
    old_done = q.claim_next(conn, "t")
    q.complete(conn, old_done.id)
    conn.execute(
        "UPDATE jobs SET finished_at='2000-01-01T00:00:00.000Z' WHERE id=?", (old_done.id,)
    )
    q.enqueue(conn, "t", {"n": 2})
    fresh_done = q.claim_next(conn, "t")
    q.complete(conn, fresh_done.id)
    q.enqueue(conn, "t", {"n": 3}, max_attempts=1)
    failed = q.claim_next(conn, "t")
    q.fail(conn, failed, "boom")
    conn.execute(
        "UPDATE jobs SET finished_at='2000-01-01T00:00:00.000Z' WHERE id=?", (failed.id,)
    )

    assert q.prune_done_jobs(conn, keep_days=30) == 1
    remaining = {r["id"] for r in conn.execute("SELECT id FROM jobs").fetchall()}
    assert remaining == {fresh_done.id, failed.id}  # failed rows are never pruned
