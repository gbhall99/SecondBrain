"""queue_overview / recent_failures — queue inspection for ops."""

from __future__ import annotations

from secondbrain.pipeline import queue as q
from secondbrain.query import service


def _dead_letter(conn, audio_id, error):
    q.enqueue(conn, "transcribe", {"audio_file_id": audio_id}, max_attempts=1)
    job = q.claim_next(conn, "transcribe")
    q.fail(conn, job, error)


def test_queue_overview_empty(conn):
    ov = service.queue_overview(conn)
    assert ov["counts"] == {}
    assert ov["recent_failures"] == []


def test_queue_overview_reports_failures(conn):
    _dead_letter(conn, 1, "boom one")
    _dead_letter(conn, 2, "boom two")
    ov = service.queue_overview(conn)
    assert ov["counts"].get("failed") == 2
    errors = [f["error"] for f in ov["recent_failures"]]
    assert "boom one" in errors and "boom two" in errors
    assert all(f["attempts"] >= 1 for f in ov["recent_failures"])


def test_recent_failures_limit(conn):
    for i in range(5):
        _dead_letter(conn, i, f"err{i}")
    assert len(q.recent_failures(conn, limit=3)) == 3


def test_reclaim_stale_jobs_helper(conn):
    q.enqueue(conn, "transcribe", {"audio_file_id": 1})
    q.claim_next(conn, "transcribe")  # now 'running'
    conn.execute("UPDATE jobs SET started_at='2000-01-01T00:00:00.000Z'")
    assert service.reclaim_stale_jobs(conn) == 1
    assert q.counts(conn).get("pending") == 1
