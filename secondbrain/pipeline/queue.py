"""A small, durable, SQLite-backed job queue.

Single-machine, no Redis/broker. Decouples cheap real-time capture from the
heavy (and thermally significant) transcription work, which can be drained at
any pace — including off-peak. Jobs are claimed atomically so multiple worker
threads/processes are safe.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import timedelta

from secondbrain.storage import models
from secondbrain.storage.models import utcnow_iso


@dataclass
class Job:
    id: int
    type: str
    payload: dict
    attempts: int
    max_attempts: int


# Retry backoff schedule (minutes) indexed by attempt number. Long enough that a
# transient Ollama/HuggingFace outage rides through instead of dead-lettering.
BACKOFF_MINUTES: tuple[int, ...] = (1, 5, 30)

DEFAULT_MAX_ATTEMPTS = 3
# Transient-prone heavy jobs (model servers, gated downloads) get extra attempts
# so an outage lasting the whole backoff schedule still doesn't dead-letter them.
TRANSIENT_PRONE_MAX_ATTEMPTS: dict[str, int] = {
    "diarize_conversation": 5,
    "extract_knowledge": 5,
}


def enqueue(
    conn: sqlite3.Connection,
    job_type: str,
    payload: dict | None = None,
    *,
    priority: int = 0,
    max_attempts: int | None = None,
    dedupe_key: str | None = None,
) -> int | None:
    """Add a job. With ``dedupe_key`` (a JSON field), skip if an open job with
    the same type+key already exists. Returns the job id, or None if deduped."""
    payload = payload or {}
    if max_attempts is None:
        max_attempts = TRANSIENT_PRONE_MAX_ATTEMPTS.get(job_type, DEFAULT_MAX_ATTEMPTS)
    if dedupe_key is not None and dedupe_key in payload:
        existing = conn.execute(
            """
            SELECT id FROM jobs
            WHERE type = ? AND state IN ('pending', 'running')
              AND json_extract(payload, '$.' || ?) = ?
            LIMIT 1
            """,
            (job_type, dedupe_key, payload[dedupe_key]),
        ).fetchone()
        if existing is not None:
            return None
    cur = conn.execute(
        "INSERT INTO jobs (type, payload, priority, max_attempts) VALUES (?, ?, ?, ?)",
        (job_type, json.dumps(payload), priority, max_attempts),
    )
    return int(cur.lastrowid)


def claim_next(conn: sqlite3.Connection, job_type: str | None = None) -> Job | None:
    """Atomically claim the highest-priority due pending job, or None."""
    type_clause = "AND type = ?" if job_type else ""
    params: tuple = (utcnow_iso(),)
    if job_type:
        params = (utcnow_iso(), job_type)
    # IMMEDIATE so the SELECT+UPDATE is atomic against other workers.
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            f"""
            SELECT id, type, payload, attempts, max_attempts FROM jobs
            WHERE state = 'pending' AND scheduled_at <= ? {type_clause}
            ORDER BY priority DESC, scheduled_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE jobs SET state='running', attempts=attempts+1, started_at=? WHERE id=?",
            (utcnow_iso(), row["id"]),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return Job(
        id=row["id"],
        type=row["type"],
        payload=json.loads(row["payload"]),
        attempts=row["attempts"] + 1,
        max_attempts=row["max_attempts"],
    )


def complete(conn: sqlite3.Connection, job_id: int) -> None:
    conn.execute(
        "UPDATE jobs SET state='done', finished_at=?, error=NULL WHERE id=?",
        (utcnow_iso(), job_id),
    )


def fail(conn: sqlite3.Connection, job: Job, error: str) -> None:
    """Mark a job failed; re-queue with exponential backoff if attempts remain."""
    if job.attempts >= job.max_attempts:
        conn.execute(
            "UPDATE jobs SET state='failed', finished_at=?, error=? WHERE id=?",
            (utcnow_iso(), error[:2000], job.id),
        )
    else:
        # Back off so a persistently-failing (often heavy) job doesn't busy-loop.
        delay_min = BACKOFF_MINUTES[min(job.attempts, len(BACKOFF_MINUTES)) - 1]
        scheduled = models.iso_from_dt(
            models.parse_iso(utcnow_iso()) + timedelta(minutes=delay_min)
        )
        conn.execute(
            "UPDATE jobs SET state='pending', error=?, scheduled_at=? WHERE id=?",
            (error[:2000], scheduled, job.id),
        )


def reclaim_stale(conn: sqlite3.Connection, older_than_minutes: int = 30) -> int:
    """Return jobs stuck in 'running' (worker died mid-job) to 'pending'."""
    cutoff = models.iso_from_dt(
        models.parse_iso(utcnow_iso()) - timedelta(minutes=older_than_minutes)
    )
    cur = conn.execute(
        "UPDATE jobs SET state='pending' WHERE state='running' AND started_at IS NOT NULL "
        "AND started_at < ?",
        (cutoff,),
    )
    return cur.rowcount or 0


def requeue_failed(conn: sqlite3.Connection, job_type: str | None = None) -> int:
    """Move dead-lettered jobs (state='failed') back to pending for a fresh run.

    Attempts are reset so the full retry budget applies again; the last error is
    kept on the row for context until the job next completes. Returns the count."""
    type_clause = "AND type = ?" if job_type else ""
    params: tuple = (utcnow_iso(),)
    if job_type:
        params = (utcnow_iso(), job_type)
    cur = conn.execute(
        f"UPDATE jobs SET state='pending', attempts=0, scheduled_at=?, "
        f"started_at=NULL, finished_at=NULL WHERE state='failed' {type_clause}",
        params,
    )
    return cur.rowcount or 0


def prune_done_jobs(conn: sqlite3.Connection, keep_days: int = 30) -> int:
    """Delete completed jobs older than ``keep_days`` so the table can't grow
    forever (one row per chunk adds up). Returns the number pruned."""
    if keep_days < 0:
        return 0
    cutoff = models.iso_from_dt(
        models.parse_iso(utcnow_iso()) - timedelta(days=keep_days)
    )
    cur = conn.execute(
        "DELETE FROM jobs WHERE state='done' AND finished_at IS NOT NULL AND finished_at < ?",
        (cutoff,),
    )
    return cur.rowcount or 0


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
    return {r["state"]: r["n"] for r in rows}


def recent_failures(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """The most recently dead-lettered jobs (state='failed'), newest first."""
    rows = conn.execute(
        "SELECT id, type, attempts, max_attempts, error, finished_at "
        "FROM jobs WHERE state='failed' ORDER BY finished_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
