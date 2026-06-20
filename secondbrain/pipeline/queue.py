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


def enqueue(
    conn: sqlite3.Connection,
    job_type: str,
    payload: dict | None = None,
    *,
    priority: int = 0,
    max_attempts: int = 3,
    dedupe_key: str | None = None,
) -> int | None:
    """Add a job. With ``dedupe_key`` (a JSON field), skip if an open job with
    the same type+key already exists. Returns the job id, or None if deduped."""
    payload = payload or {}
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
        delay_min = 2 ** (job.attempts - 1)
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
