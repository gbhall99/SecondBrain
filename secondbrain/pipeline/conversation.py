"""Group consecutive transcribed chunks into conversations for diarization.

Ambient audio is continuous; diarization is far more accurate over a whole
conversation than over isolated 60s chunks. A chunk joins the open conversation
if it starts within ``max_gap_minutes`` of the previous chunk's end; otherwise
the open conversation is closed (→ a ``diarize`` job is enqueued) and a new one
opens. Closing is also forced when the most recent chunk is old enough that no
more are coming.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.storage.models import parse_iso, utcnow_iso

JOB_DIARIZE = "diarize_conversation"


def _open_conversation(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM conversations WHERE status='open' ORDER BY id DESC LIMIT 1"
    ).fetchone()


def assign_chunk(
    conn: sqlite3.Connection, audio_file_id: int, settings: Settings | None = None
) -> int:
    """Attach a freshly-transcribed chunk to the open conversation (or start one).

    Returns the conversation id. Closes+enqueues the previous conversation if the
    gap since its last chunk exceeds ``max_gap_minutes``.
    """
    settings = settings or get_settings()
    af = conn.execute("SELECT * FROM audio_files WHERE id=?", (audio_file_id,)).fetchone()
    if af is None:
        raise ValueError(f"unknown audio_file {audio_file_id}")

    conv = _open_conversation(conn)
    if conv is not None and _gap_exceeded(conn, conv, af["started_at"], settings):
        close_conversation(conn, conv["id"], settings)
        conv = None

    if conv is None:
        cur = conn.execute(
            "INSERT INTO conversations (started_at, status, chunk_count) VALUES (?, 'open', 0)",
            (af["started_at"],),
        )
        conv_id = int(cur.lastrowid)
    else:
        conv_id = int(conv["id"])

    conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (conv_id, audio_file_id))
    # Use MIN/MAX so a retried (out-of-order) chunk widens the span correctly
    # rather than rewriting ended_at backward or leaving started_at > ended_at.
    conn.execute(
        "UPDATE conversations SET chunk_count = chunk_count + 1, "
        "started_at = MIN(started_at, ?), "
        "ended_at = MAX(COALESCE(ended_at, ''), ?) WHERE id=?",
        (af["started_at"], af["ended_at"] or af["started_at"], conv_id),
    )
    return conv_id


def _gap_exceeded(
    conn: sqlite3.Connection, conv: sqlite3.Row, next_started_at: str, settings: Settings
) -> bool:
    if not conv["ended_at"]:
        return False
    try:
        gap = parse_iso(next_started_at) - parse_iso(conv["ended_at"])
    except ValueError:
        return False
    return gap > timedelta(minutes=settings.conversation.max_gap_minutes)


def close_conversation(
    conn: sqlite3.Connection, conversation_id: int, settings: Settings | None = None
) -> int | None:
    """Mark a conversation closed and enqueue its diarization job.

    A sub-``min_conversation_seconds`` conversation (e.g. a short partial tail chunk
    in an idle period) is closed WITHOUT a heavy diarize+extract job — it's marked
    done and its raw audio gets a normal retention deadline. Returns the enqueued
    job id, or None when diarization was skipped / the conversation was already closed.
    """
    settings = settings or get_settings()
    conv = conn.execute(
        "SELECT started_at, ended_at FROM conversations WHERE id=? AND status='open'",
        (conversation_id,),
    ).fetchone()
    if conv is None:
        return None  # already closed / unknown

    dur = _duration_s(conv["started_at"], conv["ended_at"])
    if dur is not None and dur < settings.conversation.min_conversation_seconds:
        # Too short to be worth diarizing: mark done and finalize retention directly.
        from secondbrain.storage import retention

        conn.execute(
            "UPDATE conversations SET status='diarized', knowledge_status='extracted' WHERE id=?",
            (conversation_id,),
        )
        conn.execute(
            "UPDATE audio_files SET retention_delete_after=? "
            "WHERE conversation_id=? AND status='transcribed'",
            (retention.compute_delete_after(settings), conversation_id),
        )
        return None

    conn.execute(
        "UPDATE conversations SET status='closed' WHERE id=? AND status='open'", (conversation_id,)
    )
    return q.enqueue(
        conn, JOB_DIARIZE, {"conversation_id": conversation_id}, dedupe_key="conversation_id"
    )


def _duration_s(started_at: str | None, ended_at: str | None) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        return (parse_iso(ended_at) - parse_iso(started_at)).total_seconds()
    except ValueError:
        return None


def close_stale_conversations(conn: sqlite3.Connection, settings: Settings | None = None) -> int:
    """Close any open conversation whose last chunk is older than the gap window.

    Run periodically (daemon maintenance) so the final conversation of an idle
    period gets diarized without waiting for the next chunk.
    """
    settings = settings or get_settings()
    cutoff = utcnow_iso()
    rows = conn.execute("SELECT * FROM conversations WHERE status='open'").fetchall()
    closed = 0
    for conv in rows:
        if not conv["ended_at"]:
            continue
        try:
            gap = parse_iso(cutoff) - parse_iso(conv["ended_at"])
        except ValueError:
            continue
        if gap > timedelta(minutes=settings.conversation.max_gap_minutes):
            close_conversation(conn, conv["id"], settings)
            closed += 1
    return closed
