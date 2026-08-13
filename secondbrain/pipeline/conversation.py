"""Group consecutive transcribed chunks into conversations for diarization.

Ambient audio is continuous; diarization is far more accurate over a whole
conversation than over isolated 60s chunks. A chunk joins the open conversation
whose time span (± ``max_gap_minutes``) covers it; otherwise the open
conversation is closed (→ a ``diarize`` job is enqueued when diarization is
enabled) and a new one opens. Closing is also forced when the most recent chunk
is old enough that no more are coming, or when a chunk would stretch the
conversation past ``max_conversation_minutes``.

Segmentation runs even with diarization disabled: conversations are the unit
knowledge extraction operates on, so they are closed straight to the
``diarized`` state extraction expects instead of enqueueing a diarize job.
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


def _matching_conversation(
    conn: sqlite3.Connection, chunk_start: str, chunk_end: str, settings: Settings
) -> sqlite3.Row | None:
    """The open conversation whose time span (± the gap tolerance) covers the chunk.

    Matching by timestamp window (not "newest open conversation") means a
    late-arriving old chunk (e.g. a retried transcription job) attaches to the
    conversation it belongs to — or gets a fresh one — instead of stretching
    the current meeting backward. A match that would stretch the conversation
    past ``max_conversation_minutes`` closes it instead (a new one opens).
    """
    gap = timedelta(minutes=settings.conversation.max_gap_minutes)
    max_len = timedelta(minutes=settings.conversation.max_conversation_minutes)
    try:
        c_start = parse_iso(chunk_start)
        c_end = parse_iso(chunk_end)
    except ValueError:
        return _open_conversation(conn)  # unparseable timestamps: legacy behavior
    rows = conn.execute(
        "SELECT * FROM conversations WHERE status='open' ORDER BY id DESC"
    ).fetchall()
    for conv in rows:
        try:
            v_start = parse_iso(conv["started_at"])
            v_end = parse_iso(conv["ended_at"] or conv["started_at"])
        except (TypeError, ValueError):
            continue
        if not (v_start - gap <= c_start <= v_end + gap):
            continue
        if max(v_end, c_end) - min(v_start, c_start) > max_len:
            close_conversation(conn, conv["id"], settings)
            continue
        return conv
    return None


def assign_chunk(
    conn: sqlite3.Connection, audio_file_id: int, settings: Settings | None = None
) -> int:
    """Attach a freshly-transcribed chunk to its conversation (or start one).

    Returns the conversation id. Closes+enqueues the previous conversation if the
    gap since its last chunk exceeds ``max_gap_minutes``.
    """
    settings = settings or get_settings()
    af = conn.execute("SELECT * FROM audio_files WHERE id=?", (audio_file_id,)).fetchone()
    if af is None:
        raise ValueError(f"unknown audio_file {audio_file_id}")
    chunk_start = af["started_at"]
    chunk_end = af["ended_at"] or af["started_at"]

    conv = _matching_conversation(conn, chunk_start, chunk_end, settings)
    if conv is None:
        # Only a chunk arriving AFTER the idle window ends the current meeting;
        # an out-of-window OLD chunk gets its own conversation and must not
        # close (or extend) the one still receiving live chunks.
        newest = _open_conversation(conn)
        if newest is not None and _gap_exceeded(conn, newest, chunk_start, settings):
            close_conversation(conn, newest["id"], settings)
        cur = conn.execute(
            "INSERT INTO conversations (started_at, status, chunk_count) VALUES (?, 'open', 0)",
            (chunk_start,),
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
        (chunk_start, chunk_end, conv_id),
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
    done (``knowledge_status='skipped'``, never a lying ``'extracted'``) and its
    raw audio gets a normal retention deadline. With diarization disabled the
    conversation closes straight to ``diarized`` so knowledge extraction (which
    only needs the transcript) still picks it up. Returns the enqueued job id, or
    None when no diarize job was enqueued / the conversation was already closed.
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
        # Too short to be worth diarizing/extracting: mark done and finalize
        # retention directly. 'skipped' keeps it distinct from real extraction.
        from secondbrain.storage import retention

        conn.execute(
            "UPDATE conversations SET status='diarized', knowledge_status='skipped' WHERE id=?",
            (conversation_id,),
        )
        conn.execute(
            "UPDATE audio_files SET retention_delete_after=? "
            "WHERE conversation_id=? AND status='transcribed'",
            (retention.compute_delete_after(settings), conversation_id),
        )
        return None

    if not settings.diarization.enabled:
        # No diarization: close straight to the state extraction expects (the
        # daemon catch-up looks for status='diarized' + knowledge_status='pending').
        from secondbrain.storage import retention

        conn.execute(
            "UPDATE conversations SET status='diarized' WHERE id=? AND status='open'",
            (conversation_id,),
        )
        # Chunks normally get their deadline at transcription when diarization is
        # off; cover any deferred stragglers (e.g. diarization was on earlier).
        conn.execute(
            "UPDATE audio_files SET retention_delete_after=? "
            "WHERE conversation_id=? AND status='transcribed' "
            "AND retention_delete_after IS NULL",
            (retention.compute_delete_after(settings), conversation_id),
        )
        if settings.extraction.enabled:
            from secondbrain.knowledge.extract import enqueue_extraction

            enqueue_extraction(conn, conversation_id)
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
    period gets diarized without waiting for the next chunk. A conversation with
    no usable ``ended_at`` derives its end from its newest chunk — or is closed
    immediately — so a malformed row can never stay open forever.
    """
    settings = settings or get_settings()
    cutoff = utcnow_iso()
    rows = conn.execute("SELECT * FROM conversations WHERE status='open'").fetchall()
    closed = 0
    for conv in rows:
        end = conv["ended_at"]
        if not end:
            # Derive the end from the newest attached chunk (NULL ended_at can
            # happen when a crash interrupted the span update).
            row = conn.execute(
                "SELECT MAX(COALESCE(ended_at, started_at)) AS e FROM audio_files "
                "WHERE conversation_id=?",
                (conv["id"],),
            ).fetchone()
            end = (row["e"] if row else None) or conv["started_at"]
        try:
            gap = parse_iso(cutoff) - parse_iso(end) if end else None
        except ValueError:
            gap = None
        if gap is None or gap > timedelta(minutes=settings.conversation.max_gap_minutes):
            # No parseable timestamp at all → close now rather than leak forever.
            close_conversation(conn, conv["id"], settings)
            closed += 1
    return closed
