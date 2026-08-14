"""Conversation segmentation: span integrity + sub-threshold skip."""

from __future__ import annotations

from secondbrain.pipeline import conversation
from secondbrain.storage import models
from secondbrain.storage.models import AudioFile


def _chunk(conn, aid, start, end):
    return models.insert_audio_file(
        conn,
        AudioFile(path=f"/c{aid}.flac", started_at=start, ended_at=end,
                  sample_rate=16000, status="transcribed"),
    )


def _diarize_jobs(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE type=?", (conversation.JOB_DIARIZE,)
    ).fetchone()["n"]


def test_out_of_order_chunk_keeps_span_correct(conn, settings):
    settings.conversation.max_gap_minutes = 5.0
    # A later chunk opens the conversation first (its retry ran ahead).
    b = _chunk(conn, 2, "2026-06-16T10:05:00.000Z", "2026-06-16T10:06:00.000Z")
    conversation.assign_chunk(conn, b, settings)
    # The retried earlier chunk then joins (gap is negative, within window).
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:01:00.000Z")
    cid = conversation.assign_chunk(conn, a, settings)
    row = conn.execute(
        "SELECT started_at, ended_at FROM conversations WHERE id=?", (cid,)
    ).fetchone()
    assert row["started_at"] == "2026-06-16T10:00:00.000Z"   # earliest
    assert row["ended_at"] == "2026-06-16T10:06:00.000Z"     # latest
    assert row["started_at"] <= row["ended_at"]


def test_subthreshold_conversation_skips_diarization(conn, settings):
    settings.conversation.min_conversation_seconds = 120.0
    settings.consent.raw_audio_retention_hours = 168
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:00:02.000Z")  # 2s
    cid = conversation.assign_chunk(conn, a, settings)
    assert conversation.close_conversation(conn, cid, settings) is None  # no diarize job
    assert _diarize_jobs(conn) == 0
    conv = conn.execute(
        "SELECT status, knowledge_status FROM conversations WHERE id=?", (cid,)
    ).fetchone()
    assert conv["status"] == "diarized"  # marked done, not left open
    # 'skipped' — never a lying 'extracted' — so extraction stats stay honest
    assert conv["knowledge_status"] == "skipped"
    af = conn.execute("SELECT retention_delete_after FROM audio_files WHERE id=?", (a,)).fetchone()
    assert af["retention_delete_after"] is not None  # retention finalized despite skip


def test_normal_conversation_enqueues_diarization(conn, settings):
    settings.diarization.enabled = True
    settings.conversation.min_conversation_seconds = 5.0
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:01:00.000Z")  # 60s
    cid = conversation.assign_chunk(conn, a, settings)
    assert conversation.close_conversation(conn, cid, settings) is not None
    assert _diarize_jobs(conn) == 1
    assert conn.execute(
        "SELECT status FROM conversations WHERE id=?", (cid,)
    ).fetchone()["status"] == "closed"


def test_diarization_disabled_conversation_reaches_extraction(conn, settings):
    """With diarization off, a closed conversation must still become eligible
    for knowledge extraction (status 'diarized' + knowledge_status 'pending')."""
    settings.diarization.enabled = False
    settings.extraction.enabled = True
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:01:00.000Z")  # 60s
    cid = conversation.assign_chunk(conn, a, settings)
    assert conversation.close_conversation(conn, cid, settings) is None  # no diarize job
    assert _diarize_jobs(conn) == 0
    conv = conn.execute(
        "SELECT status, knowledge_status FROM conversations WHERE id=?", (cid,)
    ).fetchone()
    # exactly the state the daemon extraction catch-up query selects
    assert conv["status"] == "diarized" and conv["knowledge_status"] == "pending"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE type='extract_knowledge'"
    ).fetchone()["n"] == 1


def test_max_conversation_length_starts_new_conversation(conn, settings):
    settings.diarization.enabled = True
    settings.conversation.max_gap_minutes = 10.0
    settings.conversation.max_conversation_minutes = 10.0
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:05:00.000Z")
    c1 = conversation.assign_chunk(conn, a, settings)
    # within the gap, but attaching would stretch the conversation to 14 min
    b = _chunk(conn, 2, "2026-06-16T10:08:00.000Z", "2026-06-16T10:14:00.000Z")
    c2 = conversation.assign_chunk(conn, b, settings)
    assert c2 != c1
    assert conn.execute(
        "SELECT status FROM conversations WHERE id=?", (c1,)
    ).fetchone()["status"] == "closed"  # capped conversation was closed for diarization
    assert conn.execute(
        "SELECT ended_at FROM conversations WHERE id=?", (c1,)
    ).fetchone()["ended_at"] == "2026-06-16T10:05:00.000Z"  # span not stretched


def test_late_old_chunk_does_not_extend_current_meeting(conn, settings):
    settings.conversation.max_gap_minutes = 5.0
    # a live meeting is underway
    live = _chunk(conn, 1, "2026-06-16T10:30:00.000Z", "2026-06-16T10:31:00.000Z")
    c_live = conversation.assign_chunk(conn, live, settings)
    # a retried chunk from 90 minutes ago finally lands
    old = _chunk(conn, 2, "2026-06-16T09:00:00.000Z", "2026-06-16T09:01:00.000Z")
    c_old = conversation.assign_chunk(conn, old, settings)
    assert c_old != c_live  # attached to its own conversation, not the meeting
    row = conn.execute(
        "SELECT started_at, ended_at, status FROM conversations WHERE id=?", (c_live,)
    ).fetchone()
    assert row["status"] == "open"                            # meeting not closed
    assert row["started_at"] == "2026-06-16T10:30:00.000Z"    # span not stretched back
    # the next live chunk still lands in the live meeting
    nxt = _chunk(conn, 3, "2026-06-16T10:32:00.000Z", "2026-06-16T10:33:00.000Z")
    assert conversation.assign_chunk(conn, nxt, settings) == c_live


def test_close_stale_handles_null_ended_at(conn, settings):
    settings.diarization.enabled = True
    settings.conversation.max_gap_minutes = 5.0
    # ended_at NULL but a chunk carries the real end → derived, then closed
    cid = conn.execute(
        "INSERT INTO conversations (started_at, ended_at, status) "
        "VALUES ('2026-06-16T09:00:00.000Z', NULL, 'open')"
    ).lastrowid
    af = _chunk(conn, 1, "2026-06-16T09:00:00.000Z", "2026-06-16T09:01:00.000Z")
    conn.execute("UPDATE audio_files SET conversation_id=? WHERE id=?", (cid, af))
    # no timestamps anywhere → must still close (immediately), not leak forever
    empty = conn.execute(
        "INSERT INTO conversations (started_at, ended_at, status) VALUES (NULL, NULL, 'open')"
    ).lastrowid
    assert conversation.close_stale_conversations(conn, settings) == 2
    statuses = {
        r["id"]: r["status"]
        for r in conn.execute("SELECT id, status FROM conversations").fetchall()
    }
    assert statuses[cid] != "open" and statuses[empty] != "open"
