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
    conv = conn.execute("SELECT status FROM conversations WHERE id=?", (cid,)).fetchone()
    assert conv["status"] == "diarized"  # marked done, not left open
    af = conn.execute("SELECT retention_delete_after FROM audio_files WHERE id=?", (a,)).fetchone()
    assert af["retention_delete_after"] is not None  # retention finalized despite skip


def test_normal_conversation_enqueues_diarization(conn, settings):
    settings.conversation.min_conversation_seconds = 5.0
    a = _chunk(conn, 1, "2026-06-16T10:00:00.000Z", "2026-06-16T10:01:00.000Z")  # 60s
    cid = conversation.assign_chunk(conn, a, settings)
    assert conversation.close_conversation(conn, cid, settings) is not None
    assert _diarize_jobs(conn) == 1
    assert conn.execute(
        "SELECT status FROM conversations WHERE id=?", (cid,)
    ).fetchone()["status"] == "closed"
