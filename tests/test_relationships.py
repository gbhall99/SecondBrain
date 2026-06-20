"""Relationship intelligence — ranked people + stale-reconnect detector."""

from __future__ import annotations

from datetime import UTC, datetime

from secondbrain.config import Settings
from secondbrain.proactive import detectors
from secondbrain.query import service


def _audio(conn, aid, conv, day):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (conv, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (aid, f"/tmp/{aid}.flac", f"{day}T09:00:00.000Z", conv),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (aid, aid)
    )


def _seg(conn, sid, aid, day, sec, speaker_id, dur=2.0):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id) VALUES (?, ?, ?, ?, ?, ?, 'hi', ?)",
        (sid, aid, aid, sec, sec + dur, f"{day}T09:00:{sec:02d}.000Z", speaker_id),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (3, 'Sam', 'known', 0)")
    # Dana in 2 conversations, Sam in 1
    _audio(conn, 1, 1, "2026-06-16")
    _audio(conn, 2, 2, "2026-06-17")
    _seg(conn, 1, 1, "2026-06-16", 0, 2)
    _seg(conn, 2, 1, "2026-06-16", 2, 3)
    _seg(conn, 3, 2, "2026-06-17", 0, 2)


def test_relationships_ranked(conn, settings):
    _seed(conn)
    rel = service.relationships(conn, settings)
    labels = [r["label"] for r in rel]
    assert labels[0] == "Dana"  # most conversations
    assert "Sam" in labels
    assert "Me" not in labels  # owner excluded
    dana = next(r for r in rel if r["label"] == "Dana")
    assert dana["conversations"] == 2


def test_relationships_excludes_opted_out(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=3")
    labels = [r["label"] for r in service.relationships(conn, settings)]
    assert "Sam" not in labels


def test_stale_relationship_detector(conn):
    _seed(conn)  # last interaction 2026-06-17
    s = Settings(proactive={"reconnect_days": 30})
    now = datetime(2026, 9, 1, tzinfo=UTC)  # well past 30 days
    sugg = detectors.detect_stale_relationships(conn, s, owner_id=1, now=now)
    names = {x.title for x in sugg}
    assert any("Dana" in n for n in names)
    assert all(x.kind == "relationship_reconnect" for x in sugg)


def test_stale_relationship_not_flagged_when_recent(conn):
    _seed(conn)
    s = Settings(proactive={"reconnect_days": 30})
    now = datetime(2026, 6, 18, tzinfo=UTC)  # 1 day later
    assert detectors.detect_stale_relationships(conn, s, owner_id=1, now=now) == []


def test_stale_relationship_skips_opted_out(conn):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    s = Settings(proactive={"reconnect_days": 30})
    now = datetime(2026, 9, 1, tzinfo=UTC)
    names = {x.title for x in detectors.detect_stale_relationships(conn, s, owner_id=1, now=now)}
    assert not any("Dana" in n for n in names)
