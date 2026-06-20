"""Data "forget" — purge person/day/range and confirm nothing lingers."""

from __future__ import annotations

import pytest

from secondbrain.query import service


def _audio(conn, aid, day, path):
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status) "
        "VALUES (?, ?, ?, 16000, 'transcribed')",
        (aid, path, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')",
        (aid, aid),
    )


def _seg(conn, sid, aid, day, text, speaker_id=None):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, "
        " start_at, text, speaker_id) VALUES (?, ?, ?, 0, 1, ?, ?, ?)",
        (sid, aid, aid, f"{day}T09:00:0{sid}.000Z", text, speaker_id),
    )


def _fts_count(conn, term):
    return conn.execute(
        "SELECT COUNT(*) AS n FROM transcript_segments_fts WHERE transcript_segments_fts MATCH ?",
        (term,),
    ).fetchone()["n"]


def test_forget_day_removes_segments_fts_and_audio(conn, tmp_path):
    f1 = tmp_path / "mon.flac"
    f1.write_bytes(b"x")
    _audio(conn, 1, "2026-06-15", str(f1))
    _audio(conn, 2, "2026-06-16", str(tmp_path / "tue.flac"))
    _seg(conn, 1, 1, "2026-06-15", "monday secret")
    _seg(conn, 2, 2, "2026-06-16", "tuesday keeper")

    res = service.forget_day(conn, "2026-06-15")

    assert res["segments"] == 1
    assert res["audio_files"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM audio_files").fetchone()["n"] == 1
    assert _fts_count(conn, "monday") == 0  # FTS trigger kept index in sync
    assert _fts_count(conn, "tuesday") == 1
    assert not f1.exists()  # raw audio removed from disk


def test_forget_range_inclusive(conn, tmp_path):
    for i, day in enumerate(["2026-06-14", "2026-06-15", "2026-06-16"], start=1):
        _audio(conn, i, day, str(tmp_path / f"{i}.flac"))
        _seg(conn, i, i, day, f"day {i}")

    res = service.forget_range(conn, "2026-06-15", "2026-06-16")

    assert res["segments"] == 2
    remaining = [r["text"] for r in conn.execute("SELECT text FROM transcript_segments")]
    assert remaining == ["day 1"]


def test_forget_person_removes_profile_segments_and_graph(conn, tmp_path):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)"
    )
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Alice', 'known', 0)"
    )
    _audio(conn, 1, "2026-06-16", str(tmp_path / "a.flac"))
    _seg(conn, 1, 1, "2026-06-16", "me talking", speaker_id=1)
    _seg(conn, 2, 1, "2026-06-16", "alice talking", speaker_id=2)
    conn.execute("INSERT INTO speaker_observations (speaker_id, audio_file_id) VALUES (2, 1)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (1, 'person', 'Alice', 2)"
    )
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name) VALUES (2, 'project', 'Atlas')"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind) VALUES (1, 1, 2, 'mention')"
    )

    res = service.forget_person(conn, 2)

    assert res["speakers"] == 1
    assert res["kg_nodes"] == 1
    assert res["segments"] == 1  # only Alice's segment
    assert conn.execute("SELECT COUNT(*) AS n FROM speakers").fetchone()["n"] == 1  # owner kept
    assert conn.execute("SELECT COUNT(*) AS n FROM speaker_observations").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_nodes").fetchone()["n"] == 1  # Atlas kept
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_edges").fetchone()["n"] == 0  # cascaded
    assert _fts_count(conn, "alice") == 0


def test_forget_person_refuses_owner(conn):
    conn.execute(
        "INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)"
    )
    with pytest.raises(ValueError):
        service.forget_person(conn, 1)


def test_vacuum_runs(conn, tmp_path):
    _audio(conn, 1, "2026-06-16", str(tmp_path / "a.flac"))
    _seg(conn, 1, 1, "2026-06-16", "ephemeral")
    # vacuum requires autocommit; should not raise
    service.forget_day(conn, "2026-06-16", vacuum=True)
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"] == 0
