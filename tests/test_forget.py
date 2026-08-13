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


def test_forget_day_prunes_graph_citations(conn, tmp_path):
    import json

    _audio(conn, 1, "2026-06-15", str(tmp_path / "mon.flac"))
    _audio(conn, 2, "2026-06-16", str(tmp_path / "tue.flac"))
    _seg(conn, 1, 1, "2026-06-15", "monday")
    _seg(conn, 2, 2, "2026-06-16", "tuesday")
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (1, 'person', 'A')")
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (2, 'project', 'B')")
    # edge1 grounded only in the forgotten day → dropped
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, source_segment_ids) "
        "VALUES (1, 1, 2, 'fact', ?)",
        (json.dumps([1]),),
    )
    # edge2 cites both days → kept, but citation to the forgotten segment removed
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, source_segment_ids) "
        "VALUES (2, 1, 2, 'fact', ?)",
        (json.dumps([1, 2]),),
    )

    res = service.forget_day(conn, "2026-06-15")

    assert res["kg_edges"] == 1  # edge1 dropped
    assert conn.execute("SELECT COUNT(*) AS n FROM kg_edges").fetchone()["n"] == 1
    remaining = conn.execute("SELECT source_segment_ids FROM kg_edges WHERE id=2").fetchone()
    assert json.loads(remaining["source_segment_ids"]) == [2]  # forgotten cite pruned


def test_forget_day_purges_fully_forgotten_conversation_extractions(conn, tmp_path):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) "
        "VALUES (1, '2026-06-15T09:00:00.000Z', 'diarized')"
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (1, ?, '2026-06-15T09:00:00.000Z', 16000, 'transcribed', 1)",
        (str(tmp_path / "a.flac"),),
    )
    conn.execute("INSERT INTO transcripts (id, audio_file_id, backend) VALUES (1, 1, 'mock')")
    _seg(conn, 1, 1, "2026-06-15", "secret")
    conn.execute(
        "INSERT INTO knowledge_extractions (id, conversation_id, backend, raw_json) "
        "VALUES (1, 1, 'mock', '{\"secret\":\"text\"}')"
    )
    # a node first seen from that extraction (back-reference must be nulled, not error)
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, source_extraction_id) "
        "VALUES (1, 'project', 'Atlas', 1)"
    )

    res = service.forget_day(conn, "2026-06-15")

    assert res["conversations"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM knowledge_extractions").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"] == 0
    # the node survives but its dangling extraction ref was nulled
    node = conn.execute("SELECT source_extraction_id FROM kg_nodes WHERE id=1").fetchone()
    assert node["source_extraction_id"] is None


def test_forget_day_uses_local_day_bounds(conn, tmp_path):
    """'Forget Tuesday' means the LOCAL Tuesday the day view shows, not UTC."""
    import os
    import time

    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"  # UTC-7 in June (PDT)
    time.tzset()
    try:
        _audio(conn, 1, "2026-06-16", str(tmp_path / "a.flac"))
        # 02:00Z on June 16 is 19:00 on June 15 in Los Angeles
        conn.execute(
            "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, "
            "start_offset_s, end_offset_s, start_at, text) "
            "VALUES (1, 1, 1, 0, 1, '2026-06-16T02:00:00.000Z', 'late monday chat')"
        )
        _audio(conn, 2, "2026-06-16", str(tmp_path / "b.flac"))
        _seg(conn, 2, 2, "2026-06-16", "tuesday keeper")  # 09:00Z = June 16 local

        res = service.forget_day(conn, "2026-06-15")

        assert res["segments"] == 1  # the local-Monday segment, despite its UTC date
        remaining = [r["text"] for r in conn.execute("SELECT text FROM transcript_segments")]
        assert remaining == ["tuesday keeper"]
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        time.tzset()


def test_forget_day_purges_undated_segments_via_audio_timestamps(conn, tmp_path):
    _audio(conn, 1, "2026-06-15", str(tmp_path / "mon.flac"))
    # a segment with no start_at of its own must not survive its forgotten day
    conn.execute(
        "INSERT INTO transcript_segments (id, transcript_id, audio_file_id, "
        "start_offset_s, end_offset_s, start_at, text) "
        "VALUES (1, 1, 1, 0, 1, NULL, 'undated monday secret')"
    )
    _audio(conn, 2, "2026-06-16", str(tmp_path / "tue.flac"))
    _seg(conn, 2, 2, "2026-06-16", "tuesday keeper")

    res = service.forget_day(conn, "2026-06-15")

    assert res["segments"] == 1
    remaining = [r["text"] for r in conn.execute("SELECT text FROM transcript_segments")]
    assert remaining == ["tuesday keeper"]
    assert _fts_count(conn, "undated") == 0


def _obs(conn, speaker_id, audio_id, emb):
    from secondbrain.speaker import registry

    conn.execute(
        "INSERT INTO speaker_observations (speaker_id, audio_file_id, embedding) "
        "VALUES (?, ?, ?)",
        (speaker_id, audio_id, registry.serialize_embedding(emb)),
    )


def test_forget_day_refreshes_surviving_speaker_profiles(conn, tmp_path):
    """A voiceprint must stop encoding forgotten audio."""
    from secondbrain.speaker import registry

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Bob', 'known', 0)")
    _audio(conn, 1, "2026-06-15", str(tmp_path / "mon.flac"))
    _audio(conn, 2, "2026-06-16", str(tmp_path / "tue.flac"))
    _seg(conn, 1, 1, "2026-06-15", "monday words", speaker_id=2)
    _seg(conn, 2, 2, "2026-06-16", "tuesday words", speaker_id=2)
    _obs(conn, 2, 1, [1.0, 0.0, 0.0, 0.0])   # forgotten with Monday's audio
    _obs(conn, 2, 2, [0.0, 1.0, 0.0, 0.0])   # survives
    registry.recompute_centroid(conn, 2)     # centroid mixes both days

    res = service.forget_day(conn, "2026-06-15")

    assert res["speakers_refreshed"] == 1
    row = conn.execute(
        "SELECT centroid, exemplar_count, segment_count FROM speakers WHERE id=2"
    ).fetchone()
    vec = registry.deserialize_embedding(row["centroid"])
    assert abs(vec[0]) < 1e-6 and vec[1] > 0.999  # Monday's voice is gone from it
    assert row["exemplar_count"] == 1
    assert row["segment_count"] == 1


def test_forget_person_refreshes_other_speakers_on_shared_audio(conn, tmp_path):
    from secondbrain.speaker import registry

    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Alice', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (3, 'Bob', 'known', 0)")
    _audio(conn, 1, "2026-06-16", str(tmp_path / "a.flac"))
    _seg(conn, 1, 1, "2026-06-16", "alice talking", speaker_id=2)
    # Bob was heard on the same audio (observation) but has no segment on it
    _obs(conn, 3, 1, [1.0, 0.0, 0.0, 0.0])
    _obs(conn, 3, None, [0.0, 1.0, 0.0, 0.0])  # from another source; survives
    registry.recompute_centroid(conn, 3)

    res = service.forget_person(conn, 2)

    assert res["speakers"] == 1
    assert res["speakers_refreshed"] == 1
    vec = registry.deserialize_embedding(
        conn.execute("SELECT centroid FROM speakers WHERE id=3").fetchone()["centroid"]
    )
    assert abs(vec[0]) < 1e-6 and vec[1] > 0.999  # shared-audio obs cascaded away
    # the shared audio's cascade removed Bob's observation on it
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM speaker_observations WHERE speaker_id=3"
    ).fetchone()["n"] == 1


def test_vacuum_runs(conn, tmp_path):
    _audio(conn, 1, "2026-06-16", str(tmp_path / "a.flac"))
    _seg(conn, 1, 1, "2026-06-16", "ephemeral")
    # vacuum requires autocommit; should not raise
    service.forget_day(conn, "2026-06-16", vacuum=True)
    assert conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"] == 0
