"""Memory timeline — day grouped into conversations with inline extractions."""

from __future__ import annotations

from secondbrain.query import service


def _conv(conn, cid, day):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (cid, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (cid, f"/tmp/{cid}.flac", f"{day}T09:00:00.000Z", cid),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (cid, cid)
    )


def _seg(conn, sid, aid, day, sec, text, speaker_id):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, aid, aid, sec, sec + 2.0, f"{day}T09:00:{sec:02d}.000Z", text, speaker_id),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    _conv(conn, 1, "2026-06-16")
    _seg(conn, 1, 1, "2026-06-16", 0, "kick off the project", 1)
    _seg(conn, 2, 1, "2026-06-16", 2, "sounds good", 2)
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (1, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, conversation_id, valid, "
        " source_segment_ids) VALUES (1, 1, 'decision', 'start Atlas', 1, 1, '[1]')"
    )


def test_timeline_groups_conversation_with_extractions(conn, settings):
    _seed(conn)
    tl = service.timeline(conn, "2026-06-16", settings)
    assert len(tl) == 1
    block = tl[0]
    assert block["conversation_id"] == 1
    assert block["participants"] == ["Dana", "Me"]
    assert [s["text"] for s in block["segments"]] == ["kick off the project", "sounds good"]
    assert block["extractions"]["decision"][0]["object_text"] == "start Atlas"
    assert block["extractions"]["decision"][0]["segment_ids"] == [1]


def test_timeline_empty_day(conn, settings):
    _seed(conn)
    assert service.timeline(conn, "2026-01-01", settings) == []


def test_timeline_opt_out_filtered(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    tl = service.timeline(conn, "2026-06-16", settings)
    texts = [s["text"] for s in tl[0]["segments"]]
    assert "sounds good" not in texts  # Dana opted out
    assert "kick off the project" in texts
    assert "Dana" not in tl[0]["participants"]
