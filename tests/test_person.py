"""Person dossier — aggregates identity, interactions, facts, commitments, quotes."""

from __future__ import annotations

from secondbrain.query import service


def _audio(conn, aid, conv, day, path="/tmp/a.flac"):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (conv, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (aid, f"{path}{aid}", f"{day}T09:00:00.000Z", conv),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (aid, aid)
    )


def _seg(conn, sid, aid, day, sec, text, speaker_id, conf=0.95):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id, speaker_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, aid, aid, sec, sec + 2.0, f"{day}T09:00:{sec:02d}.000Z", text, speaker_id, conf),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (3, 'Sam', 'known', 0)")
    _audio(conn, 1, 1, "2026-06-16")
    _seg(conn, 1, 1, "2026-06-16", 0, "me talking", 1)
    _seg(conn, 2, 1, "2026-06-16", 2, "dana talking here", 2)
    _seg(conn, 3, 1, "2026-06-16", 4, "sam talking too", 3)
    # person nodes for Dana + a project; facts + action items
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (1, 'person', 'Dana', 2)"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (2, 'project', 'Atlas')")
    conn.execute(
        "INSERT INTO kg_aliases (node_id, alias) VALUES (1, 'Dana Smith')"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        " confidence, valid) VALUES (1, 1, 2, 'works_on', 'fact', 'Atlas', 0.9, 1)"
    )
    # Dana owes something; something owed to Dana
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, due_date, valid) "
        "VALUES (2, 1, 'action_item', 'send the deck', '2026-06-20', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, valid) "
        "VALUES (3, 2, 1, 'action_item', 'review by Dana', 1)"
    )


def test_dossier_aggregates(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    assert d["label"] == "Dana"
    assert d["node_id"] == 1
    assert "Dana Smith" in d["aliases"]
    assert d["interactions"]["segments"] == 1
    assert d["interactions"]["conversations"] == 1
    assert d["interactions"]["talk_minutes"] == round(2.0 / 60, 1)
    assert any(f["object_text"] == "Atlas" for f in d["facts"])
    assert any(c["object_text"] == "send the deck" for c in d["commitments"]["owed_by"])
    assert any(c["object_text"] == "review by Dana" for c in d["commitments"]["owed_to"])
    quotes = [q["text"] for q in d["recent_quotes"]]
    assert "dana talking here" in quotes


def test_dossier_connections(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 2, settings)
    labels = {c["label"] for c in d["connections"]}
    assert "Me" in labels and "Sam" in labels
    assert 2 not in {c["speaker_id"] for c in d["connections"]}  # not self


def test_dossier_owner(conn, settings):
    _seed(conn)
    d = service.person_dossier(conn, 1, settings)
    assert d["is_owner"] is True
    assert d["label"] == "Me"


def test_dossier_opted_out_hides_content(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")
    d = service.person_dossier(conn, 2, settings)
    assert d["opted_out"] is True
    assert d["facts"] == []
    assert d["recent_quotes"] == []
    assert d["commitments"] == {"owed_by": [], "owed_to": []}
    # identity/interaction shape still present
    assert d["interactions"]["segments"] == 1


def test_dossier_resolves_merged_speaker(conn, settings):
    _seed(conn)
    conn.execute("INSERT INTO speakers (id, name, kind, merged_into) VALUES (9, 'Dana?', 'unknown', 2)")
    d = service.person_dossier(conn, 9, settings)
    assert d["speaker_id"] == 2
    assert d["label"] == "Dana"


def test_dossier_missing_speaker_returns_none(conn, settings):
    assert service.person_dossier(conn, 999, settings) is None
