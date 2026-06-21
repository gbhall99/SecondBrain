"""Project intelligence (Phase 9) — list_projects ranking + project_dossier."""

from __future__ import annotations

from secondbrain.query import service


def _audio(conn, aid, conv, day):
    conn.execute(
        "INSERT INTO conversations (id, started_at, status) VALUES (?, ?, 'diarized')",
        (conv, f"{day}T09:00:00.000Z"),
    )
    conn.execute(
        "INSERT INTO audio_files (id, path, started_at, sample_rate, status, conversation_id) "
        "VALUES (?, ?, ?, 16000, 'transcribed', ?)",
        (aid, f"/tmp/a{aid}.flac", f"{day}T09:00:00.000Z", conv),
    )
    conn.execute(
        "INSERT INTO transcripts (id, audio_file_id, backend) VALUES (?, ?, 'mock')", (aid, aid)
    )


def _seg(conn, sid, aid, day, sec, text, speaker_id):
    conn.execute(
        "INSERT INTO transcript_segments "
        "(id, transcript_id, audio_file_id, start_offset_s, end_offset_s, start_at, text, "
        " speaker_id, speaker_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.95)",
        (sid, aid, aid, sec, sec + 2.0, f"{day}T09:00:{sec:02d}.000Z", text, speaker_id),
    )


def _seed(conn):
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (2, 'Dana', 'known', 0)")
    conn.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (3, 'Sam', 'known', 0)")
    _audio(conn, 1, 1, "2026-06-16")
    _seg(conn, 1, 1, "2026-06-16", 0, "we should ship Atlas v2", 1)
    _seg(conn, 2, 1, "2026-06-16", 2, "dana on atlas", 2)
    _seg(conn, 3, 1, "2026-06-16", 4, "sam writes the docs", 3)
    _audio(conn, 2, 2, "2026-06-17")
    _seg(conn, 4, 2, "2026-06-17", 0, "beacon mention", 1)

    # nodes: people + two projects (+ an alias for Atlas)
    conn.execute("INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (1, 'person', 'Dana', 2)")
    conn.execute("INSERT INTO kg_nodes (id, type, name, speaker_id) VALUES (3, 'person', 'Sam', 3)")
    conn.execute(
        "INSERT INTO kg_nodes (id, type, name, first_seen, last_seen) "
        "VALUES (2, 'project', 'Atlas', '2026-06-16', '2026-06-17')"
    )
    conn.execute("INSERT INTO kg_nodes (id, type, name) VALUES (4, 'project', 'Beacon')")
    conn.execute("INSERT INTO kg_aliases (node_id, alias) VALUES (2, 'Project Atlas')")

    # Atlas edges (all conversation 1): person→project fact, project fact, decision, action item
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        "confidence, conversation_id, source_segment_ids, valid) "
        "VALUES (1, 1, 2, 'works_on', 'fact', 'Atlas', 0.9, 1, '[2]', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, predicate, kind, object_text, "
        "confidence, conversation_id, source_segment_ids, valid) "
        "VALUES (2, 2, 'status', 'fact', 'on track', 0.8, 1, '[1]', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, conversation_id, "
        "source_segment_ids, valid) VALUES (3, 2, 'decision', 'ship Atlas v2', 1, '[1]', 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, due_date, "
        "conversation_id, source_segment_ids, valid) "
        "VALUES (4, 3, 2, 'action_item', 'write docs', '2026-06-20', 1, '[3]', 1)"
    )
    # Beacon: a single edge in conversation 2
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, kind, object_text, "
        "conversation_id, source_segment_ids, valid) "
        "VALUES (5, 1, 4, 'fact', 'Beacon', 2, '[4]', 1)"
    )

    # link a goal to Atlas
    gid = service.create_goal(conn, title="Ship Atlas", priority=1)
    conn.execute(
        "INSERT INTO goal_links (goal_id, kind, ref_id, relation, score) VALUES (?, 'node', 2, "
        "'advances', 0.9)",
        (gid,),
    )
    return gid


def test_list_projects_ranks_by_activity(conn, settings):
    _seed(conn)
    projects = service.list_projects(conn, settings)
    labels = [p["label"] for p in projects]
    assert labels[0] == "Atlas"  # more edges than Beacon
    atlas = projects[0]
    assert atlas["conversations"] == 1
    assert atlas["edges"] == 4
    assert atlas["linked_goals"] == 1
    assert atlas["open_action_items"] == 1
    assert "Beacon" in labels


def test_project_dossier_aggregates(conn, settings):
    gid = _seed(conn)
    d = service.project_dossier(conn, 2, settings)
    assert d["label"] == "Atlas"
    assert "Project Atlas" in d["aliases"]
    assert d["activity"]["conversations"] == 1
    assert d["activity"]["edges"] == 4
    assert {p["label"] for p in d["people"]} == {"Dana", "Sam"}
    assert any(g["id"] == gid for g in d["linked_goals"])
    assert any(x["object_text"] == "ship Atlas v2" for x in d["decisions"])
    assert any(f["object_text"] == "on track" for f in d["facts"])
    assert any(c["object_text"] == "write docs" for c in d["open_commitments"])
    quotes = {q["text"] for q in d["recent_quotes"]}
    assert "we should ship Atlas v2" in quotes


def test_project_dossier_opt_out_filters_people_and_quotes(conn, settings):
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")  # Dana opts out
    d = service.project_dossier(conn, 2, settings)
    labels = {p["label"] for p in d["people"]}
    assert "Dana" not in labels and "Sam" in labels
    quotes = {q["text"] for q in d["recent_quotes"]}
    assert "dana on atlas" not in quotes  # Dana's cited segment hidden


def test_project_dossier_resolves_merged_node(conn, settings):
    _seed(conn)
    conn.execute("INSERT INTO kg_nodes (id, type, name, merged_into) VALUES (9, 'project', 'Atlas?', 2)")
    d = service.project_dossier(conn, 9, settings)
    assert d["node_id"] == 2 and d["label"] == "Atlas"


def test_project_dossier_rejects_non_project(conn, settings):
    _seed(conn)
    assert service.project_dossier(conn, 1, settings) is None  # node 1 is a person
    assert service.project_dossier(conn, 999, settings) is None
