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


def test_project_dossier_fact_subjects_and_redundant_objects(conn, settings):
    """Facts name their subject; objects that just repeat the project are flagged."""
    _seed(conn)
    d = service.project_dossier(conn, 2, settings)
    works = next(f for f in d["facts"] if f["predicate"] == "works_on")
    # person→project fact: subject exposed (label + person link), object is
    # the project itself so the UI can say "this project" instead of "Atlas".
    assert works["src_label"] == "Dana" and works["src_node_id"] == 1
    assert works["src_speaker_id"] == 2 and works["src_hidden"] is False
    assert works["dst_label"] == "Atlas" and works["dst_node_id"] == 2
    assert works["object_redundant"] is True
    # project→text fact: subject is the project itself, object is real content
    status = next(f for f in d["facts"] if f["predicate"] == "status")
    assert status["src_node_id"] == 2 and status["object_redundant"] is False
    # an alias counts as "just the project's name again" too
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        "confidence, conversation_id, source_segment_ids, valid) "
        "VALUES (7, 3, 2, 'leads', 'fact', 'Project Atlas', 0.7, 1, '[3]', 1)"
    )
    d = service.project_dossier(conn, 2, settings)
    leads = next(f for f in d["facts"] if f["predicate"] == "leads")
    assert leads["src_label"] == "Sam" and leads["object_redundant"] is True
    # action items carry the same subject fields ("Sam — write docs")
    item = next(c for c in d["open_commitments"] if c["object_text"] == "write docs")
    assert item["src_label"] == "Sam" and item["src_node_id"] == 3


def test_project_dossier_dedupes_repeated_facts(conn, settings):
    """The same statement heard twice renders once (highest confidence kept),
    while activity.edges keeps counting every mention."""
    _seed(conn)
    _audio(conn, 3, 3, "2026-06-18")
    _seg(conn, 5, 3, "2026-06-18", 0, "dana still on atlas", 2)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, dst_node_id, predicate, kind, object_text, "
        "confidence, conversation_id, source_segment_ids, valid) "
        "VALUES (8, 1, 2, 'works_on', 'fact', 'Atlas', 0.6, 3, '[5]', 1)"
    )
    d = service.project_dossier(conn, 2, settings)
    works = [f for f in d["facts"] if f["predicate"] == "works_on"]
    assert len(works) == 1
    assert works[0]["confidence"] == 0.9  # the stronger copy survives
    assert d["activity"]["edges"] == 5  # mention count still includes both


def test_project_dossier_quote_cap_reports_total(conn, settings):
    _seed(conn)
    d = service.project_dossier(conn, 2, settings, quotes=2)
    assert len(d["recent_quotes"]) == 2
    assert d["quotes_total"] == 3  # segments 1–3 are all cited and quotable
    d = service.project_dossier(conn, 2, settings)  # default cap not hit
    assert len(d["recent_quotes"]) == 3 and d["quotes_total"] == 3


def test_project_dossier_hides_opted_out_fact_subjects(conn, settings):
    """Opted-out people never leak through the new subject labels."""
    _seed(conn)
    conn.execute("UPDATE speakers SET opted_out=1 WHERE id=2")  # Dana opts out
    d = service.project_dossier(conn, 2, settings)
    works = next(f for f in d["facts"] if f["predicate"] == "works_on")
    assert works["src_hidden"] is True
    assert works["src_label"] is None and works["src_speaker_id"] is None


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


def test_project_dossier_provenance_and_quote_attribution(conn, settings):
    """Every fact/decision/item carries a deep-linkable source; quotes say who spoke."""
    _seed(conn)
    d = service.project_dossier(conn, 2, settings)
    dec = next(x for x in d["decisions"] if x["object_text"] == "ship Atlas v2")
    assert dec["edge_id"] == 3
    assert dec["source_seg"] == 1  # earliest cited segment
    assert dec["source_day"]  # local /day date to link to
    fact = next(f for f in d["facts"] if f["object_text"] == "on track")
    assert fact["source_seg"] == 1 and fact["source_day"]
    item = next(c for c in d["open_commitments"] if c["object_text"] == "write docs")
    assert item["source_seg"] == 3 and item["source_day"]
    q = next(q for q in d["recent_quotes"] if q["text"] == "we should ship Atlas v2")
    assert q["speaker"] == "Me" and q["day"] and q["segment_id"] == 1


def test_action_item_lifecycle(conn, settings):
    """Unpromoted → open; promoted → tracked (still open); task done → closed."""
    _seed(conn)
    d = service.project_dossier(conn, 2, settings)
    item = next(c for c in d["open_commitments"] if c["object_text"] == "write docs")
    assert item["task_id"] is None and item["done"] is False
    assert item["overdue"] is True  # due 2026-06-20, long past
    assert service.list_projects(conn, settings)[0]["open_action_items"] == 1

    tid = service.promote_action_item(conn, item["edge_id"])
    d = service.project_dossier(conn, 2, settings)
    item = next(c for c in d["open_commitments"] if c["object_text"] == "write docs")
    assert item["task_id"] == tid and item["task_status"] == "backlog"
    assert item["done"] is False  # tracked but not finished — still open
    assert service.list_projects(conn, settings)[0]["open_action_items"] == 1

    service.task_set_status(conn, tid, "done")
    d = service.project_dossier(conn, 2, settings)
    item = next(c for c in d["open_commitments"] if c["object_text"] == "write docs")
    assert item["done"] is True and item["task_status"] == "done"
    assert item["overdue"] is False  # finished items are never flagged overdue
    atlas = next(p for p in service.list_projects(conn, settings) if p["label"] == "Atlas")
    assert atlas["open_action_items"] == 0


def test_action_items_sort_open_before_done(conn, settings):
    _seed(conn)
    conn.execute(
        "INSERT INTO kg_edges (id, src_node_id, kind, object_text, conversation_id, "
        "source_segment_ids, valid) VALUES (6, 2, 'action_item', 'book venue', 1, '[1]', 1)"
    )
    tid = service.promote_action_item(conn, 4)  # 'write docs'
    service.task_set_status(conn, tid, "done")
    d = service.project_dossier(conn, 2, settings)
    assert [c["object_text"] for c in d["open_commitments"]] == ["book venue", "write docs"]
    assert [c["done"] for c in d["open_commitments"]] == [False, True]


def test_project_dossier_dedupes_goal_links(conn, settings):
    """Auto-linking plus a manual link to the same goal is still one goal entry,
    displayed with the strongest relation."""
    gid = _seed(conn)  # manual 'advances' link; create_goal may auto-add 'related'
    conn.execute(
        "INSERT OR IGNORE INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, 'node', 2, 'related', 0.5)",
        (gid,),
    )
    d = service.project_dossier(conn, 2, settings)
    mine = [g for g in d["linked_goals"] if g["id"] == gid]
    assert len(mine) == 1 and mine[0]["relation"] == "advances"
    atlas = next(p for p in service.list_projects(conn, settings) if p["label"] == "Atlas")
    assert atlas["linked_goals"] == 1  # distinct goals, not link rows
