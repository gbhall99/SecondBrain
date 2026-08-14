from datetime import UTC, datetime, timedelta

from secondbrain.knowledge import graph
from secondbrain.proactive import detectors
from secondbrain.speaker import registry

NOW = datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC)


def _owner_node(conn):
    owner_spk = registry.get_or_create_owner(conn, "Me")
    return graph.create_node(conn, type="person", name="Me", embedding=None,
                            confidence=1.0, extraction_id=None, speaker_id=owner_spk)


def _person(conn, name):
    return graph.create_node(conn, type="person", name=name, embedding=None,
                             confidence=0.9, extraction_id=None)


def test_commitment_owed_and_overdue(conn, settings):
    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    tomorrow = (NOW + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    # owner owes Dana, due tomorrow
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="send deck", due_date=tomorrow,
                      confidence=0.9, source_segment_ids=[1])
    # Dana owes owner, overdue
    graph.upsert_edge(conn, src_node_id=dana, dst_node_id=owner, predicate="action_item",
                      kind="action_item", object_text="send report", due_date=yesterday,
                      confidence=0.9, source_segment_ids=[2])
    out = detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW)
    kinds = sorted(s.kind for s in out)
    assert kinds == ["commitment_overdue", "commitment_owed"]


def test_owner_own_overdue_commitment_surfaces(conn, settings):
    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    yesterday = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    # The OWNER owes Dana, and it's overdue — must surface, not silently drop.
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="send deck", due_date=yesterday,
                      confidence=0.9, source_segment_ids=[1])
    out = detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW)
    over = [s for s in out if s.kind == "commitment_overdue"]
    assert len(over) == 1 and over[0].title.startswith("You still owe")


def test_goal_alignment_honors_recency_window(conn, settings):
    gid = conn.execute(
        "INSERT INTO goals (title, status) VALUES ('Improve onboarding','active')"
    ).lastrowid
    three_days_ago = (NOW - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%fZ")
    edge = graph.upsert_edge(conn, src_node_id=_owner_node(conn), dst_node_id=None,
                             predicate="idea", kind="idea", object_text="onboarding flow",
                             source_segment_ids=[3], when=three_days_ago)
    conn.execute(
        "INSERT INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, 'edge', ?, 'related', 0.9)", (gid, edge),
    )
    # Default recent_days=1: a 3-day-old edge is outside the window.
    settings.proactive.recent_days = 1
    assert not detectors.detect_goal_alignment(conn, settings, owner_id=None, now=NOW)
    # Weekly widening (recent_days=7): now within the window.
    settings.proactive.recent_days = 7
    out = detectors.detect_goal_alignment(conn, settings, owner_id=None, now=NOW)
    assert any(s.kind == "goal_alignment" and s.goal_id == gid for s in out)


def test_connection_detected_by_keyword(conn, settings):
    recent = graph.create_node(conn, type="topic", name="caching strategy", embedding=None,
                               confidence=0.9, extraction_id=None,
                               when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    old_when = (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%fZ")
    graph.create_node(conn, type="topic", name="caching strategy", embedding=None,
                      confidence=0.9, extraction_id=None, when=old_when)
    out = detectors.detect_connections(conn, settings, owner_id=None, now=NOW)
    assert any(s.kind == "connection" for s in out)
    assert recent  # recent node exists


def test_goal_alignment_advances(conn, settings):
    gid = conn.execute(
        "INSERT INTO goals (title, status) VALUES ('Improve onboarding','active')"
    ).lastrowid
    edge = graph.upsert_edge(conn, src_node_id=_owner_node(conn), dst_node_id=None,
                             predicate="idea", kind="idea", object_text="new onboarding flow",
                             source_segment_ids=[3], when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    conn.execute(
        "INSERT INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, 'edge', ?, 'related', 0.9)", (gid, edge),
    )
    out = detectors.detect_goal_alignment(conn, settings, owner_id=None, now=NOW)
    assert any(s.kind == "goal_alignment" and s.goal_id == gid for s in out)


def test_stale_goal_detected(conn, settings):
    old = (NOW - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%fZ")
    conn.execute(
        "INSERT INTO goals (title, status, created_at) VALUES ('Old goal','active',?)", (old,)
    )
    out = detectors.detect_stale_goals(conn, settings, owner_id=None, now=NOW)
    assert any(s.kind == "stale_goal" for s in out)


def test_dedupe_hash_stable():
    a = detectors.Suggestion(kind="connection", title="x", detail="", confidence=0.9,
                             payload={"key": {"pair": [1, 2]}})
    b = detectors.Suggestion(kind="connection", title="y", detail="z", confidence=0.5,
                             payload={"key": {"pair": [1, 2]}})
    assert a.dedupe_hash == b.dedupe_hash  # same key → same hash regardless of title


# --- fulfilled commitments stop resurfacing (promoted → done/dropped) ---------


def test_fulfilled_commitment_stops_resurfacing(conn, settings):
    from secondbrain.tasks import store as tstore

    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    tomorrow = (NOW + timedelta(days=1)).strftime("%Y-%m-%d")
    edge = graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana,
                             predicate="action_item", kind="action_item",
                             object_text="send deck", due_date=tomorrow,
                             confidence=0.9, source_segment_ids=[1])
    assert detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW)
    tid = tstore.promote_action_item(conn, edge)
    # promoted-but-open commitments still surface — the promise is live
    assert detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW)
    tstore.set_status(conn, tid, "done")
    assert detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW) == []


# --- task/plan detectors -------------------------------------------------------


def test_detect_tasks_due_summarizes_with_overdue_count(conn, settings):
    from secondbrain.tasks import store as tstore

    tstore.create_task(conn, title="overdue thing",
                       due_date=(NOW - timedelta(days=2)).strftime("%Y-%m-%d"))
    tstore.create_task(conn, title="due tomorrow",
                       due_date=(NOW + timedelta(days=1)).strftime("%Y-%m-%d"))
    tstore.create_task(conn, title="far future",
                       due_date=(NOW + timedelta(days=30)).strftime("%Y-%m-%d"))
    done = tstore.create_task(conn, title="finished",
                              due_date=(NOW - timedelta(days=1)).strftime("%Y-%m-%d"))
    tstore.set_status(conn, done, "done")
    out = detectors.detect_tasks_due(conn, settings, owner_id=None, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert s.kind == "tasks_due"
    assert "2 tasks due by tomorrow" in s.title and "(1 overdue)" in s.title
    assert "overdue thing" in s.detail


def test_detect_tasks_due_quiet_when_nothing_due(conn, settings):
    assert detectors.detect_tasks_due(conn, settings, owner_id=None, now=NOW) == []


def test_detect_plan_carryover(conn, settings):
    import json as _json

    from secondbrain.tasks import store as tstore

    a = tstore.create_task(conn, title="slipped A")
    b = tstore.create_task(conn, title="finished B")
    tstore.set_status(conn, b, "done")
    yesterday = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO day_plans (date, capacity_minutes, status, task_ids) "
        "VALUES (?, 240, 'accepted', ?)",
        (yesterday, _json.dumps([a, b])),
    )
    out = detectors.detect_plan_carryover(conn, settings, owner_id=None, now=NOW)
    assert len(out) == 1
    assert out[0].kind == "plan_carryover"
    assert "1 planned item slipped from yesterday" in out[0].title
    assert "slipped A" in out[0].detail


def test_detect_plan_carryover_quiet_without_yesterday_plan(conn, settings):
    assert detectors.detect_plan_carryover(conn, settings, owner_id=None, now=NOW) == []


# --- undated commitments nudge -------------------------------------------------


def test_detect_undated_commitment_nudges_for_a_date(conn, settings):
    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="review the contract",
                      confidence=0.8, source_segment_ids=[4],
                      when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    out = detectors.detect_undated_commitments(conn, settings, owner_id=owner, now=NOW)
    assert len(out) == 1
    assert out[0].kind == "commitment_undated"
    assert "add a due date?" in out[0].title


def test_detect_undated_skips_old_dated_and_others_promises(conn, settings):
    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    old = (NOW - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%fZ")
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="ancient promise", when=old,
                      confidence=0.8)
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="dated promise",
                      due_date="2026-07-01", confidence=0.8,
                      when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    graph.upsert_edge(conn, src_node_id=dana, dst_node_id=owner, predicate="action_item",
                      kind="action_item", object_text="their promise", confidence=0.8,
                      when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    assert detectors.detect_undated_commitments(conn, settings, owner_id=owner, now=NOW) == []


# --- urgency re-admission: due-state is part of the dedupe key -----------------


def test_commitment_dedupe_hash_changes_when_it_goes_overdue(conn, settings):
    owner = _owner_node(conn)
    dana = _person(conn, "Dana")
    tomorrow = (NOW + timedelta(days=1)).strftime("%Y-%m-%d")
    edge = graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana,
                             predicate="action_item", kind="action_item",
                             object_text="send deck", due_date=tomorrow,
                             confidence=0.9, source_segment_ids=[1])
    before = detectors.detect_commitments(conn, settings, owner_id=owner, now=NOW)
    # two days later the same edge is overdue: fresh kind AND fresh hash, so a
    # dismissal of the "due soon" item doesn't suppress the overdue alarm...
    later = NOW + timedelta(days=2)
    after = detectors.detect_commitments(conn, settings, owner_id=owner, now=later)
    assert before[0].dedupe_hash != after[0].dedupe_hash
    assert after[0].kind == "commitment_overdue"
    assert after[0].payload["key"]["state"] == "overdue"
    # ...while the overdue item's own hash is stable, so dismissing IT sticks.
    even_later = NOW + timedelta(days=5)
    again = detectors.detect_commitments(conn, settings, owner_id=owner, now=even_later)
    assert again[0].dedupe_hash == after[0].dedupe_hash
    assert edge  # silence unused warnings


# --- connections: stored embeddings + bounded keyword fallback -----------------


def test_connections_use_stored_node_embeddings(conn, settings):
    emb = registry.serialize_embedding([1.0, 0.0, 0.0])
    recent = graph.create_node(conn, type="topic", name="atlas rollout", embedding=None,
                               confidence=0.9, extraction_id=None,
                               when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    old_when = (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%fZ")
    older = graph.create_node(conn, type="topic", name="zeus launch", embedding=None,
                              confidence=0.9, extraction_id=None, when=old_when)
    conn.execute("UPDATE kg_nodes SET embedding=? WHERE id IN (?, ?)", (emb, recent, older))
    # a citation reachable through an edge touching the recent node
    graph.upsert_edge(conn, src_node_id=recent, dst_node_id=None, predicate="idea",
                      kind="idea", object_text="tie-in", source_segment_ids=[9],
                      when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    out = detectors.detect_connections(conn, settings, owner_id=None, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert sorted(s.payload["nodes"]) == sorted([recent, older])
    assert 9 in s.citations
    assert s.confidence == 1.0  # cosine of identical vectors


def test_connections_keyword_fallback_has_own_threshold(conn, settings):
    # only 1 of 4 distinct words shared → jaccard 0.2 < 0.35 → no suggestion
    graph.create_node(conn, type="topic", name="atlas budget review", embedding=None,
                      confidence=0.9, extraction_id=None,
                      when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    graph.create_node(conn, type="topic", name="atlas offsite", embedding=None,
                      confidence=0.9, extraction_id=None,
                      when=(NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%fZ"))
    out = detectors.detect_connections(conn, settings, owner_id=None, now=NOW)
    assert out == []


# --- goal at risk --------------------------------------------------------------


def test_goal_at_risk_detected_with_numbers(conn, settings):
    from secondbrain.tasks import store as tstore

    due = (NOW + timedelta(days=5)).strftime("%Y-%m-%d")
    gid = conn.execute(
        "INSERT INTO goals (title, status, target_date) VALUES ('Ship v2','active',?)",
        (due,),
    ).lastrowid
    for i in range(3):
        tstore.create_task(conn, title=f"step {i}", goal_id=gid)
    out = detectors.detect_goal_at_risk(conn, settings, owner_id=None, now=NOW)
    assert len(out) == 1
    assert out[0].kind == "goal_at_risk" and out[0].goal_id == gid
    assert "0/3 tasks done" in out[0].title and "in 5 days" in out[0].title


def test_goal_at_risk_quiet_when_on_track_or_far_out(conn, settings):
    from secondbrain.tasks import store as tstore

    near = (NOW + timedelta(days=5)).strftime("%Y-%m-%d")
    far = (NOW + timedelta(days=60)).strftime("%Y-%m-%d")
    on_track = conn.execute(
        "INSERT INTO goals (title, status, target_date) VALUES ('On track','active',?)",
        (near,),
    ).lastrowid
    t1 = tstore.create_task(conn, title="a", goal_id=on_track)
    tstore.set_status(conn, t1, "done")
    conn.execute(
        "INSERT INTO goals (title, status, target_date) VALUES ('Far out','active',?)",
        (far,),
    )
    assert detectors.detect_goal_at_risk(conn, settings, owner_id=None, now=NOW) == []


# --- goal alignment records progress + advances links --------------------------


def test_goal_alignment_marks_progress_and_links_advance(conn, settings):
    gid = conn.execute(
        "INSERT INTO goals (title, status, created_at) VALUES ('Improve onboarding',"
        "'active', ?)",
        ((NOW - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%fZ"),),
    ).lastrowid
    edge = graph.upsert_edge(conn, src_node_id=_owner_node(conn), dst_node_id=None,
                             predicate="decision", kind="decision",
                             object_text="ship new onboarding flow",
                             source_segment_ids=[3],
                             when=NOW.strftime("%Y-%m-%dT%H:%M:%fZ"))
    conn.execute(
        "INSERT INTO goal_links (goal_id, kind, ref_id, relation, score) "
        "VALUES (?, 'edge', ?, 'related', 0.9)", (gid, edge),
    )
    # before: the goal is stale (no progress in 60 days)
    assert detectors.detect_stale_goals(conn, settings, owner_id=None, now=NOW)
    out = detectors.detect_goal_alignment(conn, settings, owner_id=None, now=NOW)
    assert any(s.kind == "goal_alignment" for s in out)
    # progress recorded from the conversation → the stale nag stops
    row = conn.execute("SELECT last_progress_at FROM goals WHERE id=?", (gid,)).fetchone()
    assert row["last_progress_at"]
    assert detectors.detect_stale_goals(conn, settings, owner_id=None, now=NOW) == []
    # the decision edge now advances the goal
    adv = conn.execute(
        "SELECT relation FROM goal_links WHERE goal_id=? AND ref_id=? AND relation='advances'",
        (gid, edge),
    ).fetchone()
    assert adv is not None


# --- reconnect nudges weighted by relationship volume --------------------------


def _seed_voice_with_segments(conn, name, n_segments, last_seen):
    sid = conn.execute(
        "INSERT INTO speakers (name, kind) VALUES (?, 'known')", (name,)
    ).lastrowid
    af = conn.execute(
        "INSERT INTO audio_files (path, started_at, sample_rate) VALUES (?, ?, 16000)",
        (f"{name}.flac", last_seen),
    ).lastrowid
    tr = conn.execute(
        "INSERT INTO transcripts (audio_file_id, backend) VALUES (?, 'mock')", (af,)
    ).lastrowid
    for _ in range(n_segments):
        conn.execute(
            "INSERT INTO transcript_segments (transcript_id, audio_file_id, start_offset_s,"
            " end_offset_s, start_at, text, speaker_id) VALUES (?, ?, 0, 1, ?, 'hi', ?)",
            (tr, af, last_seen, sid),
        )
    return sid


def test_reconnect_confidence_scales_with_volume(conn, settings):
    long_ago = (NOW - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%fZ")
    heavy = _seed_voice_with_segments(conn, "Skip Level", 100, long_ago)
    _seed_voice_with_segments(conn, "One-off Vendor", 2, long_ago)
    out = detectors.detect_stale_relationships(conn, settings, owner_id=None, now=NOW)
    by_name = {s.title: s for s in out}
    assert by_name["Reconnect with Skip Level"].confidence > \
        by_name["Reconnect with One-off Vendor"].confidence
    # cheap shared-topic detail when the person has a graph node with edges
    node = graph.create_node(conn, type="person", name="Skip Level", embedding=None,
                             confidence=0.9, extraction_id=None, speaker_id=heavy)
    topic = graph.create_node(conn, type="topic", name="quarterly planning",
                              embedding=None, confidence=0.9, extraction_id=None)
    graph.upsert_edge(conn, src_node_id=node, dst_node_id=topic, predicate="mention",
                      kind="mention", object_text="quarterly planning",
                      when=long_ago)
    out = detectors.detect_stale_relationships(conn, settings, owner_id=None, now=NOW)
    s = next(x for x in out if x.title == "Reconnect with Skip Level")
    assert "Last discussed: quarterly planning" in s.detail
