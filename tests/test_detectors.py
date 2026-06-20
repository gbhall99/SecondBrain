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
