from datetime import UTC, datetime, timedelta

from secondbrain.knowledge import graph
from secondbrain.llm.client import MockLLM
from secondbrain.proactive import engine
from secondbrain.speaker import registry


def _owner(conn):
    spk = registry.get_or_create_owner(conn, "Me")
    return graph.create_node(conn, type="person", name="Me", embedding=None,
                             confidence=1.0, extraction_id=None, speaker_id=spk)


def test_run_digest_persists_suggestions_and_digest(conn, settings):
    settings.proactive.enabled = True
    owner = _owner(conn)
    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="send deck", due_date=tomorrow,
                      confidence=0.9, source_segment_ids=[1])
    d = engine.run_digest(conn, llm=MockLLM(responses=["Your brief: send the deck [1]."]),
                          settings=settings)
    assert d is not None and "deck" in d["summary_md"]
    sugg = conn.execute("SELECT kind FROM suggestions WHERE status='open'").fetchall()
    assert any(s["kind"] == "commitment_owed" for s in sugg)


def test_run_digest_idempotent_same_day(conn, settings):
    settings.proactive.enabled = True
    _owner(conn)
    engine.run_digest(conn, llm=MockLLM(responses=["a", "b"]), settings=settings)
    engine.run_digest(conn, llm=MockLLM(responses=["c"]), settings=settings)
    # one digest row per (date, daily)
    assert conn.execute("SELECT COUNT(*) AS n FROM digests WHERE kind='daily'").fetchone()["n"] == 1


def test_weekly_digest_kind(conn, settings):
    settings.proactive.enabled = True
    _owner(conn)
    engine.run_digest(conn, llm=MockLLM(responses=["weekly review"]), settings=settings, kind="weekly")
    row = conn.execute("SELECT kind FROM digests").fetchone()
    assert row["kind"] == "weekly"


def test_synthesize_drops_hallucinated_citations(conn, settings):
    settings.proactive.enabled = True
    owner = _owner(conn)
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=None, predicate="idea",
                      kind="idea", object_text="an idea", source_segment_ids=[7],
                      when=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%fZ"))
    # goal so the idea is linked + surfaced
    gid = conn.execute("INSERT INTO goals (title,status) VALUES ('x','active')").lastrowid
    eid = conn.execute("SELECT id FROM kg_edges LIMIT 1").fetchone()["id"]
    conn.execute("INSERT INTO goal_links (goal_id,kind,ref_id,relation,score) VALUES (?, 'edge', ?, 'related', 0.9)", (gid, eid))
    d = engine.run_digest(
        conn, llm=MockLLM(responses=["Good progress [7] and also [999]."]), settings=settings
    )
    assert "[999]" not in d["summary_md"]  # hallucinated citation stripped
