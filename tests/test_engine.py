from datetime import UTC, datetime, timedelta

import pytest

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


def test_run_digest_refuses_overlapping_run(conn, settings):
    from secondbrain.proactive import store

    settings.proactive.enabled = True
    _owner(conn)
    store.mark_generating(conn, "daily")
    with pytest.raises(engine.DigestInFlight):
        engine.run_digest(conn, llm=MockLLM(responses=["x"]), settings=settings)
    store.clear_generating(conn, "daily")  # run finished (or crashed): unblocked
    assert engine.run_digest(conn, llm=MockLLM(responses=["x"]), settings=settings) is not None


def test_run_digest_clears_marker_even_when_llm_fails(conn, settings):
    from secondbrain.proactive import store

    settings.proactive.enabled = True
    owner = _owner(conn)
    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana, predicate="action_item",
                      kind="action_item", object_text="send deck", due_date=tomorrow,
                      confidence=0.9, source_segment_ids=[1])

    class BoomLLM:
        def complete(self, system, prompt):
            raise RuntimeError("model offline")

    with pytest.raises(RuntimeError):
        engine.run_digest(conn, llm=BoomLLM(), settings=settings)
    assert store.generating_since(conn, "daily") is None  # cleared in finally


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


def test_weekly_digest_keyed_to_week_monday(conn, settings):
    settings.proactive.enabled = True
    _owner(conn)
    # two generations in one week (Thursday, then Friday) update ONE row
    engine.run_digest(conn, llm=MockLLM(responses=["w1"]), settings=settings,
                      kind="weekly", date="2026-06-18")
    engine.run_digest(conn, llm=MockLLM(responses=["w2"]), settings=settings,
                      kind="weekly", date="2026-06-19")
    rows = conn.execute(
        "SELECT digest_date, summary_md FROM digests WHERE kind='weekly'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["digest_date"] == "2026-06-15"  # that week's Monday
    assert rows[0]["summary_md"] == "w2"
    assert engine.week_monday("2026-06-15") == "2026-06-15"  # Monday is a fixpoint


def test_weekly_stats_computed_and_stored_in_payload(conn, settings):
    from secondbrain.tasks import store as tstore

    settings.proactive.enabled = True
    _owner(conn)
    done = tstore.create_task(conn, title="finished this week")
    tstore.set_status(conn, done, "done")
    # a planned task that did NOT get done (adherence 50%)
    planned = tstore.create_task(conn, title="planned but skipped")
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    conn.execute("UPDATE tasks SET scheduled_for=? WHERE id IN (?, ?)",
                 (today, done, planned))
    # an hour-long meeting this week
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO conversations (started_at, ended_at, status) VALUES (?, ?, 'closed')",
        ((now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
         (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
    )
    d = engine.run_digest(conn, llm=MockLLM(responses=["weekly"]), settings=settings,
                          kind="weekly")
    stats = d["payload"]["stats"]
    assert stats["tasks_completed"] == 1
    assert stats["meeting_hours"] == 1.0
    assert stats["plan_adherence_pct"] == 50 and stats["planned_tasks"] == 2
    assert {"commitments_added", "commitments_cleared", "goals_progressed"} <= set(stats)


def test_digest_excludes_fulfilled_commitments(conn, settings):
    from secondbrain.tasks import store as tstore

    settings.proactive.enabled = True
    owner = _owner(conn)
    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    edge = graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dana,
                             predicate="action_item", kind="action_item",
                             object_text="send deck", due_date=tomorrow,
                             confidence=0.9, source_segment_ids=[1])
    tid = tstore.promote_action_item(conn, edge)
    tstore.set_status(conn, tid, "done")
    engine.run_digest(conn, llm=MockLLM(responses=["brief"]), settings=settings)
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM suggestions").fetchall()]
    assert "commitment_owed" not in kinds and "commitment_overdue" not in kinds


def test_run_digest_persists_all_scored_beyond_display_caps(conn, settings):
    settings.proactive.enabled = True
    settings.proactive.top_n = 1
    settings.proactive.per_kind_cap = 1
    owner = _owner(conn)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    for i, what in enumerate(["send deck", "book room", "email legal"]):
        dst = graph.create_node(conn, type="person", name=f"P{i}", embedding=None,
                                confidence=0.9, extraction_id=None)
        graph.upsert_edge(conn, src_node_id=owner, dst_node_id=dst,
                          predicate="action_item", kind="action_item",
                          object_text=what, due_date=tomorrow,
                          confidence=0.9, source_segment_ids=[1])
    engine.run_digest(conn, llm=MockLLM(responses=["brief"]), settings=settings)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM suggestions WHERE status='open' "
        "AND kind='commitment_owed'"
    ).fetchone()["n"]
    assert n == 3  # everything scored is persisted; caps are display-only


def test_run_digest_records_relink_high_water_mark(conn, settings):
    from secondbrain.proactive.engine import RELINK_EDGE_KEY
    from secondbrain.storage import state

    settings.proactive.enabled = True
    owner = _owner(conn)
    graph.upsert_edge(conn, src_node_id=owner, dst_node_id=None, predicate="idea",
                      kind="idea", object_text="an idea", source_segment_ids=[7])
    conn.execute("INSERT INTO goals (title, status) VALUES ('g', 'active')")
    engine.run_digest(conn, llm=MockLLM(responses=["x"]), settings=settings)
    max_edge = conn.execute("SELECT MAX(id) FROM kg_edges").fetchone()[0]
    assert state.get_state(conn, RELINK_EDGE_KEY) == str(max_edge)
