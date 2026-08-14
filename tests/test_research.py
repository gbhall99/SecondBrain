import json

import pytest

from secondbrain.llm.client import MockLLM
from secondbrain.tasks import research, store


def test_local_research_stores_note(conn, settings):
    tid = store.create_task(conn, title="what did we decide about pricing")
    note_id = research.run_research(
        conn, tid, web=False, settings=settings,
        researcher=research.LocalResearcher(conn, settings, llm=MockLLM(responses=["A grounded answer."])),
    )
    row = conn.execute("SELECT * FROM task_research WHERE id=?", (note_id,)).fetchone()
    assert row["backend"] == "local"
    assert "grounded" in row["summary_md"]
    assert isinstance(json.loads(row["sources"]), list)


def test_web_research_blocked_when_disabled(conn, settings):
    assert settings.tasks.web_research_enabled is False
    with pytest.raises(RuntimeError, match="web_research_enabled"):
        research.get_researcher(conn, web=True, settings=settings)


def test_mock_researcher(conn, settings):
    tid = store.create_task(conn, title="explore options")
    research.run_research(conn, tid, "explore options", settings=settings,
                          researcher=research.MockResearcher())
    row = conn.execute("SELECT backend FROM task_research WHERE task_id=?", (tid,)).fetchone()
    assert row["backend"] == "mock"


def test_local_sources_carry_day_and_readable_title(conn, settings, monkeypatch):
    from datetime import datetime

    from secondbrain.knowledge import chat

    monkeypatch.setattr(chat, "answer", lambda *a, **k: {
        "answer": "grounded", "citations": [
            {"segment_id": 7, "speaker": "Dana", "start_at": "2026-06-16T09:00:00.000Z"},
            {"segment_id": 8, "speaker": "Me", "start_at": None},
        ]})
    note = research.LocalResearcher(conn, settings).research("q")
    first, second = note.sources
    assert first["ref"] == "seg:7"
    local = datetime.fromisoformat("2026-06-16T09:00:00+00:00").astimezone()
    assert first["day"] == local.strftime("%Y-%m-%d")   # powers /day?date=…#seg-7 links
    assert first["title"] == f"Dana · {local.strftime('%b')} {local.day}, {local.strftime('%H:%M')}"
    assert second["ref"] == "seg:8" and "day" not in second  # unparseable time: no link


def test_run_research_query_appends_task_detail(conn, settings):
    tid = store.create_task(conn, title="Book flights",
                            detail="Sydney in October, aim under $900")
    research.run_research(conn, tid, settings=settings, researcher=research.MockResearcher())
    q = conn.execute("SELECT query FROM task_research WHERE task_id=?", (tid,)).fetchone()["query"]
    assert q == "Book flights — Sydney in October, aim under $900"


def test_query_includes_goal_title_and_counterparty(conn, settings):
    from secondbrain.knowledge import graph

    gid = conn.execute(
        "INSERT INTO goals (title, status) VALUES ('Close the Acme deal', 'active')"
    ).lastrowid
    me = graph.create_node(conn, type="person", name="Me", embedding=None,
                           confidence=1.0, extraction_id=None)
    dana = graph.create_node(conn, type="person", name="Dana", embedding=None,
                             confidence=0.9, extraction_id=None)
    edge = graph.upsert_edge(conn, src_node_id=me, dst_node_id=dana,
                             predicate="action_item", kind="action_item",
                             object_text="send the pricing sheet",
                             source_segment_ids=[1])
    tid = store.create_task(conn, title="send the pricing sheet", goal_id=gid,
                            source="conversation", source_edge_id=edge)
    research.run_research(conn, tid, settings=settings,
                          researcher=research.MockResearcher())
    q = conn.execute(
        "SELECT query FROM task_research WHERE task_id=?", (tid,)
    ).fetchone()["query"]
    # the FULL query is stored, goal + participants included
    assert "send the pricing sheet" in q
    assert "goal: Close the Acme deal" in q
    assert "Dana" in q and "Me" in q


def test_explicit_query_is_stored_verbatim(conn, settings):
    tid = store.create_task(conn, title="whatever")
    research.run_research(conn, tid, "my exact question", settings=settings,
                          researcher=research.MockResearcher())
    q = conn.execute(
        "SELECT query FROM task_research WHERE task_id=?", (tid,)
    ).fetchone()["query"]
    assert q == "my exact question"
