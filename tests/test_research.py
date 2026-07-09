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
