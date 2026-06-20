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
