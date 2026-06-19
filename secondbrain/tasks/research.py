"""Task research: local graph-RAG by default; opt-in web research.

Local research stays fully offline (it reasons over your own knowledge graph +
transcripts). Web research only runs when ``tasks.web_research_enabled`` is set
AND a search endpoint is configured AND it's explicitly requested — and is
clearly recorded as ``backend='web'`` on the note.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from secondbrain.config import Settings, get_settings
from secondbrain.llm.client import LLM, get_llm
from secondbrain.storage.models import utcnow_iso


@dataclass
class ResearchNote:
    summary_md: str
    backend: str
    sources: list[dict] = field(default_factory=list)


class Researcher(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def research(self, query: str) -> ResearchNote:
        ...


class LocalResearcher(Researcher):
    """Graph-RAG over the user's own data — fully local."""

    backend_name = "local"

    def __init__(self, conn: sqlite3.Connection, settings: Settings, llm: LLM | None = None):
        self.conn = conn
        self.settings = settings
        self.llm = llm

    def research(self, query: str) -> ResearchNote:
        from secondbrain.knowledge import chat

        ans = chat.answer(self.conn, query, llm=self.llm, settings=self.settings)
        sources = [
            {"title": f"{c['speaker']} · {(c['start_at'] or '')[:19]}",
             "ref": f"seg:{c['segment_id']}"}
            for c in ans.get("citations", [])
        ]
        return ResearchNote(summary_md=ans["answer"], backend=self.backend_name, sources=sources)


class MockResearcher(Researcher):
    backend_name = "mock"

    def research(self, query: str) -> ResearchNote:
        return ResearchNote(summary_md=f"[mock research: {query}]", backend=self.backend_name)


class WebResearcher(Researcher):
    """Opt-in web research via a user-configured search endpoint + LLM summary."""

    backend_name = "web"

    def __init__(self, settings: Settings, llm: LLM | None = None):
        self.settings = settings
        self.llm = llm

    def research(self, query: str) -> ResearchNote:
        import httpx  # lazy

        url = self.settings.tasks.web_search_url
        if not url:
            raise RuntimeError("tasks.web_search_url is not configured")
        r = httpx.get(url, params={"q": query, "format": "json"}, timeout=15.0)
        r.raise_for_status()
        results = r.json().get("results", [])[:5]
        sources = [{"title": x.get("title", ""), "ref": x.get("url", "")} for x in results]
        context = "\n".join(f"- {s['title']}: {s['ref']}" for s in sources)
        llm = self.llm or get_llm(self.settings)
        summary = llm.complete(
            system="Summarise these web results for the task; cite source titles.",
            prompt=f"Query: {query}\n{context}",
        ).text
        return ResearchNote(summary_md=summary, backend=self.backend_name, sources=sources)


def get_researcher(
    conn: sqlite3.Connection, *, web: bool = False, settings: Settings | None = None,
    llm: LLM | None = None,
) -> Researcher:
    settings = settings or get_settings()
    if web:
        if not settings.tasks.web_research_enabled:
            raise RuntimeError("web research requested but tasks.web_research_enabled is false")
        return WebResearcher(settings, llm)
    return LocalResearcher(conn, settings, llm)


def run_research(
    conn: sqlite3.Connection,
    task_id: int,
    query: str | None = None,
    *,
    web: bool = False,
    settings: Settings | None = None,
    llm: LLM | None = None,
    researcher: Researcher | None = None,
) -> int:
    """Run research for a task and store the note. Returns the note id."""
    settings = settings or get_settings()
    if query is None:
        row = conn.execute("SELECT title FROM tasks WHERE id=?", (task_id,)).fetchone()
        query = row["title"] if row else ""
    researcher = researcher or get_researcher(conn, web=web, settings=settings, llm=llm)
    note = researcher.research(query)
    cur = conn.execute(
        """
        INSERT INTO task_research (task_id, query, backend, summary_md, sources, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (task_id, query, note.backend, note.summary_md, json.dumps(note.sources), utcnow_iso()),
    )
    return int(cur.lastrowid)
