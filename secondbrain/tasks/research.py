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
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.llm.client import LLM, get_llm
from secondbrain.storage.models import utcnow_iso

# Detail text appended to auto-built research queries is capped so a very long
# note can't drown the retrieval query the title anchors.
_QUERY_DETAIL_CHARS = 500


def _local_dt(ts: str | None) -> datetime | None:
    """Stored UTC timestamp → aware datetime in the machine's local timezone.

    Sources carry the local calendar day derived from this, matching how the
    Day view buckets segments, so ``/day?date=<day>#seg-<id>`` links built from
    research sources land on the cited moment.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


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
        sources = []
        for c in ans.get("citations", []):
            dt = _local_dt(c.get("start_at"))
            when = (f"{dt.strftime('%b')} {dt.day}, {dt.strftime('%H:%M')}" if dt
                    else (c.get("start_at") or "")[:19])
            src = {"title": f"{c['speaker']} · {when}", "ref": f"seg:{c['segment_id']}"}
            if dt:  # lets the UI link the source to /day?date=<day>#seg-<id>
                src["day"] = dt.strftime("%Y-%m-%d")
            sources.append(src)
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
        if not url.lower().startswith("https://"):
            raise RuntimeError(
                "tasks.web_search_url must be https:// — task titles may be sensitive"
            )
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


def _query_context(conn: sqlite3.Connection, row) -> list[str]:
    """Extra grounding for an auto-built query: the task's goal title and the
    people on the source conversation commitment (via source_edge_id)."""
    extras: list[str] = []
    if row["goal_id"]:
        g = conn.execute("SELECT title FROM goals WHERE id=?", (row["goal_id"],)).fetchone()
        if g and g["title"]:
            extras.append(f"goal: {g['title']}")
    if row["source_edge_id"]:
        e = conn.execute(
            """
            SELECT COALESCE(s.display_label, s.name) AS src_name,
                   COALESCE(d.display_label, d.name) AS dst_name
            FROM kg_edges e
            JOIN kg_nodes s ON s.id = e.src_node_id
            LEFT JOIN kg_nodes d ON d.id = e.dst_node_id
            WHERE e.id = ?
            """,
            (row["source_edge_id"],),
        ).fetchone()
        if e:
            names = [n for n in (e["src_name"], e["dst_name"]) if n]
            if names:
                extras.append("with " + ", ".join(names))
    return extras


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
    """Run research for a task and store the note (the full query is stored
    alongside it, so a note is always auditable). Returns the note id."""
    settings = settings or get_settings()
    if query is None:
        row = conn.execute(
            "SELECT title, detail, goal_id, source_edge_id FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        query = row["title"] if row else ""
        detail = (row["detail"] or "").strip() if row else ""
        if detail:  # context-rich tasks get better-grounded notes
            query = f"{query} — {detail[:_QUERY_DETAIL_CHARS]}"
        extras = _query_context(conn, row) if row else []
        if extras:
            query = f"{query} ({'; '.join(extras)})"
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
