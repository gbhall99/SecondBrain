"""AI goal decomposition: propose a milestones→tasks tree (you approve)."""

from __future__ import annotations

import sqlite3
from datetime import date as _date
from datetime import datetime

from pydantic import BaseModel, Field

from secondbrain.config import Settings, get_settings
from secondbrain.llm.client import LLM, get_llm
from secondbrain.llm.jsonout import complete_json
from secondbrain.tasks import store

_SYSTEM = (
    "You break a goal into an actionable plan. Return JSON: a list of milestones, "
    "each with a short title and concrete sub-tasks (steps). For each task give an "
    "estimate_minutes, effort (1-5), and value (1-5); you may set a due_date "
    "(YYYY-MM-DD) working backwards from the goal's target date. Be specific and "
    "ordered; do not invent facts about the user, and do not repeat tasks the "
    "user already has."
)

# How much existing context feeds the prompt (titles are short; keep it bounded).
_MAX_CONTEXT_LINKS = 5
_MAX_EXISTING_TITLES = 20


class PlanTask(BaseModel):
    title: str
    detail: str | None = None
    estimate_minutes: int | None = None
    effort: int = 3
    value: int = 3
    energy: str | None = None
    due_date: str | None = None


class Milestone(BaseModel):
    title: str
    detail: str | None = None
    tasks: list[PlanTask] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    milestones: list[Milestone] = Field(default_factory=list)


def _valid_day(s: str | None) -> str | None:
    if not s:
        return None
    try:
        datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return s[:10]


def _goal_prompt(conn: sqlite3.Connection, goal_id: int) -> str | None:
    """Prompt grounded in what we know: target date, linked context from
    conversations, and existing open tasks (so the model doesn't repeat them)."""
    goal = conn.execute(
        "SELECT title, description, target_date FROM goals WHERE id=?", (goal_id,)
    ).fetchone()
    if goal is None:
        return None
    lines = [f"Goal: {goal['title']}", f"Description: {goal['description'] or ''}"]
    if goal["target_date"]:
        lines.append(
            f"Target date: {goal['target_date']} (today is {_date.today().isoformat()})"
        )
    links = conn.execute(
        """
        SELECT CASE gl.kind WHEN 'node' THEN COALESCE(n.display_label, n.name)
               ELSE e.object_text END AS label
        FROM goal_links gl
        LEFT JOIN kg_nodes n ON gl.kind = 'node' AND n.id = gl.ref_id
        LEFT JOIN kg_edges e ON gl.kind = 'edge' AND e.id = gl.ref_id
        WHERE gl.goal_id = ? ORDER BY gl.score DESC, gl.id LIMIT ?
        """,
        (goal_id, _MAX_CONTEXT_LINKS),
    ).fetchall()
    labels = [r["label"] for r in links if r["label"]]
    if labels:
        lines.append("Related context from the user's conversations: " + "; ".join(labels))
    existing = conn.execute(
        "SELECT title FROM tasks WHERE goal_id=? AND status NOT IN ('done','dropped') "
        "ORDER BY id LIMIT ?",
        (goal_id, _MAX_EXISTING_TITLES),
    ).fetchall()
    titles = [r["title"] for r in existing]
    if titles:
        lines.append("Existing open tasks (don't repeat these): " + "; ".join(titles))
    return "\n".join(lines)


def propose_plan(
    conn: sqlite3.Connection,
    goal_id: int,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> dict:
    """Ask the LLM for a plan. Returns the parsed tree WITHOUT committing it."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    prompt = _goal_prompt(conn, goal_id)
    if prompt is None:
        return {"milestones": []}
    schema = DecompositionResult.model_json_schema()
    result = DecompositionResult.model_validate(
        complete_json(llm, system=_SYSTEM, prompt=prompt, schema=schema)
    )
    return result.model_dump()


def accept_plan(conn: sqlite3.Connection, goal_id: int, plan: dict) -> list[int]:
    """Persist a (possibly user-edited) plan tree as tasks. Returns NEW task ids.

    Idempotent per title: milestones/steps whose titles already exist under
    this goal are skipped (accepting the same plan twice creates nothing), and
    a milestone with zero kept steps is not created as a schedulable task.
    """
    created: list[int] = []
    result = DecompositionResult.model_validate(plan)
    existing = {
        (t["title"] or "").strip().lower(): int(t["id"])
        for t in store.list_tasks(conn, goal_id=goal_id)
    }
    for ms in result.milestones:
        if not ms.tasks:
            continue  # a milestone with no kept steps isn't schedulable work
        new_tasks = [
            t for t in ms.tasks if (t.title or "").strip().lower() not in existing
        ]
        ms_key = (ms.title or "").strip().lower()
        parent = existing.get(ms_key)
        if parent is None:
            if not new_tasks:
                continue  # every step already exists — no empty container
            parent = store.create_task(
                conn, title=ms.title, goal_id=goal_id, detail=ms.detail, source="ai"
            )
            created.append(parent)
            existing[ms_key] = parent
        prev: int | None = None
        child_ids: list[int] = []
        for t in new_tasks:
            tid = store.create_task(
                conn, title=t.title, goal_id=goal_id, parent_task_id=parent,
                detail=t.detail, estimate_minutes=t.estimate_minutes, effort=t.effort,
                value=t.value, energy=t.energy, due_date=_valid_day(t.due_date),
                source="ai",
            )
            created.append(tid)
            child_ids.append(tid)
            existing[(t.title or "").strip().lower()] = tid
            # Chain steps so the planner runs them in order (step N after step N-1).
            if prev is not None:
                store.add_dependency(conn, tid, prev)
            prev = tid
        # The milestone container isn't itself work — hold it back until its
        # sub-tasks are done (so it's never proposed as a schedulable item).
        for cid in child_ids:
            store.add_dependency(conn, parent, cid)
    return created
