"""AI goal decomposition: propose a milestones→tasks tree (you approve)."""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel, Field

from secondbrain.config import Settings, get_settings
from secondbrain.llm.client import LLM, get_llm
from secondbrain.llm.jsonout import parse_json
from secondbrain.tasks import store

_SYSTEM = (
    "You break a goal into an actionable plan. Return JSON: a list of milestones, "
    "each with a short title and concrete sub-tasks (steps). For each task give an "
    "estimate_minutes, effort (1-5), and value (1-5). Be specific and ordered; do "
    "not invent facts about the user."
)


class PlanTask(BaseModel):
    title: str
    detail: str | None = None
    estimate_minutes: int | None = None
    effort: int = 3
    value: int = 3
    energy: str | None = None


class Milestone(BaseModel):
    title: str
    detail: str | None = None
    tasks: list[PlanTask] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    milestones: list[Milestone] = Field(default_factory=list)


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
    goal = conn.execute("SELECT title, description FROM goals WHERE id=?", (goal_id,)).fetchone()
    if goal is None:
        return {"milestones": []}
    prompt = f"Goal: {goal['title']}\nDescription: {goal['description'] or ''}"
    schema = DecompositionResult.model_json_schema()
    resp = llm.complete(system=_SYSTEM, prompt=prompt, schema=schema)
    result = DecompositionResult.model_validate(parse_json(resp.text))
    return result.model_dump()


def accept_plan(conn: sqlite3.Connection, goal_id: int, plan: dict) -> list[int]:
    """Persist a (possibly user-edited) plan tree as tasks. Returns task ids."""
    created: list[int] = []
    result = DecompositionResult.model_validate(plan)
    for ms in result.milestones:
        parent = store.create_task(
            conn, title=ms.title, goal_id=goal_id, detail=ms.detail, source="ai"
        )
        created.append(parent)
        prev: int | None = None
        child_ids: list[int] = []
        for t in ms.tasks:
            tid = store.create_task(
                conn, title=t.title, goal_id=goal_id, parent_task_id=parent,
                detail=t.detail, estimate_minutes=t.estimate_minutes, effort=t.effort,
                value=t.value, energy=t.energy, source="ai",
            )
            created.append(tid)
            child_ids.append(tid)
            # Chain steps so the planner runs them in order (step N after step N-1).
            if prev is not None:
                store.add_dependency(conn, tid, prev)
            prev = tid
        # The milestone container isn't itself work — hold it back until its
        # sub-tasks are done (so it's never proposed as a schedulable item).
        for cid in child_ids:
            store.add_dependency(conn, parent, cid)
    return created
