"""Importance scoring + noise control (deterministic; no LLM)."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from secondbrain.config import Settings
from secondbrain.proactive import store
from secondbrain.proactive.detectors import Suggestion, _as_date

BASE_WEIGHT = {
    "commitment_overdue": 1.0,
    "commitment_owed": 0.9,
    "goal_at_risk": 0.9,
    "tasks_due": 0.85,
    "goal_alignment": 0.8,
    "commitment_undated": 0.75,
    "plan_carryover": 0.7,
    "connection": 0.6,
    "stale_goal": 0.5,
    "relationship_reconnect": 0.45,
    "coaching": 0.4,
}
_PRIORITY_FACTOR = {1: 1.0, 2: 0.7, 3: 0.4}

# Commitments are the core need — they get a far looser per-kind cap than the
# nice-to-have kinds when the visible list is cut down at render time.
COMMITMENT_KIND_CAP = 10


def _urgency(s: Suggestion, today) -> float:
    due = _as_date(s.payload.get("due_date"))
    if due is None:
        return 1.0
    days = (due - today).days
    if days <= 0:
        return 1.3            # overdue / due today
    return max(0.6, 1.2 - 0.1 * days)


def _goal_priority(conn: sqlite3.Connection, goal_id: int | None) -> float:
    if goal_id is None:
        return 1.0
    row = conn.execute("SELECT priority FROM goals WHERE id=?", (goal_id,)).fetchone()
    return _PRIORITY_FACTOR.get(row["priority"] if row else 2, 0.7)


def rank(
    conn: sqlite3.Connection,
    suggestions: list[Suggestion],
    settings: Settings,
    *,
    now: datetime,
) -> list[Suggestion]:
    """Score and filter (floor/snooze/suppress), sorted by importance.

    Every scored suggestion is returned (and persisted by the engine); the
    top_n / per-kind display cut happens at render time via :func:`apply_caps`
    so "show more" can reveal the rest without a re-run.
    """
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%fZ")
    weights = store.get_feedback_weights(conn)
    snoozed = store.snoozed_kinds(conn, now_iso)
    snoozed_items = store.snoozed_hashes(conn, now_iso)
    suppressed = store.suppressed_hashes(conn, settings, now_iso)
    today = now.date()
    cfg = settings.proactive

    scored: list[Suggestion] = []
    for s in suggestions:
        if s.confidence < cfg.confidence_floor:
            continue
        if s.kind in snoozed or s.dedupe_hash in snoozed_items or s.dedupe_hash in suppressed:
            continue
        s.importance = round(
            BASE_WEIGHT.get(s.kind, 0.5)
            * _urgency(s, today)
            * max(0.0, min(1.0, s.confidence))
            * _goal_priority(conn, s.goal_id)
            * weights.get(s.kind, 1.0),
            4,
        )
        scored.append(s)

    scored.sort(key=lambda x: x.importance, reverse=True)
    return scored


def _kind_of(s) -> str:
    return s["kind"] if isinstance(s, dict) else s.kind


def apply_caps(items: list, settings: Settings) -> tuple[list, list]:
    """Split an importance-sorted list into (visible, overflow) for display.

    ``top_n`` bounds the visible list; ``per_kind_cap`` bounds each kind within
    it — except commitment kinds, which get :data:`COMMITMENT_KIND_CAP`
    (commitments are the whole point of the brief). Works on Suggestion
    objects and on the dict rows the API serves.
    """
    visible: list = []
    overflow: list = []
    per_kind: dict[str, int] = {}
    for s in items:
        kind = _kind_of(s)
        cap = COMMITMENT_KIND_CAP if kind.startswith("commitment") \
            else settings.proactive.per_kind_cap
        if per_kind.get(kind, 0) >= cap or len(visible) >= settings.proactive.top_n:
            overflow.append(s)
            continue
        per_kind[kind] = per_kind.get(kind, 0) + 1
        visible.append(s)
    return visible, overflow
