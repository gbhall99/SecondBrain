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
    "goal_alignment": 0.8,
    "connection": 0.6,
    "stale_goal": 0.5,
    "stale_commitment": 0.5,
    "coaching": 0.4,
}
_PRIORITY_FACTOR = {1: 1.0, 2: 0.7, 3: 0.4}


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
    """Score, filter (floor/snooze/suppress), cap per-kind and to top_n."""
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%fZ")
    weights = store.get_feedback_weights(conn)
    snoozed = store.snoozed_kinds(conn, now_iso)
    suppressed = store.suppressed_hashes(conn, settings, now_iso)
    today = now.date()
    cfg = settings.proactive

    scored: list[Suggestion] = []
    for s in suggestions:
        if s.confidence < cfg.confidence_floor:
            continue
        if s.kind in snoozed or s.dedupe_hash in suppressed:
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

    out: list[Suggestion] = []
    per_kind: dict[str, int] = {}
    for s in scored:
        if per_kind.get(s.kind, 0) >= cfg.per_kind_cap:
            continue
        per_kind[s.kind] = per_kind.get(s.kind, 0) + 1
        out.append(s)
        if len(out) >= cfg.top_n:
            break
    return out
