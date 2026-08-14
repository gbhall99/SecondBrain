"""Prioritisation: Eisenhower quadrant (view) + a weighted score (ordering)."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from secondbrain.config import Settings

# Eisenhower quadrants
DO = "do"               # urgent + important
SCHEDULE = "schedule"   # important, not urgent
DELEGATE = "delegate"   # urgent, not important
ELIMINATE = "eliminate"  # neither
# Quadrant weights scale the NON-urgency part of the score (base value × goal);
# the smooth urgency factor multiplies on top, so a near-due task in a
# down-weighted quadrant still rises as its deadline approaches instead of the
# cliff-y urgent_days boundary fighting a 0.9+ urgency signal.
_QUADRANT_WEIGHT = {DO: 1.0, SCHEDULE: 0.8, DELEGATE: 0.65, ELIMINATE: 0.4}
_PRIORITY_FACTOR = {1: 1.0, 2: 0.7, 3: 0.4}

# Quick wins (effort ≤ 2) get a multiplicative nudge so the bonus scales with
# the task's own importance instead of a flat +0.1 that dwarfs low scores.
QUICK_WIN_FACTOR = 1.15

# Overdue urgency starts at 1.3 and grows mildly with days overdue, capped so
# an ancient zombie task can't drown everything else.
_OVERDUE_BASE = 1.3
_OVERDUE_PER_DAY = 0.02
_OVERDUE_CAP = 1.6


def _as_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_urgent(task: dict, settings: Settings, today: date) -> bool:
    """Due within urgent_days, or overdue."""
    due = _as_date(task.get("due_date"))
    return due is not None and (due - today).days <= settings.tasks.urgent_days


def _goal_priority(conn: sqlite3.Connection, task: dict) -> int | None:
    if not task.get("goal_id"):
        return None
    row = conn.execute("SELECT priority FROM goals WHERE id=?", (task["goal_id"],)).fetchone()
    return int(row["priority"]) if row else None


def _is_important(conn: sqlite3.Connection, task: dict, settings: Settings) -> bool:
    """Graduated importance: high value, priority-1 goal, a medium-priority goal
    paired with decent value, or a task promoted from a spoken commitment."""
    value = task.get("value") or 0
    if value >= settings.tasks.important_value:
        return True
    if task.get("source_edge_id"):
        return True  # you said you'd do it — commitments are important
    priority = _goal_priority(conn, task)
    if priority == 1:
        return True
    return bool(priority == 2 and value >= 3)


def quadrant(conn: sqlite3.Connection, task: dict, settings: Settings, today: date) -> str:
    urgent = _is_urgent(task, settings, today)
    important = _is_important(conn, task, settings)
    if urgent and important:
        return DO
    if important:
        return SCHEDULE
    if urgent:
        return DELEGATE
    return ELIMINATE


def _urgency_factor(task: dict, today: date) -> float:
    due = _as_date(task.get("due_date"))
    if due is None:
        return 0.7
    days = (due - today).days
    if days <= 0:
        return min(_OVERDUE_CAP, _OVERDUE_BASE + _OVERDUE_PER_DAY * (-days))
    return max(0.6, 1.2 - 0.08 * days)


def _goal_factor(conn: sqlite3.Connection, task: dict) -> float:
    if not task.get("goal_id"):
        return 0.6
    priority = _goal_priority(conn, task)
    return _PRIORITY_FACTOR.get(priority if priority is not None else 2, 0.7)


def score_breakdown(
    conn: sqlite3.Connection, task: dict, settings: Settings, today: date
) -> dict:
    """Every factor behind a task's rank, plus the final score.

    ``score = base × goal × quadrant_weight × urgency × quick_win`` — the
    quadrant weight scales the non-urgency part, urgency multiplies on top.
    """
    base = (task.get("value") or 3) / 5.0
    goal = _goal_factor(conn, task)
    q = quadrant(conn, task, settings, today)
    qw = _QUADRANT_WEIGHT[q]
    urgency = _urgency_factor(task, today)
    quick = QUICK_WIN_FACTOR if (task.get("effort") or 3) <= 2 else 1.0
    return {
        "base": round(base, 4),
        "goal": round(goal, 4),
        "quadrant": q,
        "quadrant_weight": qw,
        "urgency": round(urgency, 4),
        "quick_win": quick,
        "score": round(base * goal * qw * urgency * quick, 4),
    }


def score(conn: sqlite3.Connection, task: dict, settings: Settings, today: date) -> float:
    return score_breakdown(conn, task, settings, today)["score"]
