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
_QUADRANT_WEIGHT = {DO: 1.0, SCHEDULE: 0.8, DELEGATE: 0.55, ELIMINATE: 0.3}
_PRIORITY_FACTOR = {1: 1.0, 2: 0.7, 3: 0.4}


def _as_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_urgent(task: dict, settings: Settings, today: date) -> bool:
    due = _as_date(task.get("due_date"))
    return due is not None and (due - today).days <= settings.tasks.urgent_days


def _is_important(conn: sqlite3.Connection, task: dict, settings: Settings) -> bool:
    if (task.get("value") or 0) >= settings.tasks.important_value:
        return True
    if task.get("goal_id"):
        row = conn.execute("SELECT priority FROM goals WHERE id=?", (task["goal_id"],)).fetchone()
        if row and row["priority"] == 1:
            return True
    return False


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
        return 1.3
    return max(0.6, 1.2 - 0.08 * days)


def _goal_factor(conn: sqlite3.Connection, task: dict) -> float:
    if not task.get("goal_id"):
        return 0.6
    row = conn.execute("SELECT priority FROM goals WHERE id=?", (task["goal_id"],)).fetchone()
    return _PRIORITY_FACTOR.get(row["priority"] if row else 2, 0.7)


def score(conn: sqlite3.Connection, task: dict, settings: Settings, today: date) -> float:
    base = (task.get("value") or 3) / 5.0
    quick_win = 0.1 if (task.get("effort") or 3) <= 2 else 0.0
    q = quadrant(conn, task, settings, today)
    s = base * _urgency_factor(task, today) * _goal_factor(conn, task) * _QUADRANT_WEIGHT[q]
    return round(s + quick_win, 4)
