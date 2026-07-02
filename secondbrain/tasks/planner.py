"""Daily planner: propose a capacity-fitted Today list (you approve)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.tasks import prioritize, store

_DEFAULT_TASK_MINUTES = 30


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def propose_day(
    conn: sqlite3.Connection,
    date: str | None = None,
    capacity_minutes: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Build (and store as 'proposed') a capacity-fitted ranked Today plan."""
    settings = settings or get_settings()
    date = date or _today()
    capacity = capacity_minutes or settings.tasks.daily_capacity_minutes
    today = datetime.strptime(date, "%Y-%m-%d").date()

    ranked = sorted(
        store.ready_tasks(conn),
        key=lambda t: prioritize.score(conn, t, settings, today),
        reverse=True,
    )
    chosen: list[int] = []
    used = 0
    for t in ranked:
        est = t.get("estimate_minutes") or _DEFAULT_TASK_MINUTES
        if used + est > capacity and chosen:
            continue
        chosen.append(t["id"])
        used += est
        if used >= capacity:
            break

    conn.execute(
        """
        INSERT INTO day_plans (date, capacity_minutes, status, task_ids)
        VALUES (?, ?, 'proposed', ?)
        ON CONFLICT(date) DO UPDATE SET
            capacity_minutes=excluded.capacity_minutes,
            status='proposed', task_ids=excluded.task_ids
        """,
        (date, capacity, json.dumps(chosen)),
    )
    return get_day(conn, date)


def accept_day(conn: sqlite3.Connection, date: str | None = None) -> dict | None:
    date = date or _today()
    plan = get_day(conn, date)
    if plan is None:
        return None
    for tid in plan["task_ids"]:
        # Don't clobber an in-progress task back to 'scheduled'; just set the day.
        conn.execute(
            "UPDATE tasks SET scheduled_for=?, "
            "status=CASE WHEN status='in_progress' THEN status ELSE 'scheduled' END, "
            "updated_at=? WHERE id=? AND status NOT IN ('done','dropped')",
            (date, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%fZ"), tid),
        )
    conn.execute("UPDATE day_plans SET status='accepted' WHERE date=?", (date,))
    return get_day(conn, date)


def get_day(conn: sqlite3.Connection, date: str | None = None) -> dict | None:
    date = date or _today()
    row = conn.execute("SELECT * FROM day_plans WHERE date=?", (date,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["task_ids"] = json.loads(d["task_ids"] or "[]")
    d["tasks"] = [store.get_task(conn, tid) for tid in d["task_ids"]]
    return d
