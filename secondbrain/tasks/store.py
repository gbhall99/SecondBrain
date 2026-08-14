"""Task CRUD, dependencies/readiness, and promotion from conversation actions."""

from __future__ import annotations

import json
import sqlite3

from secondbrain.storage.models import utcnow_iso

ACTIVE_STATUSES = ("backlog", "next", "scheduled", "in_progress", "blocked")
# Statuses eligible for day planning — 'blocked' is explicitly held back by the user.
SCHEDULABLE_STATUSES = ("backlog", "next", "scheduled", "in_progress")
DONE_STATUSES = ("done", "dropped")


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    goal_id: int | None = None,
    parent_task_id: int | None = None,
    detail: str | None = None,
    estimate_minutes: int | None = None,
    due_date: str | None = None,
    effort: int = 3,
    value: int = 3,
    energy: str | None = None,
    source: str = "manual",
    source_edge_id: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO tasks
            (goal_id, parent_task_id, title, detail, estimate_minutes, due_date,
             effort, value, energy, source, source_edge_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (goal_id, parent_task_id, title, detail, estimate_minutes, due_date,
         effort, value, energy, source, source_edge_id, utcnow_iso()),
    )
    return int(cur.lastrowid)


def update_task(conn: sqlite3.Connection, task_id: int, **fields) -> None:
    allowed = {
        "title", "detail", "estimate_minutes", "due_date", "scheduled_for",
        "effort", "value", "energy", "status", "goal_id", "position",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    sets["updated_at"] = utcnow_iso()
    cols = ", ".join(f"{k}=?" for k in sets)
    conn.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*sets.values(), task_id))


def release_stale_scheduled(conn: sqlite3.Connection, before_day: str) -> int:
    """Send tasks still 'scheduled' for a day before ``before_day`` back to
    the backlog.

    A task accepted into an earlier day's plan but never finished would keep
    a stale 'scheduled' pill forever — contradicting a fresh Today section
    that says there's no plan yet. In-progress and done/dropped tasks are
    left untouched. Each released task records the slip: ``rollover_count``
    increments and ``last_planned_for`` keeps the day it was planned for, so
    the UI can show "slipped ×3". Returns how many tasks were released.
    """
    cur = conn.execute(
        "UPDATE tasks SET status='backlog', last_planned_for=scheduled_for, "
        "rollover_count=rollover_count+1, scheduled_for=NULL, updated_at=? "
        "WHERE status='scheduled' AND scheduled_for IS NOT NULL AND scheduled_for < ?",
        (utcnow_iso(), before_day),
    )
    return cur.rowcount


def set_status(conn: sqlite3.Connection, task_id: int, status: str) -> None:
    completed = utcnow_iso() if status == "done" else None
    conn.execute(
        "UPDATE tasks SET status=?, completed_at=?, updated_at=? WHERE id=?",
        (status, completed, utcnow_iso(), task_id),
    )
    if status == "done":
        _bump_goal_progress(conn, task_id)
    if status in DONE_STATUSES:
        # A finished (or abandoned) task leaves any day plan that still lists
        # it — the Today section shows what's left to do, not history.
        _drop_from_day_plans(conn, task_id)


def _drop_from_day_plans(conn: sqlite3.Connection, task_id: int) -> None:
    for r in conn.execute("SELECT date, task_ids FROM day_plans").fetchall():
        try:
            ids = json.loads(r["task_ids"] or "[]")
        except (TypeError, ValueError):
            continue
        if task_id in ids:
            conn.execute(
                "UPDATE day_plans SET task_ids=? WHERE date=?",
                (json.dumps([t for t in ids if t != task_id]), r["date"]),
            )


def _bump_goal_progress(conn: sqlite3.Connection, task_id: int) -> None:
    row = conn.execute("SELECT goal_id FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row and row["goal_id"]:
        conn.execute(
            "UPDATE goals SET last_progress_at=? WHERE id=?", (utcnow_iso(), row["goal_id"])
        )


def get_task(conn: sqlite3.Connection, task_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_tasks(
    conn: sqlite3.Connection, *, goal_id: int | None = None, status: str | None = None
) -> list[dict]:
    q = "SELECT * FROM tasks WHERE 1=1"
    params: list = []
    if goal_id is not None:
        q += " AND goal_id=?"
        params.append(goal_id)
    if status is not None:
        q += " AND status=?"
        params.append(status)
    q += " ORDER BY position, id"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


# --- dependencies + readiness ------------------------------------------------


def add_dependency(conn: sqlite3.Connection, task_id: int, depends_on_task_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO task_deps (task_id, depends_on_task_id) VALUES (?, ?)",
        (task_id, depends_on_task_id),
    )


def is_ready(conn: sqlite3.Connection, task_id: int) -> bool:
    """A task is ready when it has no incomplete dependencies."""
    rows = conn.execute(
        """
        SELECT t.status FROM task_deps d JOIN tasks t ON t.id = d.depends_on_task_id
        WHERE d.task_id = ?
        """,
        (task_id,),
    ).fetchall()
    return all(r["status"] in DONE_STATUSES for r in rows)


def ready_tasks(conn: sqlite3.Connection) -> list[dict]:
    """Open, unblocked tasks eligible for scheduling (excludes 'blocked')."""
    rows = conn.execute(
        f"SELECT * FROM tasks WHERE status IN ({','.join('?' * len(SCHEDULABLE_STATUSES))})",
        SCHEDULABLE_STATUSES,
    ).fetchall()
    return [dict(r) for r in rows if is_ready(conn, r["id"])]


# --- promotion from conversation action items --------------------------------


def promote_action_item(
    conn: sqlite3.Connection,
    edge_id: int,
    goal_id: int | None = None,
    *,
    title: str | None = None,
) -> int | None:
    """Turn a kg_edges action_item into a task (idempotent per edge).

    ``title`` overrides the task title (the "Chase" flow turns an owed-to-you
    item into a follow-up task instead of copying their work into your list).
    """
    existing = conn.execute("SELECT id FROM tasks WHERE source_edge_id=?", (edge_id,)).fetchone()
    if existing:
        return int(existing["id"])
    edge = conn.execute(
        "SELECT object_text, due_date, due_date_norm FROM kg_edges "
        "WHERE id=? AND kind='action_item'",
        (edge_id,),
    ).fetchone()
    if edge is None:
        return None
    return create_task(
        conn,
        title=title or edge["object_text"] or "(action item)",
        goal_id=goal_id,
        due_date=edge["due_date_norm"] or edge["due_date"],
        source="conversation",
        source_edge_id=edge_id,
    )
