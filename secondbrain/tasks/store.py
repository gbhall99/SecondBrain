"""Task CRUD, dependencies/readiness, and promotion from conversation actions."""

from __future__ import annotations

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


def set_status(conn: sqlite3.Connection, task_id: int, status: str) -> None:
    completed = utcnow_iso() if status == "done" else None
    conn.execute(
        "UPDATE tasks SET status=?, completed_at=?, updated_at=? WHERE id=?",
        (status, completed, utcnow_iso(), task_id),
    )
    if status == "done":
        _bump_goal_progress(conn, task_id)


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
    conn: sqlite3.Connection, edge_id: int, goal_id: int | None = None
) -> int | None:
    """Turn a kg_edges action_item into a task (idempotent per edge)."""
    existing = conn.execute("SELECT id FROM tasks WHERE source_edge_id=?", (edge_id,)).fetchone()
    if existing:
        return int(existing["id"])
    edge = conn.execute(
        "SELECT object_text, due_date FROM kg_edges WHERE id=? AND kind='action_item'", (edge_id,)
    ).fetchone()
    if edge is None:
        return None
    return create_task(
        conn,
        title=edge["object_text"] or "(action item)",
        goal_id=goal_id,
        due_date=edge["due_date"],
        source="conversation",
        source_edge_id=edge_id,
    )
