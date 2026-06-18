"""Goal CRUD. Goals are first-class user-authored objects (kept out of the KG so
extraction/merge stays purely transcript-derived)."""

from __future__ import annotations

import sqlite3

from secondbrain.config import Settings, get_settings
from secondbrain.search import semantic
from secondbrain.speaker import registry
from secondbrain.storage.models import utcnow_iso


def _embed(title: str, description: str | None, settings: Settings) -> bytes | None:
    embedder = semantic.get_embedder(settings)
    if embedder is None:
        return None
    try:
        vec = embedder.encode([f"{title}\n{description or ''}"])[0]
        return registry.serialize_embedding(vec)
    except Exception:  # noqa: BLE001 - embedding is best-effort
        return None


def create_goal(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: str | None = None,
    target_date: str | None = None,
    priority: int = 2,
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    cur = conn.execute(
        """
        INSERT INTO goals (title, description, target_date, priority, embedding, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (title, description, target_date, priority,
         _embed(title, description, settings), utcnow_iso()),
    )
    return int(cur.lastrowid)


def update_goal(
    conn: sqlite3.Connection, goal_id: int, settings: Settings | None = None, **fields
) -> None:
    settings = settings or get_settings()
    allowed = {"title", "description", "target_date", "priority", "status"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    if "title" in sets or "description" in sets:
        row = conn.execute("SELECT title, description FROM goals WHERE id=?", (goal_id,)).fetchone()
        title = sets.get("title", row["title"])
        desc = sets.get("description", row["description"])
        sets["embedding"] = _embed(title, desc, settings)
    sets["updated_at"] = utcnow_iso()
    cols = ", ".join(f"{k}=?" for k in sets)
    conn.execute(f"UPDATE goals SET {cols} WHERE id=?", (*sets.values(), goal_id))


def set_status(conn: sqlite3.Connection, goal_id: int, status: str) -> None:
    conn.execute(
        "UPDATE goals SET status=?, updated_at=? WHERE id=?", (status, utcnow_iso(), goal_id)
    )


def mark_progress(conn: sqlite3.Connection, goal_id: int, when: str | None = None) -> None:
    conn.execute("UPDATE goals SET last_progress_at=? WHERE id=?", (when or utcnow_iso(), goal_id))


def delete_goal(conn: sqlite3.Connection, goal_id: int) -> None:
    conn.execute("DELETE FROM goals WHERE id=?", (goal_id,))


def list_goals(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    if status:
        rows = conn.execute(
            "SELECT * FROM goals WHERE status=? ORDER BY priority, target_date", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM goals ORDER BY status, priority, target_date").fetchall()
    return [dict(r) for r in rows]


def get_goal(conn: sqlite3.Connection, goal_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
    if row is None:
        return None
    links = conn.execute(
        "SELECT kind, ref_id, relation, score FROM goal_links WHERE goal_id=? ORDER BY score DESC",
        (goal_id,),
    ).fetchall()
    return {"goal": dict(row), "links": [dict(link) for link in links]}
