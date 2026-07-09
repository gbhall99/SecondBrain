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


# The raw float32 `embedding` BLOB is deliberately excluded from every read:
# it isn't JSON-serializable (FastAPI's encoder raises UnicodeDecodeError on
# it) and no caller needs it — linking reads the column directly in link.py.
_GOAL_COLS = (
    "id", "title", "description", "target_date", "priority", "status",
    "last_progress_at", "created_at", "updated_at",
)

# Per-goal task progress; dropped tasks no longer count toward the plan.
_TASK_STATS = (
    "SELECT goal_id, COUNT(*) AS total, SUM(status='done') AS done "
    "FROM tasks WHERE goal_id IS NOT NULL AND status != 'dropped' GROUP BY goal_id"
)

# Active work first, then paused, done, dropped (blind ORDER BY status would
# bury paused goals below done ones alphabetically).
_STATUS_RANK = (
    "CASE g.status WHEN 'active' THEN 0 WHEN 'paused' THEN 1 WHEN 'done' THEN 2 ELSE 3 END"
)


def list_goals(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    """Goals (sans embedding blob) with task progress counts, in display order:
    status rank, then priority, then target date with undated goals last.
    ``links_count`` (additive) lets the UI surface auto-linking evidence."""
    where = "WHERE g.status=?" if status else ""
    params = (status,) if status else ()
    cols = ", ".join(f"g.{c}" for c in _GOAL_COLS)
    rows = conn.execute(
        f"SELECT {cols}, COALESCE(t.total, 0) AS tasks_total, COALESCE(t.done, 0) AS tasks_done, "
        f"(SELECT COUNT(*) FROM goal_links gl WHERE gl.goal_id = g.id) AS links_count "
        f"FROM goals g LEFT JOIN ({_TASK_STATS}) t ON t.goal_id = g.id {where} "
        f"ORDER BY {_STATUS_RANK}, g.priority, g.target_date IS NULL, g.target_date, g.id",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """How many goals per status (all four keys present), plus 'total'."""
    counts = {"active": 0, "paused": 0, "done": 0, "dropped": 0}
    for r in conn.execute("SELECT status, COUNT(*) AS n FROM goals GROUP BY status").fetchall():
        if r["status"] in counts:
            counts[r["status"]] = int(r["n"])
    counts["total"] = sum(counts.values())
    return counts


def get_goal(conn: sqlite3.Connection, goal_id: int) -> dict | None:
    row = conn.execute(
        f"SELECT {', '.join(_GOAL_COLS)} FROM goals WHERE id=?", (goal_id,)
    ).fetchone()
    if row is None:
        return None
    goal = dict(row)
    stats = conn.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(status='done'), 0) AS done "
        "FROM tasks WHERE goal_id=? AND status != 'dropped'",
        (goal_id,),
    ).fetchone()
    goal["tasks_total"], goal["tasks_done"] = int(stats["total"]), int(stats["done"])
    # Resolve display info for each link (additive keys — kind/ref_id/relation/
    # score are contract): label + ref_type make the evidence human-readable,
    # src_node_id lets the UI deep-link an edge via its source entity. A link
    # whose target was merged/forgotten resolves to a NULL label; the UI is
    # expected to present those honestly rather than hide them.
    links = conn.execute(
        """
        SELECT gl.kind, gl.ref_id, gl.relation, gl.score,
               CASE gl.kind
                    WHEN 'node' THEN COALESCE(n.display_label, n.name)
                    ELSE e.object_text END AS label,
               CASE gl.kind WHEN 'node' THEN n.type ELSE e.kind END AS ref_type,
               CASE gl.kind WHEN 'edge' THEN e.src_node_id END AS src_node_id
        FROM goal_links gl
        LEFT JOIN kg_nodes n ON gl.kind = 'node' AND n.id = gl.ref_id
        LEFT JOIN kg_edges e ON gl.kind = 'edge' AND e.id = gl.ref_id
        WHERE gl.goal_id = ?
        ORDER BY gl.score DESC, gl.id
        """,
        (goal_id,),
    ).fetchall()
    return {"goal": goal, "links": [dict(link) for link in links]}
