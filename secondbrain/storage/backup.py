"""Local backup & export — portable snapshots of the user's data.

- ``backup_database`` makes a consistent SQLite snapshot via the online backup
  API (safe with WAL, unlike a file copy).
- ``export_json`` / ``export_markdown`` write portable dumps (transcripts, graph,
  goals, tasks). Opted-out speakers' segments are excluded.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry
from secondbrain.storage.db import connect


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def backup_database(settings: Settings | None = None, dest: Path | None = None) -> Path:
    """Write a consistent snapshot of the database to ``dest``. Returns the path."""
    settings = settings or get_settings()
    dest = Path(dest) if dest else settings.data_path / "backups" / f"secondbrain-{_stamp()}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = connect(settings=settings)
    try:
        out = sqlite3.connect(str(dest))
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    return dest


def _non_opted_segments(conn: sqlite3.Connection, settings: Settings) -> list[dict]:
    opted = registry.opted_out_speaker_ids(conn, settings)
    rows = conn.execute(
        """
        SELECT ts.id, ts.start_at, ts.text, ts.speaker_id, ts.speaker_confidence,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        ORDER BY ts.start_at, ts.id
        """
    ).fetchall()
    out = []
    for r in rows:
        if r["speaker_id"] in opted:
            continue
        d = dict(r)
        d["speaker"] = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        out.append(d)
    return out


def export_json(conn: sqlite3.Connection, out_dir: Path, settings: Settings | None = None) -> Path:
    """Dump transcripts/graph/goals/tasks to a single JSON file (no embeddings)."""
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "exported_at": datetime.now(UTC).isoformat(),
        "speakers": [
            dict(r) for r in conn.execute(
                "SELECT id, name, display_label, kind, is_owner FROM speakers "
                "WHERE merged_into IS NULL"
            ).fetchall()
        ],
        "segments": [
            {k: s[k] for k in ("id", "start_at", "speaker", "text", "speaker_confidence")}
            for s in _non_opted_segments(conn, settings)
        ],
        "kg_nodes": [
            dict(r) for r in conn.execute(
                "SELECT id, type, name FROM kg_nodes WHERE merged_into IS NULL"
            ).fetchall()
        ],
        "kg_edges": [
            dict(r) for r in conn.execute(
                "SELECT src_node_id, dst_node_id, predicate, kind, object_text "
                "FROM kg_edges WHERE valid=1"
            ).fetchall()
        ],
        "goals": [dict(r) for r in conn.execute(
            "SELECT id, title, description, status, priority, target_date FROM goals"
        ).fetchall()],
        "tasks": [dict(r) for r in conn.execute(
            "SELECT id, goal_id, title, status, due_date, estimate_minutes FROM tasks"
        ).fetchall()],
    }
    path = out_dir / f"secondbrain-export-{_stamp()}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def export_markdown(
    conn: sqlite3.Connection, out_dir: Path, settings: Settings | None = None
) -> Path:
    """Write a human-readable Markdown export (daily transcripts + goals + tasks)."""
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = _non_opted_segments(conn, settings)
    lines = ["# SecondBrain export", ""]
    current_day = None
    for s in segs:
        day = (s["start_at"] or "")[:10]
        if day != current_day:
            current_day = day
            lines.append(f"\n## {day or 'Undated'}\n")
        when = (s["start_at"] or "")[11:19]
        lines.append(f"- **{s['speaker']}** ({when}): {s['text']}")
    goals = conn.execute("SELECT title, status, priority FROM goals ORDER BY priority").fetchall()
    if goals:
        lines.append("\n## Goals\n")
        lines += [f"- [{g['status']}] P{g['priority']} {g['title']}" for g in goals]
    tasks = conn.execute(
        "SELECT title, status, due_date FROM tasks WHERE status NOT IN ('done','dropped')"
    ).fetchall()
    if tasks:
        lines.append("\n## Open tasks\n")
        lines += [f"- {t['title']}" + (f" (due {t['due_date']})" if t["due_date"] else "")
                  for t in tasks]
    path = out_dir / f"secondbrain-export-{_stamp()}.md"
    path.write_text("\n".join(lines) + "\n")
    return path
