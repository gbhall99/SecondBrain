"""Local backup & export — portable snapshots of the user's data.

- ``backup_database`` makes a consistent SQLite snapshot via the online backup
  API (safe with WAL, unlike a file copy).
- ``export_json`` / ``export_markdown`` write portable dumps (transcripts, graph,
  goals, tasks). Opted-out speakers' segments are excluded.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.speaker import registry
from secondbrain.storage.db import connect


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


class BackupError(Exception):
    """Raised when a snapshot fails post-write verification."""


def _verify_snapshot(path: Path, settings: Settings) -> None:
    """Run SQLite's integrity checks on a freshly-written snapshot.

    ``quick_check`` covers page/btree corruption at a fraction of the cost of a
    full ``integrity_check``; a backup of an already-corrupt live DB (or a
    truncated write) fails here instead of being discovered at restore time.
    Raises :class:`BackupError` on any problem.
    """
    try:
        c = connect(path, settings=settings)
    except Exception as exc:  # noqa: BLE001 - unreadable snapshot is a failed backup
        raise BackupError(f"snapshot unreadable: {exc}") from exc
    try:
        res = c.execute("PRAGMA quick_check").fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"snapshot verification failed: {exc}") from exc
    finally:
        c.close()
    if res != "ok":
        raise BackupError(f"snapshot failed integrity check: {res}")


def backup_database(settings: Settings | None = None, dest: Path | None = None) -> Path:
    """Write a consistent snapshot of the database to ``dest``. Returns the path.

    The snapshot is verified (PRAGMA quick_check) after writing; on failure it
    is removed and :class:`BackupError` is raised, so a corrupt snapshot is
    never left in place masquerading as a good backup.
    """
    settings = settings or get_settings()
    dest = Path(dest) if dest else settings.data_path / "backups" / f"secondbrain-{_stamp()}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = connect(settings=settings)
    try:
        # Open the destination with the same driver + key as the live DB, so an
        # encrypted DB produces an encrypted snapshot (not a silent plaintext leak).
        out = connect(dest, settings=settings)
        try:
            src.backup(out)
        finally:
            out.close()
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        src.close()
    try:
        _verify_snapshot(dest, settings)
    except BackupError:
        dest.unlink(missing_ok=True)
        raise
    return dest


def list_backups(settings: Settings | None = None) -> list[dict]:
    """List backup snapshots, newest first: {name, path, size_bytes, modified}."""
    settings = settings or get_settings()
    backups_dir = settings.data_path / "backups"
    if not backups_dir.is_dir():
        return []
    out = []
    for p in sorted(backups_dir.glob("secondbrain-*.db"), reverse=True):
        st = p.stat()
        out.append(
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
            }
        )
    return out


def prune_backups(settings: Settings | None = None, keep: int = 10) -> int:
    """Keep the newest ``keep`` backup snapshots; delete older ones. Returns count removed.

    Operates on ``<data>/backups/secondbrain-*.db``, newest by filename
    timestamp. ``*-pre-restore.db`` safety snapshots (taken automatically before
    a restore) are never deleted. ``keep <= 0`` is a no-op guard so an
    accidental 0 never wipes every backup.
    """
    settings = settings or get_settings()
    if keep <= 0:
        return 0
    backups_dir = settings.data_path / "backups"
    if not backups_dir.is_dir():
        return 0
    snaps = sorted(
        (p for p in backups_dir.glob("secondbrain-*.db")
         if not p.name.endswith("-pre-restore.db")),
        reverse=True,
    )
    removed = 0
    for old in snaps[keep:]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


class RestoreError(Exception):
    """Raised when a restore source isn't a usable SecondBrain database."""


def _is_secondbrain_db(path: Path, settings: Settings) -> bool:
    """A readable SecondBrain DB. Opened with the configured driver + key, so an
    encrypted snapshot validates (and a wrong key / plaintext file is rejected)."""
    try:
        c = connect(path, settings=settings)
    except Exception:  # noqa: BLE001 - unreadable / wrong key / not a DB
        return False
    try:
        names = {
            r[0]
            for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    except Exception:  # noqa: BLE001 - SQLCipher raises on a bad key here
        return False
    finally:
        c.close()
    return {"transcript_segments", "speakers"}.issubset(names)


def restore_database(
    settings: Settings | None = None, src: Path | str | None = None, *, backup_current: bool = True
) -> Path:
    """Replace the live database with the snapshot at ``src``.

    Validates ``src`` is a real SecondBrain DB first. By default the current
    database is snapshotted to a timestamped ``*-pre-restore.db`` before being
    replaced, so a restore is itself reversible. Run with the daemon stopped.
    Returns the restored database path.
    """
    settings = settings or get_settings()
    if src is None:
        raise RestoreError("no restore source provided")
    src = Path(src)
    if not src.exists():
        raise RestoreError(f"restore source not found: {src}")
    if not _is_secondbrain_db(src, settings):
        raise RestoreError(f"not a SecondBrain database: {src}")

    db_path = settings.db_path
    if backup_current and db_path.exists():
        backup_database(
            settings=settings,
            dest=db_path.parent / "backups" / f"secondbrain-{_stamp()}-pre-restore.db",
        )

    # Atomic replacement: write a clean single-file copy of the snapshot into a
    # temp file NEXT TO the live DB (same filesystem), then os.replace() it into
    # place — a crash mid-restore never leaves a missing/half-written live DB.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.parent / (db_path.name + ".restore-tmp")
    for suffix in ("", "-wal", "-shm"):  # clear leftovers from a crashed restore
        Path(str(tmp) + suffix).unlink(missing_ok=True)
    # Both sides via the configured driver + key, so an encrypted snapshot restores
    # to an encrypted live DB (and never writes an unreadable plaintext file).
    source = connect(src, settings=settings)
    try:
        out = connect(tmp, settings=settings)
        try:
            source.backup(out)
        finally:
            out.close()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        source.close()
    # Drop stale WAL sidecars (they belong to the old DB) before swapping in.
    for suffix in ("-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    os.replace(tmp, db_path)
    return db_path


def _non_opted_segments(
    conn: sqlite3.Connection,
    settings: Settings,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    opted = registry.opted_out_speaker_ids(conn, settings)
    where, params = [], []
    # Undated segments (NULL start_at) can't be placed in a date window; retain them
    # so a filtered export doesn't silently drop data the unfiltered path keeps.
    if since:
        where.append("(ts.start_at IS NULL OR substr(ts.start_at, 1, 10) >= ?)")
        params.append(since)
    if until:
        where.append("(ts.start_at IS NULL OR substr(ts.start_at, 1, 10) <= ?)")
        params.append(until)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"""
        SELECT ts.id, ts.start_at, ts.text, ts.speaker_id, ts.speaker_confidence,
               COALESCE(sp.name, sp.display_label) AS speaker,
               CASE WHEN sp.is_owner THEN 1 ELSE 0 END AS is_owner
        FROM transcript_segments ts
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        {clause}
        ORDER BY ts.start_at, ts.id
        """,
        params,
    ).fetchall()
    out = []
    for r in rows:
        if r["speaker_id"] in opted:
            continue
        d = dict(r)
        d["speaker"] = "Me" if r["is_owner"] else (r["speaker"] or "Unknown")
        out.append(d)
    return out


def export_json(
    conn: sqlite3.Connection,
    out_dir: Path,
    settings: Settings | None = None,
    since: str | None = None,
    until: str | None = None,
) -> Path:
    """Dump transcripts/graph/goals/tasks to a single JSON file (no embeddings)."""
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "exported_at": datetime.now(UTC).isoformat(),
        "since": since,
        "until": until,
        "speakers": [
            dict(r) for r in conn.execute(
                "SELECT id, name, display_label, kind, is_owner FROM speakers "
                "WHERE merged_into IS NULL"
            ).fetchall()
        ],
        "segments": [
            {k: s[k] for k in ("id", "start_at", "speaker", "text", "speaker_confidence")}
            for s in _non_opted_segments(conn, settings, since, until)
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
    conn: sqlite3.Connection,
    out_dir: Path,
    settings: Settings | None = None,
    since: str | None = None,
    until: str | None = None,
) -> Path:
    """Write a human-readable Markdown export (daily transcripts + goals + tasks)."""
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = _non_opted_segments(conn, settings, since, until)
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
