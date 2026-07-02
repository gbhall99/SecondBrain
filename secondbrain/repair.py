"""Self-healing: detect and auto-fix common problems, safely and idempotently.

Only performs SAFE remediations — create missing directories, bring the schema to
head, seed the local config, re-queue crashed jobs, checkpoint a bloated WAL. It
never deletes user data; genuine corruption is reported (for a restore), not
"fixed". Used by ``sb repair``, ``deploy/install.sh``, and daemon startup so the
system heals itself on every run instead of failing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.storage import schema


@dataclass
class RepairAction:
    name: str
    fixed: bool          # True → a problem was found and repaired
    detail: str = ""
    ok: bool = True       # False → a problem remains that repair can't safely fix


def repair(conn: sqlite3.Connection, settings: Settings | None = None) -> list[RepairAction]:
    """Run all safe self-heal steps; return what was checked/fixed."""
    settings = settings or get_settings()
    return [
        _dirs(settings),
        _schema(conn),
        _local_config(),
        _stale_jobs(conn),
        _wal(conn),
        _integrity(conn),
    ]


def _dirs(settings: Settings) -> RepairAction:
    missing = [
        d for d in (settings.data_path, settings.audio_raw_dir,
                    settings.audio_processed_dir, settings.models_dir)
        if not d.exists()
    ]
    settings.ensure_dirs()
    if missing:
        return RepairAction("data dirs", True, f"created {len(missing)} missing dir(s)")
    return RepairAction("data dirs", False, "present")


def _schema(conn: sqlite3.Connection) -> RepairAction:
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        ver = row["version_num"] if row else None
    except sqlite3.Error:
        ver = None
    if ver == schema.SCHEMA_VERSION:
        return RepairAction("schema", False, f"at head ({ver})")
    schema.apply_base_schema(conn)  # idempotent forward upgrade + re-stamp
    return RepairAction("schema", True, f"upgraded {ver} → {schema.SCHEMA_VERSION}")


def _local_config() -> RepairAction:
    example = Path("config.local.toml.example")
    local = Path("config.local.toml")
    if local.exists() or not example.exists():
        return RepairAction("config", False, "present" if local.exists() else "no template")
    local.write_text(example.read_text())
    return RepairAction("config", True, "seeded config.local.toml from template")


def _stale_jobs(conn: sqlite3.Connection) -> RepairAction:
    n = q.reclaim_stale(conn)  # 'running' jobs from a dead worker → 'pending'
    return RepairAction("stale jobs", n > 0, f"re-queued {n} crashed job(s)" if n else "none")


def _wal(conn: sqlite3.Connection) -> RepairAction:
    # Routine maintenance (shrinks the WAL sidecar); not counted as an "issue fixed".
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return RepairAction("wal", False, "checkpointed")
    except sqlite3.Error as exc:
        return RepairAction("wal", False, str(exc))


def _integrity(conn: sqlite3.Connection) -> RepairAction:
    try:
        res = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.Error as exc:
        return RepairAction("integrity", False, str(exc), ok=False)
    if res == "ok":
        return RepairAction("integrity", False, "ok")
    # Don't guess at corruption — surface it so the user can `sb restore`.
    return RepairAction("integrity", False, f"CORRUPT: {res} — restore a backup", ok=False)
