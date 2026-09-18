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
from datetime import UTC, datetime
from pathlib import Path

from secondbrain.config import REPO_ROOT, Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.storage import schema
from secondbrain.storage.db import DB_ERRORS


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
        _orphan_chunks(conn),
        _stuck_conversations(conn),
        _wal(conn),
        _integrity(conn, settings),
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
    except DB_ERRORS:
        ver = None
    if ver == schema.SCHEMA_VERSION:
        return RepairAction("schema", False, f"at head ({ver})")
    schema.apply_base_schema(conn)  # idempotent forward upgrade + re-stamp
    return RepairAction("schema", True, f"upgraded {ver} → {schema.SCHEMA_VERSION}")


def _local_config(root: Path | None = None) -> RepairAction:
    # Resolve against the repo root, not the CWD — launchd/cron may start us
    # anywhere, and seeding config.local.toml into a random directory is useless.
    root = root or REPO_ROOT
    example = root / "config.local.toml.example"
    local = root / "config.local.toml"
    if local.exists() or not example.exists():
        return RepairAction("config", False, "present" if local.exists() else "no template")
    local.write_text(example.read_text())
    return RepairAction("config", True, "seeded config.local.toml from template")


def _stale_jobs(conn: sqlite3.Connection) -> RepairAction:
    n = q.reclaim_stale(conn)  # 'running' jobs from a dead worker → 'pending'
    return RepairAction("stale jobs", n > 0, f"re-queued {n} crashed job(s)" if n else "none")


def _orphan_chunks(conn: sqlite3.Connection) -> RepairAction:
    """Re-enqueue 'recorded' chunks whose transcription job vanished.

    A chunk registered right before a crash (or whose job was pruned/lost) sits
    in status 'recorded' forever with nothing queued for it. Re-enqueueing is
    safe: transcription is idempotent and enqueue dedupes on audio_file_id.
    """
    from secondbrain.pipeline.worker import JOB_TRANSCRIBE, enqueue_transcription

    rows = conn.execute(
        """
        SELECT af.id FROM audio_files af
        WHERE af.status = 'recorded'
          AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.type = ? AND j.state IN ('pending', 'running')
              AND json_extract(j.payload, '$.audio_file_id') = af.id
          )
        """,
        (JOB_TRANSCRIBE,),
    ).fetchall()
    for r in rows:
        enqueue_transcription(conn, r["id"])
    n = len(rows)
    return RepairAction(
        "orphan chunks", n > 0,
        f"re-enqueued {n} recorded chunk(s) with no job" if n else "none",
    )


def _stuck_conversations(conn: sqlite3.Connection) -> RepairAction:
    """Re-enqueue diarization for conversations stuck mid-pipeline with no job.

    'closed' means a diarize job was enqueued (it may have been lost);
    'diarizing' means a worker started and died (or the job dead-lettered).
    Either way, with no pending/running diarize job the conversation would sit
    there forever. Re-enqueueing is safe: attribution re-runs from scratch.
    """
    from secondbrain.pipeline.conversation import JOB_DIARIZE

    rows = conn.execute(
        """
        SELECT c.id FROM conversations c
        WHERE c.status IN ('closed', 'diarizing')
          AND NOT EXISTS (
            SELECT 1 FROM jobs j
            WHERE j.type = ? AND j.state IN ('pending', 'running')
              AND json_extract(j.payload, '$.conversation_id') = c.id
          )
        """,
        (JOB_DIARIZE,),
    ).fetchall()
    for r in rows:
        q.enqueue(conn, JOB_DIARIZE, {"conversation_id": r["id"]},
                  dedupe_key="conversation_id")
    n = len(rows)
    return RepairAction(
        "stuck conversations", n > 0,
        f"re-enqueued diarization for {n} conversation(s)" if n else "none",
    )


def _wal(conn: sqlite3.Connection) -> RepairAction:
    # Routine maintenance (shrinks the WAL sidecar); not counted as an "issue fixed".
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return RepairAction("wal", False, "checkpointed")
    except DB_ERRORS as exc:
        return RepairAction("wal", False, str(exc))


def _integrity(conn: sqlite3.Connection, settings: Settings | None = None) -> RepairAction:
    try:
        res = conn.execute("PRAGMA quick_check").fetchone()[0]
    except DB_ERRORS as exc:
        return RepairAction("integrity", False, str(exc), ok=False)
    if res == "ok":
        return RepairAction("integrity", False, "ok")
    # Don't guess at corruption — surface it (with the best restore candidate)
    # so the user can `sb restore`.
    hint = "no backups found — check `sb backups`"
    try:
        from secondbrain.storage import backup

        snaps = backup.list_backups(settings)
        if snaps:
            newest = snaps[0]
            age_days = (
                datetime.now(UTC) - datetime.fromisoformat(newest["modified"])
            ).days
            hint = f"newest backup: {newest['path']} ({age_days}d old)"
    except Exception:  # noqa: BLE001 - the hint must never mask the corruption report
        pass
    return RepairAction(
        "integrity", False, f"CORRUPT: {res} — restore a backup ({hint})", ok=False
    )
