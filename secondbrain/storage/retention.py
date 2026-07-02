"""Raw-audio retention and disk guardrails (consent/privacy controls).

Transcripts are kept indefinitely; raw audio (the sensitive voiceprint-bearing
artifact) is deleted after a configurable window once it has been transcribed.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from secondbrain.config import Settings, get_settings
from secondbrain.storage.models import iso_from_dt

log = logging.getLogger(__name__)

# Safety net: a diarization-deferred chunk (retention_delete_after NULL) whose
# conversation never finishes diarizing would otherwise keep its raw audio forever.
# Once it is older than (retention window + this grace), force-expire it so raw
# audio can never outlive the policy even when diarization stalls.
ORPHAN_GRACE_HOURS = 72


def compute_delete_after(settings: Settings, transcribed_at: datetime | None = None) -> str | None:
    """Deadline after which raw audio may be deleted.

    Returns an ISO timestamp, or None to keep indefinitely (negative config).
    A zero window yields "now" (delete on next sweep).
    """
    hours = settings.consent.raw_audio_retention_hours
    if hours < 0:
        return None
    base = transcribed_at or datetime.now(UTC)
    return iso_from_dt(base + timedelta(hours=hours))


def sweep_expired_audio(conn: sqlite3.Connection, settings: Settings | None = None) -> int:
    """Delete raw audio files whose retention deadline has passed.

    Only deletes files that have been transcribed. Marks rows as ``deleted`` and
    removes the file from disk. Returns the number of files deleted.
    """
    settings = settings or get_settings()
    now_dt = datetime.now(UTC)
    now = iso_from_dt(now_dt)
    rows = list(
        conn.execute(
            """
            SELECT id, path FROM audio_files
            WHERE status = 'transcribed'
              AND retention_delete_after IS NOT NULL
              AND retention_delete_after <= ?
            """,
            (now,),
        ).fetchall()
    )
    # Safety net for diarization-deferred chunks (NULL deadline) whose conversation
    # never diarized. NULL also legitimately means "keep forever" when retention is
    # negative, so only force-expire when retention is finite.
    hours = settings.consent.raw_audio_retention_hours
    if hours >= 0:
        cutoff = iso_from_dt(now_dt - timedelta(hours=hours + ORPHAN_GRACE_HOURS))
        orphans = conn.execute(
            """
            SELECT id, path FROM audio_files
            WHERE status = 'transcribed'
              AND retention_delete_after IS NULL
              AND started_at IS NOT NULL
              AND started_at <= ?
            """,
            (cutoff,),
        ).fetchall()
        if orphans:
            log.warning(
                "retention: force-expiring %d deferred chunk(s) whose diarization "
                "never finalized (older than retention + %dh grace)",
                len(orphans), ORPHAN_GRACE_HOURS,
            )
        rows += orphans
    deleted = 0
    for r in rows:
        p = Path(r["path"])
        try:
            if p.exists():
                p.unlink()
        except OSError:
            log.warning("retention: could not delete %s", p, exc_info=True)
            continue
        conn.execute("UPDATE audio_files SET status='deleted' WHERE id=?", (r["id"],))
        deleted += 1
    return deleted


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def disk_ok(settings: Settings | None = None) -> bool:
    """True if free disk space is above the configured guardrail."""
    settings = settings or get_settings()
    target = settings.data_path
    target.mkdir(parents=True, exist_ok=True)
    return free_disk_gb(target) >= settings.capture.min_free_disk_gb
