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

# Chunks that never reach 'transcribed' (dead-lettered 'failed', or 'recorded'
# ones whose job vanished) must not keep raw audio forever either. 'recorded'
# gets a generous grace so a long offline backlog is never deleted before the
# worker catches up; transcripts are unaffected (there are none for these).
STUCK_RECORDED_GRACE_DAYS = 7

# Diarization scratch files (conv_concat*.wav) older than this are swept.
SCRATCH_MAX_AGE_HOURS = 24


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
        # Dead-lettered chunks: 'failed' past retention+grace, and 'recorded'
        # past a generous grace (their job is gone — the audio would sit forever).
        stuck_failed = conn.execute(
            "SELECT id, path FROM audio_files WHERE status='failed' "
            "AND started_at IS NOT NULL AND started_at <= ?",
            (cutoff,),
        ).fetchall()
        recorded_cutoff = iso_from_dt(
            now_dt - timedelta(
                hours=max(hours + ORPHAN_GRACE_HOURS, STUCK_RECORDED_GRACE_DAYS * 24)
            )
        )
        stuck_recorded = conn.execute(
            "SELECT id, path FROM audio_files WHERE status='recorded' "
            "AND started_at IS NOT NULL AND started_at <= ?",
            (recorded_cutoff,),
        ).fetchall()
        if stuck_failed or stuck_recorded:
            log.warning(
                "retention: expiring %d failed and %d stuck-recorded chunk(s) "
                "that never transcribed",
                len(stuck_failed), len(stuck_recorded),
            )
        rows += stuck_failed + stuck_recorded
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
    _sweep_derived_clips(conn, settings)
    return deleted


def _sweep_derived_clips(conn: sqlite3.Connection, settings: Settings) -> int:
    """Delete cached audio clips whose source raw audio is gone.

    The web clip players cache slices in ``audio_processed_dir`` as
    ``sample_{observation_id}[_{window}].wav`` (People page voice samples) and
    ``segclip_{segment_id}[_{window}].wav`` (day-view per-line playback), where
    the optional ``_{window}`` suffix stamps the sliced offsets; both are
    derived raw audio and must honor the same retention policy as their source
    chunk. Stale diarization scratch files (``conv_concat*.wav``) older than a
    day are swept too — they're concatenated raw audio a crashed diarization
    can leave behind.
    """
    removed = 0
    clip_dir = settings.audio_processed_dir
    if not clip_dir.is_dir():
        return 0
    scratch_cutoff = (
        datetime.now(UTC) - timedelta(hours=SCRATCH_MAX_AGE_HOURS)
    ).timestamp()
    for p in clip_dir.glob("conv_concat*.wav"):
        try:
            if p.stat().st_mtime > scratch_cutoff:
                continue  # possibly still in use by a running diarization
            p.unlink(missing_ok=True)
            removed += 1
        except OSError:
            log.warning("retention: could not delete scratch file %s", p, exc_info=True)
    source_status_sql = {
        "sample": (
            "SELECT af.status AS status FROM speaker_observations so "
            "JOIN audio_files af ON af.id = so.audio_file_id WHERE so.id = ?"
        ),
        "segclip": (
            "SELECT af.status AS status FROM transcript_segments ts "
            "JOIN audio_files af ON af.id = ts.audio_file_id WHERE ts.id = ?"
        ),
    }
    for prefix, sql in source_status_sql.items():
        for p in clip_dir.glob(f"{prefix}_*.wav"):
            try:
                # sample_12.wav and sample_12_50-1050.wav both belong to obs 12.
                source_id = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue  # not one of ours
            row = conn.execute(sql, (source_id,)).fetchone()
            if row is not None and row["status"] != "deleted":
                continue  # source audio still lives — keep the cached clip
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except OSError:
                log.warning("retention: could not delete derived clip %s", p, exc_info=True)
    if removed:
        log.info("retention: deleted %d derived audio clip(s)", removed)
    return removed


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def disk_ok(settings: Settings | None = None) -> bool:
    """True if free disk space is above the configured guardrail."""
    settings = settings or get_settings()
    target = settings.data_path
    target.mkdir(parents=True, exist_ok=True)
    return free_disk_gb(target) >= settings.capture.min_free_disk_gb
