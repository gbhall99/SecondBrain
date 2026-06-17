"""Backend-agnostic query helpers shared by the API, CLI, and (later) chat."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import queue as q
from secondbrain.search import combined
from secondbrain.speaker import registry
from secondbrain.storage import retention, state
from secondbrain.storage.models import segments_for_day


def _speaker_label(row: sqlite3.Row | None) -> str | None:
    if row is None:
        return None
    return row["name"] or row["display_label"] or f"Speaker {row['id']}"


def _speaker_labels(conn: sqlite3.Connection, segment_ids: list[int]) -> dict[int, dict]:
    """Resolve speaker name + low-confidence flag for the given segment ids."""
    if not segment_ids:
        return {}
    placeholders = ",".join("?" * len(segment_ids))
    rows = conn.execute(
        f"""
        SELECT ts.id AS seg_id, ts.speaker_confidence AS conf,
               sp.id, sp.name, sp.display_label
        FROM transcript_segments ts
        LEFT JOIN speakers sp ON sp.id = ts.speaker_id
        WHERE ts.id IN ({placeholders})
        """,
        segment_ids,
    ).fetchall()
    low = get_settings().diarization.low_confidence_threshold
    out: dict[int, dict] = {}
    for r in rows:
        conf = r["conf"]
        out[r["seg_id"]] = {
            "speaker": _speaker_label(r) if r["id"] is not None else None,
            "speaker_confidence": conf,
            "speaker_low_confidence": conf is not None and conf < low,
        }
    return out


def search(conn: sqlite3.Connection, query: str, limit: int = 20, mode: str = "auto",
           settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    hits = combined.search(conn, query, limit, settings=settings, mode=mode)
    results = [asdict(h) for h in hits]
    labels = _speaker_labels(conn, [h["segment_id"] for h in results])
    for h in results:
        h.update(labels.get(h["segment_id"], {}))
    return results


def day_segments(conn: sqlite3.Connection, day: str | None = None) -> list[dict]:
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    rows = [dict(r) for r in segments_for_day(conn, day)]
    labels = _speaker_labels(conn, [r["id"] for r in rows])
    for r in rows:
        r.update(labels.get(r["id"], {}))
    return rows


def status(conn: sqlite3.Connection, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    seg_total = conn.execute("SELECT COUNT(*) AS n FROM transcript_segments").fetchone()["n"]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    today_segs = len(segments_for_day(conn, today))
    paused = state.is_paused(conn, default=settings.consent.paused)
    speakers_known = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind IN ('owner','known') AND merged_into IS NULL"
    ).fetchone()["n"]
    unknown_pending = conn.execute(
        "SELECT COUNT(*) AS n FROM speakers WHERE kind='unknown' AND merged_into IS NULL"
    ).fetchone()["n"]
    return {
        "recording_enabled": settings.consent.recording_enabled,
        "paused": paused,
        "recording": settings.consent.recording_enabled and not paused,
        "disk_free_gb": round(retention.free_disk_gb(settings.data_path), 2),
        "disk_ok": retention.disk_ok(settings),
        "jobs": q.counts(conn),
        "segments_total": seg_total,
        "segments_today": today_segs,
        "retention_hours": settings.consent.raw_audio_retention_hours,
        "diarization_enabled": settings.diarization.enabled,
        "speakers_known": speakers_known,
        "unknown_clusters_pending": unknown_pending,
    }


# --- speaker management (shared by CLI / API / web) --------------------------


def list_speakers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, display_label, kind, is_owner, opted_out, segment_count, "
        "last_seen_at FROM speakers WHERE merged_into IS NULL ORDER BY is_owner DESC, "
        "kind, segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def unknown_speakers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, display_label, segment_count, last_seen_at FROM speakers "
        "WHERE kind='unknown' AND merged_into IS NULL ORDER BY segment_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def speaker_samples(conn: sqlite3.Connection, speaker_id: int, n: int = 3) -> list[dict]:
    """Top observations (audio still on disk preferred) for clip playback."""
    rows = conn.execute(
        """
        SELECT so.id, so.audio_file_id, so.start_offset_s, so.end_offset_s, so.start_at,
               af.path, af.status AS audio_status
        FROM speaker_observations so
        JOIN audio_files af ON af.id = so.audio_file_id
        WHERE so.speaker_id = ?
        ORDER BY (af.status != 'deleted') DESC, so.confidence DESC
        LIMIT ?
        """,
        (resolve(conn, speaker_id), n),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve(conn: sqlite3.Connection, speaker_id: int) -> int:
    return registry.resolve_speaker_id(conn, speaker_id)


def name_speaker(conn: sqlite3.Connection, speaker_id: int, name: str,
                 settings: Settings | None = None) -> int:
    return registry.name_speaker(conn, speaker_id, name, settings)


def merge_speakers(conn: sqlite3.Connection, src: int, dst: int,
                   settings: Settings | None = None) -> int:
    return registry.merge_speakers(conn, src, dst, settings)


def set_owner(conn: sqlite3.Connection, speaker_id: int) -> None:
    """Mark an existing (history-discovered) speaker as the owner."""
    sid = registry.resolve_speaker_id(conn, speaker_id)
    conn.execute("UPDATE speakers SET is_owner=0 WHERE is_owner=1 AND id<>?", (sid,))
    conn.execute("UPDATE speakers SET is_owner=1, kind='owner' WHERE id=?", (sid,))
